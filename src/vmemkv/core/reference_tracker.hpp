// reference_tracker.hpp — Thread-local reference tracking for lock-free memory reclamation (SMR)
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vmemkv/core/spin_backoff.hpp>

namespace vmemkv {

inline constexpr size_t kCacheLineSize = 64;

// One ID per thread, shared by every tracker instance, so two threads never collide on a
// slot. Never reused after the thread exits; slots pair every acquire() with a release(),
// and this store's threads are pool-shaped, so the leak is bounded and harmless.
class GlobalThreadId {
 public:
  static auto get() noexcept -> size_t {
    thread_local const size_t thread_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    return thread_id;
  }

 private:
  static inline std::atomic<size_t> next_id_{0};
};

// Fixed chunked slot array for epoch/hazard-pointer reclamation without shared_ptr's
// cache bouncing. Slots are cacheline-aligned; chunks are append-only and never reclaimed
// while live, so reads of published chunks never race growth. Addressable thread-ID space
// is bounded at 64K (see kMaxChunks), ample for a pool-shaped thread population.
template <typename T>
class ThreadReferenceTracker {
 public:
  class Guard {
   public:
    Guard(const ThreadReferenceTracker &tracker, T val) noexcept : tracker_(tracker), val_(val) {
      tracker_.acquire(val_);
    }
    ~Guard() noexcept { tracker_.release(); }

    auto operator->() const noexcept -> T { return val_; }
    operator T() const noexcept { return val_; }

    Guard(const Guard &) = delete;
    auto operator=(const Guard &) -> Guard & = delete;

   private:
    const ThreadReferenceTracker &tracker_;
    T val_;
  };

  ThreadReferenceTracker() noexcept = default;

  ~ThreadReferenceTracker() noexcept {
    for (auto &chunk_ptr : chunks_) {
      delete[] chunk_ptr.load(std::memory_order_relaxed);
    }
  }

  ThreadReferenceTracker(const ThreadReferenceTracker &) = delete;
  auto operator=(const ThreadReferenceTracker &) -> ThreadReferenceTracker & = delete;

  // Registers the current thread's reference. seq_cst: paired with the waiters' seq_cst
  // loads, closing the store-buffering race where each side could observe the other's
  // pre-update value. See T2FlatFile::acquire_write_handle().
  void acquire(T val) const noexcept { slot_for_current_thread().store(val, std::memory_order_seq_cst); }

  // Clears the current thread's reference.
  void release() const noexcept { slot_for_current_thread().store(T{}, std::memory_order_release); }

  // Spins until no slot still references old_val. SpinBackoff (not bare yield): a real
  // deschedule is the only wait guaranteed to converge on a contended low-core machine.
  void wait_until_retired(T old_val) const noexcept {
    for_each_slot([&](std::atomic<T> &slot) {
      SpinBackoff backoff;
      while (slot.load(std::memory_order_seq_cst) == old_val) {
        backoff.wait();
      }
    });
  }

  // Spins until every slot cleared or advanced past target_epoch (epoch reclamation).
  // Same seq_cst hazard as acquire(): a registration racing this scan must not stay invisible.
  void wait_until_epoch(T target_epoch) const noexcept {
    for_each_slot([&](std::atomic<T> &slot) {
      SpinBackoff backoff;
      const auto start = std::chrono::steady_clock::now();
      auto last_warn = start;
      while (true) {
        T epoch_val = slot.load(std::memory_order_seq_cst);
        if (epoch_val == T{} || epoch_val >= target_epoch) {
          break;
        }
        backoff.wait();
        const auto now = std::chrono::steady_clock::now();
        if (now - last_warn >= std::chrono::seconds(2)) {
          last_warn = now;
          const auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
          std::cerr << "[wait_until_epoch] WAITING " << total_sec << "s for epoch " << target_epoch
                    << " (slot_val=" << epoch_val << ")\n";
        }
      }
    });
  }

 private:
  struct alignas(kCacheLineSize) AlignedSlot {
    std::atomic<T> value{T{}};
  };

  // 64 slots per chunk (one 4 KiB page); 1024 chunks bound the thread-ID space at 64K.
  // chunks_ itself costs 8 KiB of null pointers regardless of actual use.
  static constexpr size_t kChunkSize = 64;
  static constexpr size_t kMaxChunks = 1024;

  // Fast path (chunk exists): one atomic load, no lock. Growth never moves published
  // memory, so concurrent reads of other chunks stay race-free.
  auto slot_for_current_thread() const -> std::atomic<T> & {
    const size_t thread_id = GlobalThreadId::get();
    const size_t chunk_index = thread_id / kChunkSize;
    if (chunk_index >= kMaxChunks) {
      throw std::runtime_error("ThreadReferenceTracker: thread-ID space exhausted");
    }
    AlignedSlot *chunk = chunks_[chunk_index].load(std::memory_order_acquire);
    if (chunk == nullptr) {
      chunk = ensure_chunk(chunk_index);
    }
    return chunk[thread_id % kChunkSize].value;
  }

  // Double-checked allocation under the growth lock. A concurrent waiter missing a
  // just-published chunk is timing-only, never a correctness gap: an unregistered writer
  // cannot yet have observed the boundary the wait enforces.
  auto ensure_chunk(size_t chunk_index) const -> AlignedSlot * {
    const std::lock_guard<std::mutex> lock(growth_mutex_);
    AlignedSlot *chunk = chunks_[chunk_index].load(std::memory_order_relaxed);
    if (chunk == nullptr) {
      chunk = new AlignedSlot[kChunkSize];
      chunks_[chunk_index].store(chunk, std::memory_order_release);
      if (const size_t new_high = chunk_index + 1; new_high > high_chunk_.load(std::memory_order_relaxed)) {
        high_chunk_.store(new_high, std::memory_order_release);
      }
    }
    return chunk;
  }

  // Never-allocated chunks hold no live references, so they are skipped.
  template <typename Fn>
  void for_each_slot(Fn &&func) const {
    const size_t chunk_count = high_chunk_.load(std::memory_order_acquire);
    for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
      AlignedSlot *chunk = chunks_[chunk_idx].load(std::memory_order_acquire);
      if (chunk == nullptr) {
        continue;
      }
      for (size_t i = 0; i < kChunkSize; ++i) {
        func(chunk[i].value);
      }
    }
  }

  mutable std::mutex growth_mutex_;
  mutable std::array<std::atomic<AlignedSlot *>, kMaxChunks> chunks_{};
  mutable std::atomic<size_t> high_chunk_{0};
};

}  // namespace vmemkv

// hooks.hpp - Checkpoint test seams and the in-place-update barrier.
//
// Every seam is a nullary callable fired exactly once. Any CheckpointHook works, including
// lambdas; the NoOp structs are production defaults.
#pragma once

#include <concepts>
#include <vmemkv/core/reference_tracker.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>

namespace vmemkv {

template <typename Hook>
concept CheckpointHook = std::invocable<Hook &>;

// Fires inside checkpoint_internal() while writes still hand out normally. Lets a test park a
// writer's handle pre-stop, so the stop-and-wait has a real in-flight writer to drain. No-op
// in production.
struct NoOpPreStopHook {
  void operator()() const noexcept {}
};

// Fires inside checkpoint_internal() before T1 is ever touched. Lets a test throw here to
// verify T1 stays untouched (only the watermark rolls back). No-op in production.
struct NoOpPreFinishHook {
  void operator()() const {}
};

// Resumes writers on every checkpoint_internal() exit, including exception paths.
struct WriterResumeGuard {
  T2FlatFile *t2;
  ~WriterResumeGuard();
};

// Contract: past wait_until_retired(boundary), no in-place update targeting < boundary is
// still in flight. Registration tracks the actual offset (not a generic epoch), so the
// contract reads literally. Drains in-flight writes only; the caller separately ensures no
// *new* one can start below `boundary` (checkpoint uses the ownership watermark) before
// relying on it.
class InPlaceUpdateBarrier {
 public:
  using Guard = typename vmemkv::ThreadReferenceTracker<uint64_t>::Guard;

  // Registers an in-place write targeting `offset`. Hold through the update_value_at()
  // attempt; release as soon as decided, including non-writing early returns.
  [[nodiscard]] auto enter(uint64_t offset) const noexcept -> Guard {
    // +1: the tracker's T{} == 0 sentinel would collide with offset 0.
    return Guard{offsets_, offset + 1};
  }

  void wait_until_retired(uint64_t boundary) const noexcept { offsets_.wait_until_epoch(boundary + 1); }

 private:
  mutable vmemkv::ThreadReferenceTracker<uint64_t> offsets_;
};

inline WriterResumeGuard::~WriterResumeGuard() { t2->resume_writers(); }

}  // namespace vmemkv

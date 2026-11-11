// single_flight.hpp - Shared single-flight CAS guard plus flag guards.
#pragma once

#include <atomic>

namespace vmemkv {

// RAII single-flight claim on an atomic<bool> running flag.
// try_acquire() CASes false->true; the guard resets to false on destruction.
// Lets checkpoint/reorganize share one acquire/release pattern.
struct SingleFlightGuard {
  std::atomic<bool> *flag = nullptr;
  bool holds = false;

  static auto try_acquire(std::atomic<bool> &running) noexcept -> SingleFlightGuard {
    SingleFlightGuard out{&running, false};
    bool expected = false;
    out.holds = running.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    return out;
  }

  ~SingleFlightGuard() {
    if (holds && flag != nullptr) {
      flag->store(false, std::memory_order_release);
    }
  }

  SingleFlightGuard(const SingleFlightGuard &) = delete;
  auto operator=(const SingleFlightGuard &) -> SingleFlightGuard & = delete;
  SingleFlightGuard(SingleFlightGuard &&other) noexcept : flag(other.flag), holds(other.holds) {
    other.flag = nullptr;
    other.holds = false;
  }
  auto operator=(SingleFlightGuard &&other) noexcept -> SingleFlightGuard & {
    if (this != &other) {
      if (holds && flag != nullptr) {
        flag->store(false, std::memory_order_release);
      }
      flag = other.flag;
      holds = other.holds;
      other.flag = nullptr;
      other.holds = false;
    }
    return *this;
  }

 private:
  SingleFlightGuard(std::atomic<bool> *flag, bool holds) noexcept : flag(flag), holds(holds) {}
};

}  // namespace vmemkv

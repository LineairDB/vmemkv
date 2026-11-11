// background_poll.hpp - Poll intervals and worker skeleton for background maintenance.
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>

namespace vmemkv {

inline constexpr auto kDefaultPoll10ms = std::chrono::milliseconds(10);
inline constexpr auto kCompactionPoll1s = std::chrono::seconds(1);

// Background maintenance skeleton: calls work() until stop is requested, waiting `interval`
// after each unproductive round. work() reports whether it did anything (no wait after
// productive rounds keeps latency low under sustained pressure).
//
// The idle wait is condition_variable_any::wait_for() keyed on the stop token, not
// sleep_for(): jthread's destructor requests stop before joining, so a worker parked in the
// wait wakes at once and shutdown is not bounded by the interval.
template <typename Work>
inline void poll_worker_until_stop(const std::stop_token &stop, std::chrono::nanoseconds interval, Work &&work) {
  std::mutex wake_mutex;
  std::condition_variable_any wake_cv;
  while (!stop.stop_requested()) {
    if (!work()) {
      std::unique_lock wait_lock{wake_mutex};
      wake_cv.wait_for(wait_lock, stop, interval, [&stop] { return stop.stop_requested(); });
    }
  }
}

}  // namespace vmemkv

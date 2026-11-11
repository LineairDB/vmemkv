#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace vmemkv {

enum class ScanControl : std::uint8_t { Continue, Stop };

namespace scan_detail {

template <typename Callback, typename... Args>
inline auto invoke_callback(Callback &&callback, Args &&...args) -> ScanControl {
  using Result = std::invoke_result_t<Callback, Args...>;
  if constexpr (std::same_as<Result, void>) {
    std::invoke(std::forward<Callback>(callback), std::forward<Args>(args)...);
    return ScanControl::Continue;
  } else {
    static_assert(std::same_as<Result, ScanControl>, "scan callbacks must return void or ScanControl");
    return std::invoke(std::forward<Callback>(callback), std::forward<Args>(args)...);
  }
}

}  // namespace scan_detail
}  // namespace vmemkv

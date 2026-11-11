# VMemKV agent notes

- Format with `pip install clang-format==18.1.3` (pinned; must match CI — never the distro package, whose version drifts).
- Never diagnose a performance bottleneck, or propose a design change to fix one, from code-reading alone. Measure first.
- Tests are one binary per `tests/test_*.cpp`; run one area with `ctest -R <name>`, the full suite with `ctest -j "$(nproc)"`.
- Public headers are `include/vmemkv/vmemkv.hpp` plus `include/vmemkv/stats.hpp`; everything under `src/` is internal. See CONTRIBUTING.md for contribution rules.

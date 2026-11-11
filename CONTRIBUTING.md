# Contributing to vmemkv

This is a how-to guide for contributing code to vmemkv.

## How to build and test

See `README.md` Build/Test/Benchmark (build matrix, rival backends, `ctest -R` selection).

## How to format and lint

Install the pinned formatter (must match CI — never the distro package):

```bash
python3 -m pip install --user "clang-format==18.1.3"
```

```bash
find src tests benchmark include \( -name "*.cpp" -o -name "*.hpp" \) \
  -print0 | xargs -0 clang-format --dry-run --Werror
cmake --build build --target clang-tidy-check  # clang++ only
```

CI enforces the pinned formatter version.

## How to submit a change

1. Add or update tests under `tests/` for any behavior change.
2. Do not claim a performance fix from code reading alone: measure first and
   include numbers in the PR description.
3. Keep PRs small and focused; one logical change per PR.

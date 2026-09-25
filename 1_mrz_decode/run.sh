#!/usr/bin/env bash
# Convenience build & test runner.
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== unit tests ==="
./build/test_mrz
echo "=== ctest ==="
( cd build && ctest --output-on-failure )
echo "=== sample decode ==="
./build/mrz_tool decode "$(cat data/sample_td3.txt)"

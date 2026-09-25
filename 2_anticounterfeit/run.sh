#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== unit tests ==="
./build/test_ac
echo "=== ctest ==="
( cd build && ctest --output-on-failure )
echo "=== sample verify ==="
./build/ac_tool ../1_mrz_decode/data/sample_td3.txt

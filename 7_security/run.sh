#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== unit tests ==="
./build/test_security
echo "=== ctest ==="
( cd build && ctest --output-on-failure )
echo "=== sample BAC keys ==="
./build/security_tool bac L898902C3 690806 940623
echo
echo "=== sample cross-check (ERIKSSON) ==="
./build/security_tool crosscheck ../1_mrz_decode/data/sample_td3.txt "ERIKSSON ANNA MARIA, UTO, Passport L898902C3, Born 1969-08-06, Exp 1994-06-23"

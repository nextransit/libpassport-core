#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== sample: happy path ==="
./build/nfc_tool data/script_happy.json

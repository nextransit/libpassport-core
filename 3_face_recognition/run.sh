#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== unit tests ==="
./build/test_face
echo "=== ctest ==="
( cd build && ctest --output-on-failure )
echo "=== generate synthetic samples ==="
mkdir -p data
./build/gen_sample data/face_a.ppm 1   128 128
./build/gen_sample data/face_b.ppm 1   128 128
./build/gen_sample data/face_c.ppm 99  128 128
echo "=== compare face_a vs face_b (same seed) ==="
./build/face_tool data/face_a.ppm data/face_b.ppm 0
echo "=== compare face_a vs face_c (different seed) ==="
./build/face_tool data/face_a.ppm data/face_c.ppm 0

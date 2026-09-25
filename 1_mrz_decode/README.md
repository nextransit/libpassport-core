# 1_mrz_decode -- MRZ (Machine Readable Zone) TD3 decoder/encoder

Pure C99 library + CLI for encoding and decoding passport MRZ strings
following ICAO Doc 9303 Part 4 (TD3, 2 lines x 44 chars).

## Files

```
include/mrz.h        public API
src/mrz.c            implementation
src/main.c           CLI tool (encode / decode)
tests/test_mrz.c     standalone unit-test suite (89 assertions)
data/sample_td3.txt  canonical TD3 sample produced by this encoder
CMakeLists.txt       CMake build
run.sh               convenience build-and-test script
```

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

## Run tests

```sh
./build/test_mrz            # self-contained unit tests
cd build && ctest --output-on-failure
```

## CLI usage

```sh
# Encode a passport
./build/mrz_tool encode P< UTO ERIKSSON "ANNA MARIA" \
    L898902C3 UTO 690806 F 940623 "ZE184226B<<<<<"

# Decode a passport
./build/mrz_tool decode "$(cat data/sample_td3.txt)"
```

## Sample MRZ

The canonical sample in `data/sample_td3.txt` was produced by
`mrz_tool encode` and round-trips through the decoder without error:

```
P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<
L898902C36UTO6908061F9406236ZE184226B<<<<<18
```

(Where spaces in `given_names` are converted to `<` per ICAO 9303.)

## Check-digit algorithm

ICAO 9303 weight cycle is 7, 3, 1 starting at position 0. Each MRZ char
is mapped: `0-9 -> 0-9`, `A-Z -> 10-35`, `< -> 0`. The check digit of a
field is `sum(value[i] * weight[i%3]) % 10`.

The composite check digit is computed over line 2's first 43 chars
(passport_no + ck1 + nationality + birth + ck2 + sex + expiry + ck3 +
personal_no + ck4).

## Known MRZ-design limitations (not bugs)

* Line 1 carries no check digit, so single-char tampered in line 1
  cannot be detected by the check-digit mechanism.
* The composite check digit is modulo 10; therefore a single-char
  flip in a composite-only-covered field has a 1-in-10 chance of
  going undetected. This is a property of the MRZ specification.

# 2_anticounterfeit -- anti-counterfeit verification

Builds on the MRZ library to add deeper validation rules beyond
check-digit correctness:

* Document type prefix (`P<`, `I<`, ...) is in the ICAO 9303 list.
* Issuing state and nationality are 3-letter uppercase codes.
* Sex is `M`, `F`, or `<`.
* Birth and expiry dates parse as YYMMDD and pass month/day bounds.
* Expiry is strictly after birth.
* Birth year is in a plausible window; expiry is at most 30 years
  past the verifier's anchor "today".
* Name has at least 3 alphabetic characters and <80% padding.
* Personal number, if present, contains at least one non-filler char.
* Check digits are within range and consistent with the data.

## Build & test

```sh
./run.sh
```

## CLI

```sh
./build/ac_tool ../1_mrz_decode/data/sample_td3.txt
# verification: OK
#   [OK  ] doc_type         document type prefix recognised
#   [OK  ] issuing_state    issuing state code well-formed
#   ...
```

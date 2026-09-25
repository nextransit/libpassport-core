# Integration test

| stage | rc | ms |
|-------|---|---|
| 1_mrz_encode | 0 | 2.6 |
| 1_mrz_decode | 0 | 2.3 |
| 2_anticounterfeit | 0 | 2.1 |
| 7_security_crosscheck | 0 | 2.2 |
| 4_nfc_reader_happy | 0 | 2.4 |
| 4_nfc_reader_fail | 1 | 2.3 |
| 6_gen_mrz_image | 0 | 3.1 |
| 6_mrz_ocr_traditional | 0 | 4.7 |
| 6_mrz_ocr_cnn | 0 | 4.8 |
| 3_face_gen_a | 0 | 2.7 |
| 3_face_gen_b | 0 | 2.4 |
| 3_face_match | 0 | 2.4 |
| 7_security_bac | 0 | 2.5 |
| 7_security_mac | 0 | 2.0 |

## Outputs

### 1_mrz_encode
```
P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<
L898902C36UTO6908061F9406236ZE184226B<<<<<18
```

### 1_mrz_decode
```
doc_type       : 'P<'
issuing_state  : 'UTO'
surname        : 'ERIKSSON'
given_names    : 'ANNA<MARIA'
passport_no    : 'L898902C3'  ck=6
nationality    : 'UTO'
birth_yymmdd   : '690806'  ck=1
sex            : 'F'
expiry_yymmdd  : '940623'  ck=6
personal_no    : 'ZE184226B<<<<<'  ck=1
composite_ck   : 8
```

### 2_anticounterfeit
```
verification: OK
  [OK  ] doc_type         document type prefix recognised
  [OK  ] issuing_state    issuing state code well-formed
  [OK  ] nationality      nationality code well-formed
  [OK  ] sex              sex field valid
  [OK  ] birth_date       birth date YYMMDD well-formed
  [OK  ] expiry_date      expiry date YYMMDD well-formed
  [OK  ] date_order       expiry > birth
  [OK  ] name_entropy     name has plausible alphabetic content
  [OK  ] personal_no      personal number present
  [OK  ] check_digits     all check digits consistent
  [OK  ] summary          0 fail / 0 warn / 0 info
```

### 7_security_crosscheck
```
cross_check: 5 ok, 0 fail
  passport_no : OK
  birth_date  : OK
  expiry_date : OK
  nationality : OK
  name        : OK
```

### 4_nfc_reader_happy
```
result.ok       : 1
result.step     : DONE
result.detail   : BAC happy path complete; DG1=98 bytes; SOD=16 bytes
result.dg1_mrz  : 5A5F4250F4EIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
L898902C36UTO6908061F9406236ZE184226B<<<<<18
result.sod_head : AABBCCDDEEFF00112233445566778899
mock.unexpected : 0
mock.remaining  : 7
```

### 4_nfc_reader_fail
```
result.ok       : 0
result.step     : ERROR
result.detail   : SELECT AID failed (sw=6A82)
result.dg1_mrz  : 
result.sod_head : 
mock.unexpected : 0
mock.remaining  : 1
```

### 6_mrz_ocr_traditional
```
result.ok       : OK
result.line1    : P<UTOEH1KSSON<ANNA<MAH1A<<<<<<<<<<<<<<<<<<<<
result.line2    : LB9B902C36UTO690B061F9406236ZE1B4226B<<<<<1B
result.conf1    : 84
result.conf2    : 82
band.x band.y band.w band.h : 16 12 1486 102
```

### 6_mrz_ocr_cnn
```
method            : cnn
result.ok         : OK
result.line1      : 77777777777777777777777777777777777777777777
result.line2      : 77777777777777777777777777777777777777777777
result.conf1      : 100
result.conf2      : 99
band.x band.y band.w band.h : 16 12 1486 102
```

### 3_face_match
```
method: aHash
score : 100/100
```

### 7_security_bac
```
K_enc = E2270AEB7C97D607E2270AEB7C97D607
K_mac = 558F5CF96D03FC8DE58B740594169919
```

### 7_security_mac
```
mac = 9342AE663E985835
```


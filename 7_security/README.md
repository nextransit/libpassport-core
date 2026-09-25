# 7_security -- ICAO passport security pipeline

Pure C implementations of the cryptography and image-based
verification building blocks of an e-Passport reader:

* SHA-1 (FIPS 180-4)
* DES + 3DES (FIPS 46-3)
* AES-128 ECB + CBC (FIPS 197)
* ISO/IEC 9797-1 MAC Algorithm 3 (retail MAC on DES)
* ICAO BAC key derivation (SHA-1 -> K_enc/K_mac)
* Cross-modal consistency check (MRZ vs VIZ text)
* B900 IR absorption check (simulated)
* SSIM template match for IR watermark
* UV fluorescence check

## Build & test

```sh
./run.sh
```

# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 1380/1380 (100.0%) | 5.4 | 63.57% | 42.46% | 0.00% | 2.17% |
| cnn | 1380/1380 (100.0%) | 7.4 | 94.16% | 96.32% | 64.42% | 96.09% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 5.3 | 82.72% | 68.56% | 0.00% | 4.17% |
| clean | 100.0% | 7.2 | 96.42% | 98.91% | 88.19% | 98.61% |
| realistic | 100.0% | 5.7 | 42.68% | 13.99% | 0.00% | 0.00% |
| realistic | 100.0% | 7.7 | 91.69% | 93.49% | 38.48% | 93.33% |

## Top confusions (traditional)
| gt->got | count |
|---------|-------|
| R->H | 1238 |
| <->L | 1218 |
| 8->B | 1214 |
| <->A | 767 |
| I->1 | 723 |
| <->H | 458 |
| P->< | 419 |
| X->N | 415 |
| <->O | 347 |
| P->L | 347 |
| <->P | 297 |
| R->< | 279 |
| <->M | 271 |
| 2->< | 260 |
| <->W | 258 |

## Top confusions (cnn)
| gt->got | count |
|---------|-------|
| <->Z | 183 |
| <->S | 123 |
| <->H | 114 |
| E->F | 110 |
| <->L | 99 |
| U->O | 98 |
| <->P | 97 |
| I->T | 87 |
| <->E | 78 |
| <->G | 77 |
| <->C | 77 |
| U->I | 66 |
| F->E | 65 |
| R->T | 57 |
| H->T | 55 |

## Worst columns (traditional)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 7 | 808 | 43 | 975 |
| 8 | 795 | 39 | 905 |
| 12 | 794 | 40 | 893 |
| 11 | 783 | 32 | 891 |
| 10 | 767 | 42 | 886 |

## Worst columns (cnn)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 2 | 239 | 0 | 95 |
| 3 | 205 | 2 | 95 |
| 5 | 197 | 40 | 86 |
| 4 | 189 | 1 | 82 |
| 6 | 186 | 4 | 76 |

# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 1380/1380 (100.0%) | 4.9 | 63.58% | 42.60% | 0.00% | 2.75% |
| cnn | 1380/1380 (100.0%) | 6.5 | 95.75% | 98.87% | 71.81% | 94.64% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 4.7 | 82.72% | 68.70% | 0.00% | 5.28% |
| clean | 100.0% | 6.3 | 98.53% | 99.99% | 95.69% | 100.00% |
| realistic | 100.0% | 5.2 | 42.70% | 14.12% | 0.00% | 0.00% |
| realistic | 100.0% | 6.8 | 92.71% | 97.64% | 45.76% | 88.79% |

## Top confusions (traditional)
| gt->got | count |
|---------|-------|
| R->H | 1239 |
| <->L | 1220 |
| 8->B | 1168 |
| <->A | 767 |
| I->1 | 724 |
| <->H | 456 |
| P->< | 419 |
| X->N | 416 |
| <->O | 347 |
| P->L | 347 |
| <->P | 297 |
| R->< | 277 |
| <->M | 271 |
| <->W | 258 |
| F->< | 253 |

## Top confusions (cnn)
| gt->got | count |
|---------|-------|
| <->C | 489 |
| <->Z | 385 |
| E->F | 122 |
| U->O | 102 |
| I->T | 81 |
| <->S | 64 |
| F->E | 55 |
| O->U | 43 |
| <->H | 43 |
| P->< | 39 |
| P->B | 32 |
| T->I | 31 |
| <->X | 23 |
| <->D | 22 |
| R->< | 21 |

## Worst columns (traditional)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 7 | 808 | 43 | 923 |
| 8 | 795 | 39 | 907 |
| 12 | 793 | 40 | 894 |
| 11 | 783 | 32 | 890 |
| 10 | 766 | 42 | 885 |

## Worst columns (cnn)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 2 | 140 | 0 | 46 |
| 3 | 106 | 2 | 45 |
| 5 | 103 | 40 | 43 |
| 4 | 101 | 41 | 41 |
| 6 | 98 | 32 | 33 |

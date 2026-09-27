# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 1380/1380 (100.0%) | 6.3 | 63.57% | 42.46% | 0.00% | 2.17% |
| cnn | 1380/1380 (100.0%) | 8.0 | 95.75% | 98.79% | 70.94% | 97.32% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 6.2 | 82.72% | 68.56% | 0.00% | 4.17% |
| clean | 100.0% | 7.8 | 98.53% | 99.99% | 95.69% | 100.00% |
| realistic | 100.0% | 6.5 | 42.68% | 13.99% | 0.00% | 0.00% |
| realistic | 100.0% | 8.1 | 92.71% | 97.49% | 43.94% | 94.39% |

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
| <->C | 489 |
| <->Z | 384 |
| E->F | 128 |
| U->O | 104 |
| I->T | 91 |
| <->S | 64 |
| O->U | 58 |
| F->E | 55 |
| <->H | 43 |
| P->< | 39 |
| P->B | 33 |
| T->I | 31 |
| <->X | 23 |
| Z->< | 23 |
| <->D | 22 |

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
| 2 | 140 | 40 | 49 |
| 3 | 106 | 0 | 46 |
| 5 | 103 | 2 | 45 |
| 4 | 101 | 41 | 41 |
| 6 | 98 | 32 | 33 |

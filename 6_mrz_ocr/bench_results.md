# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 1380/1380 (100.0%) | 5.1 | 63.57% | 42.46% | 0.00% | 2.17% |
| cnn | 1380/1380 (100.0%) | 7.1 | 97.71% | 97.77% | 67.03% | 98.26% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 5.0 | 82.72% | 68.56% | 0.00% | 4.17% |
| clean | 100.0% | 6.9 | 99.22% | 99.83% | 92.08% | 99.86% |
| realistic | 100.0% | 5.3 | 42.68% | 13.99% | 0.00% | 0.00% |
| realistic | 100.0% | 7.2 | 96.06% | 95.52% | 39.70% | 96.52% |

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
| O->U | 147 |
| I->T | 111 |
| F->E | 90 |
| B->H | 53 |
| E->F | 51 |
| T->I | 42 |
| F->P | 37 |
| B->P | 36 |
| R->P | 35 |
| N->M | 30 |
| Q->U | 28 |
| U->P | 25 |
| E->P | 24 |
| Z->P | 24 |
| U->O | 22 |

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
| 2 | 180 | 0 | 79 |
| 4 | 122 | 40 | 66 |
| 3 | 114 | 2 | 63 |
| 5 | 109 | 41 | 62 |
| 6 | 106 | 3 | 58 |

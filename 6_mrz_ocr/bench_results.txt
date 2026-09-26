# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 778/1380 (56.4%) | 4.1 | 44.57% | 36.04% | 0.00% | 2.75% |
| cnn | 1380/1380 (100.0%) | 7.4 | 95.63% | 98.87% | 71.45% | 94.64% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 4.2 | 82.72% | 68.70% | 0.00% | 5.28% |
| clean | 100.0% | 7.2 | 98.52% | 99.99% | 95.69% | 100.00% |
| realistic | 8.8% | 4.0 | 2.95% | 0.41% | 0.00% | 0.00% |
| realistic | 100.0% | 7.6 | 92.47% | 97.64% | 45.00% | 88.79% |

## Top confusions (traditional)
| gt->got | count |
|---------|-------|
| <->_ | 14910 |
| P->_ | 1708 |
| 7->_ | 1578 |
| 3->_ | 1535 |
| 6->_ | 1470 |
| 9->_ | 1418 |
| 1->_ | 1412 |
| 2->_ | 1412 |
| 5->_ | 1367 |
| 4->_ | 1359 |
| 8->_ | 1292 |
| 0->_ | 1288 |
| F->_ | 1206 |
| 8->B | 1166 |
| M->_ | 1093 |

## Top confusions (cnn)
| gt->got | count |
|---------|-------|
| <->C | 489 |
| <->Z | 385 |
| E->F | 122 |
| U->O | 102 |
| I->T | 81 |
| P->F | 67 |
| <->S | 64 |
| F->E | 55 |
| O->U | 43 |
| <->H | 43 |
| P->B | 40 |
| P->< | 32 |
| T->I | 31 |
| <->X | 23 |
| P->R | 23 |

## Worst columns (traditional)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 7 | 903 | 43 | 956 |
| 8 | 888 | 39 | 940 |
| 2 | 885 | 32 | 935 |
| 5 | 867 | 34 | 932 |
| 12 | 867 | 28 | 931 |

## Worst columns (cnn)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 2 | 140 | 0 | 46 |
| 0 | 111 | 2 | 45 |
| 3 | 106 | 40 | 43 |
| 5 | 103 | 41 | 41 |
| 4 | 101 | 32 | 33 |

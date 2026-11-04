# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 1380/1380 (100.0%) | 7.5 | 63.57% | 42.46% | 0.00% | 2.17% |
| cnn | 1380/1380 (100.0%) | 9.5 | 87.07% | 72.59% | 24.28% | 60.07% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 7.2 | 82.72% | 68.56% | 0.00% | 4.17% |
| clean | 100.0% | 9.2 | 95.57% | 99.34% | 46.53% | 98.89% |
| realistic | 100.0% | 7.9 | 42.68% | 13.99% | 0.00% | 0.00% |
| realistic | 100.0% | 10.0 | 77.79% | 43.40% | 0.00% | 17.73% |

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
| 4->9 | 749 |
| 6->9 | 661 |
| 5->9 | 616 |
| 8->9 | 584 |
| 0->9 | 560 |
| 2->< | 549 |
| N->M | 540 |
| R->H | 359 |
| K->N | 340 |
| Z->< | 318 |
| 8->0 | 292 |
| W->H | 279 |
| G->H | 272 |
| B->H | 268 |
| 3->7 | 265 |

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
| 2 | 686 | 2 | 501 |
| 4 | 642 | 1 | 499 |
| 3 | 640 | 43 | 493 |
| 5 | 406 | 4 | 473 |
| 6 | 406 | 40 | 472 |

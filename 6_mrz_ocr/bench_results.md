# MRZ OCR benchmark

Corpus: 108 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 108/108 (100.0%) | 2.9 | 2.8 | 3.7 | 88.7% | 93.1% |
| cnn | 108/108 (100.0%) | 3.4 | 3.4 | 4.2 | 83.6% | 80.0% |

## Clean subset (noise=0, n=51)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 88.0% | 93.3% |
| cnn | 86.2% | 83.5% |

## Noisy subset (noise>0, n=57)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 89.3% | 92.9% |
| cnn | 81.3% | 76.9% |

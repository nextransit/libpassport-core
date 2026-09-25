# MRZ OCR benchmark

Corpus: 108 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 108/108 (100.0%) | 3.4 | 3.3 | 4.3 | 88.7% | 93.1% |
| cnn | 108/108 (100.0%) | 5.9 | 5.7 | 7.0 | 85.6% | 81.7% |

## Clean subset (noise=0, n=51)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 88.0% | 93.3% |
| cnn | 88.2% | 84.9% |

## Noisy subset (noise>0, n=57)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 89.3% | 92.9% |
| cnn | 83.2% | 78.8% |

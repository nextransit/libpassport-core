# MRZ OCR benchmark

Corpus: 108 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 108/108 (100.0%) | 5.3 | 5.0 | 7.0 | 88.7% | 93.1% |
| cnn | 108/108 (100.0%) | 5.4 | 5.1 | 7.1 | 80.1% | 78.0% |

## Clean subset (noise=0, n=51)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 88.0% | 93.3% |
| cnn | 80.7% | 77.0% |

## Noisy subset (noise>0, n=57)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 89.3% | 92.9% |
| cnn | 79.4% | 78.8% |

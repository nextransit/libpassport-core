# MRZ OCR benchmark

Corpus: 108 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 108/108 (100.0%) | 3.5 | 3.4 | 4.4 | 88.7% | 93.1% |
| cnn | 108/108 (100.0%) | 5.5 | 5.4 | 6.7 | 83.9% | 80.1% |

## Clean subset (noise=0, n=51)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 88.0% | 93.3% |
| cnn | 86.3% | 83.0% |

## Noisy subset (noise>0, n=57)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 89.3% | 92.9% |
| cnn | 81.8% | 77.4% |

## Breakdown by condition
| condition | method | line1 acc | line2 acc |
|-----------|--------|-----------|-----------|
| clean          (noise=0, skew=0) | traditional | 87.5% | 93.2% |
| clean          (noise=0, skew=0) | cnn | 87.6% | 84.6% |
| noisy_only     (noise>0, skew=0) | traditional | 89.3% | 92.8% |
| noisy_only     (noise>0, skew=0) | cnn | 82.1% | 77.9% |
| skew_only      (noise=0, skew>0) | traditional | 89.4% | 93.5% |
| skew_only      (noise=0, skew>0) | cnn | 80.9% | 75.0% |
| noisy_and_skew | traditional | 87.7% | 93.2% |
| noisy_and_skew | cnn | 91.8% | 88.2% |

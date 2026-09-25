# MRZ OCR benchmark

Corpus: 108 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 108/108 (100.0%) | 3.0 | 2.9 | 3.7 | 88.7% | 93.1% |
| cnn | 108/108 (100.0%) | 4.9 | 4.8 | 5.6 | 85.4% | 77.4% |

## Clean subset (noise=0, n=51)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 88.0% | 93.3% |
| cnn | 87.9% | 80.6% |

## Noisy subset (noise>0, n=57)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 89.3% | 92.9% |
| cnn | 83.2% | 74.6% |

## Breakdown by condition
| condition | method | line1 acc | line2 acc |
|-----------|--------|-----------|-----------|
| clean          (noise=0, skew=0) | traditional | 87.5% | 93.2% |
| clean          (noise=0, skew=0) | cnn | 89.1% | 82.3% |
| noisy_only     (noise>0, skew=0) | traditional | 89.3% | 92.8% |
| noisy_only     (noise>0, skew=0) | cnn | 83.5% | 74.9% |
| skew_only      (noise=0, skew>0) | traditional | 89.4% | 93.5% |
| skew_only      (noise=0, skew>0) | cnn | 82.1% | 73.3% |
| noisy_and_skew | traditional | 87.7% | 93.2% |
| noisy_and_skew | cnn | 95.0% | 88.2% |

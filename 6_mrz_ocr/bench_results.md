# MRZ OCR benchmark

Corpus: 500 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 500/500 (100.0%) | 3.1 | 3.0 | 3.8 | 90.7% | 90.9% |
| cnn | 500/500 (100.0%) | 5.0 | 4.9 | 5.8 | 76.9% | 70.5% |

## Clean subset (noise=0, n=200)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 90.3% | 91.2% |
| cnn | 78.2% | 72.0% |

## Noisy subset (noise>0, n=300)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 90.9% | 90.8% |
| cnn | 76.0% | 69.5% |

## Breakdown by condition
| condition | method | line1 acc | line2 acc |
|-----------|--------|-----------|-----------|
| clean          (noise=0, skew=0) | traditional | 89.0% | 92.1% |
| clean          (noise=0, skew=0) | cnn | 80.7% | 75.3% |
| noisy_only     (noise>0, skew=0) | traditional | 91.2% | 90.5% |
| noisy_only     (noise>0, skew=0) | cnn | 75.9% | 69.5% |
| skew_only      (noise=0, skew>0) | traditional | 91.4% | 90.4% |
| skew_only      (noise=0, skew>0) | cnn | 75.7% | 68.8% |
| noisy_and_skew | traditional | 87.7% | 93.2% |
| noisy_and_skew | cnn | 95.0% | 92.7% |

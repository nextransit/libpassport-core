# MRZ OCR benchmark

Corpus: 500 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 500/500 (100.0%) | 3.9 | 2.9 | 3.7 | 90.7% | 90.9% |
| cnn | 500/500 (100.0%) | 5.0 | 4.9 | 5.8 | 77.0% | 71.0% |

## Clean subset (noise=0, n=200)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 90.3% | 91.2% |
| cnn | 77.5% | 72.5% |

## Noisy subset (noise>0, n=300)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 90.9% | 90.8% |
| cnn | 76.6% | 69.9% |

## Breakdown by condition
| condition | method | line1 acc | line2 acc |
|-----------|--------|-----------|-----------|
| clean          (noise=0, skew=0) | traditional | 89.0% | 92.1% |
| clean          (noise=0, skew=0) | cnn | 78.7% | 76.1% |
| noisy_only     (noise>0, skew=0) | traditional | 91.2% | 90.5% |
| noisy_only     (noise>0, skew=0) | cnn | 76.6% | 69.8% |
| skew_only      (noise=0, skew>0) | traditional | 91.4% | 90.4% |
| skew_only      (noise=0, skew>0) | cnn | 75.8% | 69.1% |
| noisy_and_skew | traditional | 87.7% | 93.2% |
| noisy_and_skew | cnn | 95.9% | 95.0% |

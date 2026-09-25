# MRZ OCR benchmark

Corpus: 500 synthetic MRZ images (see data/corpus.json).

## Overall

| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 500/500 (100.0%) | 4.9 | 3.8 | 4.9 | 91.3% | 90.9% |
| cnn | 500/500 (100.0%) | 5.6 | 5.6 | 6.5 | 77.9% | 71.1% |

## Clean subset (noise=0, n=200)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 91.3% | 91.2% |
| cnn | 79.2% | 72.7% |

## Noisy subset (noise>0, n=300)
| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |
|--------|-------------|---------|--------|--------|-----------|-----------|
| traditional | 91.4% | 90.8% |
| cnn | 77.0% | 70.0% |

## Breakdown by condition
| condition | method | line1 acc | line2 acc |
|-----------|--------|-----------|-----------|
| clean          (noise=0, skew=0) | traditional | 90.8% | 92.1% |
| clean          (noise=0, skew=0) | cnn | 82.6% | 76.2% |
| noisy_only     (noise>0, skew=0) | traditional | 91.6% | 90.5% |
| noisy_only     (noise>0, skew=0) | cnn | 77.0% | 70.0% |
| skew_only      (noise=0, skew>0) | traditional | 91.6% | 90.4% |
| skew_only      (noise=0, skew>0) | cnn | 76.0% | 69.2% |
| noisy_and_skew | traditional | 92.3% | 93.2% |
| noisy_and_skew | cnn | 99.1% | 95.0% |

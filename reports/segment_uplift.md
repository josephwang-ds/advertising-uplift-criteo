# Segment Uplift Analysis

## Method

- Segments: quartiles of `f0` (highest-variance feature in dataset)
- Tests: two-proportion z-test per metric per segment
- Multiple testing: Benjamini-Hochberg FDR correction (α = 0.1)
- Effect size: Cohen's h
- Suppress logic: AND (both visit AND conversion must be simultaneously negative + significant)

## Conversion Uplift by Segment

| Segment | N | Conv lift | Conv lift/100k | Cohen h | BH sig? | Action |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 (low f0) | 25,000 | -0.054% | -54 | -0.012 | — | Retest |
| Q2 | 25,000 | 0.453% | 453 | 0.056 | ✓ | Target |
| Q3 | 25,000 | -0.055% | -55 | -0.020 | — | Retest |
| Q4 (high f0) | 25,000 | 0.006% | 6 | 0.004 | — | Retest |

## Visit Uplift by Segment

| Segment | Visit lift | Visit lift/100k | Cohen h | BH sig? |
| --- | --- | --- | --- | --- |
| Q1 (low f0) | 0.286% | 286 | 0.013 | — |
| Q2 | 3.136% | 3,136 | 0.109 | ✓ |
| Q3 | 0.337% | 337 | 0.025 | — |
| Q4 (high f0) | 0.103% | 103 | 0.009 | — |

## Product Reading

- Segments marked **Target** show statistically significant positive conversion uplift after BH correction.
- Segments marked **Suppress** show negative uplift on BOTH metrics (AND rule — conservative).
- Segments marked **Retest** have inconclusive evidence; recommend holdout replication with more power.
- Cohen's h < 0.2 indicates small effect sizes — business significance depends on volume and cost.

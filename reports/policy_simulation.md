# Policy Simulation

## Assumptions

- Value per incremental conversion: $50
- All figures are per 100,000 users
- Policy A/D share the same reach rate (top 30% of population)

## Policy Comparison

| Policy | Contact cost | Users/100k | Avg uplift | Inc conv/100k | Gross value | Contact cost | **Net ROI** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Uplift top 30% | $0.00 | 15,000 | 0.308% | 46.2 | $2,312 | $0 | **$2,312** |
| C. Response top 30% | $0.00 | 15,000 | 0.241% | 36.2 | $1,808 | $0 | $1,808 |
| B. Uplift top 10% | $0.00 | 5,000 | 0.668% | 33.4 | $1,670 | $0 | $1,670 |
| D. Random 30% | $0.00 | 15,000 | 0.055% | 8.2 | $411 | $0 | $411 |
| B. Uplift top 10% | $0.50 | 5,000 | 0.668% | 33.4 | $1,670 | $2,500 | **$-830** |
| A. Uplift top 30% | $0.50 | 15,000 | 0.308% | 46.2 | $2,312 | $7,500 | $-5,188 |
| C. Response top 30% | $0.50 | 15,000 | 0.241% | 36.2 | $1,808 | $7,500 | $-5,692 |
| D. Random 30% | $0.50 | 15,000 | 0.055% | 8.2 | $411 | $7,500 | $-7,089 |

## Breakeven Analysis

**Breakeven contact cost (Uplift top-30% vs Random 30%):** Uplift targeting always dominates within tested range

At contact costs above the breakeven point, uplift-targeted campaigns generate less net value
than random targeting. At costs below, targeting incremental users dominates.

## Product Reading

- **Uplift targeting (A)** consistently outperforms response targeting (C) in incremental conversions,
  because response-score users are likely to convert even without the ad.
- **Narrow targeting (B)** (top 10%) has the highest precision but smaller reach.
- At zero or low contact cost, all policies yield positive net ROI; at high cost, only targeted campaigns break even.

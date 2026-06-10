# Initial Criteo Uplift Readout

## Dataset Snapshot

- Rows: `100,000`
- Columns: `16`
- Treatment exposure rate among treated users: `3.488%`

## Sample Split

| Group | Users | Share |
| --- | --- | --- |
| Control | 14,954 | 14.954% |
| Treatment | 85,046 | 85.046% |

## Global Treatment Effect

| Metric | Control | Treatment | Absolute Lift | Lift / 100k | p-value |
| --- | --- | --- | --- | --- | --- |
| Visit | 3.879% | 4.896% | 1.018% | 1,018 | 0.0000 |
| Conversion | 0.201% | 0.293% | 0.092% | 92 | 0.0488 |

## Product Reading

- `visit` is a diagnostic metric for ad-driven return visits.
- `conversion` is the primary business outcome.
- The next step is uplift modeling: rank users by incremental response, not raw conversion probability.

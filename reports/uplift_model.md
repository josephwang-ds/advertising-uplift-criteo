# Uplift Model

## Model Setup

- Response baseline: logistic regression predicting conversion.
- Uplift baseline: two-model approach, `p_treated - p_control`.
- Response model AUC on test split: `0.9538`.

## Top Deciles

| Ranking | Decile | Users | Treated Conv | Control Conv | Uplift | Inc Conv / 100k |
| --- | --- | --- | --- | --- | --- | --- |
| uplift_score | 1 | 2,500 | 1.888% | 1.220% | 0.668% | 668 |
| uplift_score | 2 | 2,500 | 0.425% | 0.262% | 0.162% | 162 |
| uplift_score | 3 | 2,500 | 0.094% | 0.000% | 0.094% | 94 |
| response_score | 1 | 2,500 | 2.617% | 1.553% | 1.064% | 1,064 |
| response_score | 2 | 2,500 | 0.187% | 0.273% | -0.086% | -86 |
| response_score | 3 | 2,500 | 0.000% | 0.255% | -0.255% | -255 |

## Product Reading

- A high response score means a user is likely to convert.
- A high uplift score means the ad is more likely to change the user's behavior.
- Paid media budget should prefer high positive uplift, not necessarily high raw conversion probability.

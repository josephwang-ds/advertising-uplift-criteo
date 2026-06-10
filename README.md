# Advertising Incrementality and Budget Allocation at Scale

This project analyzes the Criteo Uplift Prediction dataset as a paid-growth decision problem: which users should receive paid ad exposure when the goal is incremental visits and conversions, not raw conversion probability.

Live demo: https://josephwang-advertising-uplift-criteo.streamlit.app/

Project page: https://www.josephjwang.com/advertising-uplift

## Business Question

For an e-commerce retargeting campaign, should paid media budget be spent on users who are most likely to convert, or on users whose behavior is most likely to change because of ad exposure?

中文问题：

电商再营销投放应该优先买“本来就最可能转化”的用户，还是买“因为广告曝光才会产生增量访问/转化”的用户？

## Dataset

Primary dataset:

Criteo Uplift Prediction Dataset, released with the paper **A Large Scale Benchmark for Uplift Modeling**.

Official dataset page:

`https://ailab.criteo.com/criteo-uplift-prediction-dataset/`

Hugging Face mirror:

`https://huggingface.co/datasets/criteo/criteo-uplift`

The public dataset includes:

- anonymized dense features: `f0` to `f11`.
- `treatment`: whether the user belongs to the treatment group.
- `exposure`: whether the user was effectively exposed.
- `visit`: whether the user visited.
- `conversion`: whether the user converted.

## Product Framing

Hillstrom answers an owned-channel CRM question:

> Should we send promotions, and to whom?

Criteo extends the same incrementality logic to paid acquisition:

> Should we pay to expose this user to an ad, and will the ad create incremental behavior?

## Analysis Plan

### 1. Data Sampling

The full dataset is large, so the first local version should use a reproducible sample.

Initial sample target:

- 100k to 500k rows for quick local iteration.
- full dataset only after the pipeline is stable.

### 2. Global Incrementality Readout

Compare treatment and control:

- visit rate lift.
- conversion rate lift.
- incremental visits per 100k users.
- incremental conversions per 100k users.
- p-values for proportion differences.

### 3. Uplift Modeling Baselines

Start with simple baselines:

- response model: predict conversion probability.
- two-model uplift: train separate treatment and control models, then estimate uplift as `p_treated - p_control`.
- treatment interaction model: include treatment-feature interactions.

### 4. Policy Evaluation

Evaluate whether targeting by uplift beats targeting by raw conversion probability.

Key outputs:

- uplift by score decile.
- incremental conversions captured by top X% targeted users.
- budget allocation curve.
- wasted spend avoided versus broad retargeting.

## Decision Metrics

Primary metric:

- `conversion`

Diagnostic metric:

- `visit`

Policy metric:

- incremental conversions captured per targeted user.
- incremental conversions per 100k eligible users.
- expected value under budget constraints.

## Results

### Global Treatment Effect

| Metric | Control | Treatment | Lift | Inc / 100k | p-value |
|--------|---------|-----------|------|-----------|---------|
| Visit | 3.879% | 4.896% | +1.018 pp | +1,018 | < 0.0001 |
| Conversion | 0.201% | 0.293% | +0.092 pp | +92 | 0.0488 |

### Uplift Model (T-Learner)

- Response model AUC: **0.9538**
- Top uplift decile: **+668 incremental conversions / 100k** users
- Uplift targeting top 30% generates **5.6× more incremental conversions** than random targeting

### Segment Analysis (BH FDR corrected, α = 0.10)

| Segment | Conv lift / 100k | BH sig? | Action |
|---------|-----------------|---------|--------|
| Q1 (low f0) | −54 | — | Retest |
| **Q2** | **+453** | **✓** | **Target** |
| Q3 | −55 | — | Retest |
| Q4 (high f0) | +6 | — | Retest |

### Policy Comparison (per 100k users, $50 conversion value)

| Policy | Inc conv / 100k | Net ROI ($0 contact cost) |
|--------|----------------|--------------------------|
| Uplift top 30% | 46 | $2,312 |
| Response top 30% | 36 | $1,808 |
| Uplift top 10% | 33 | $1,670 |
| Random 30% | 8 | $411 |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a sampled dataset:

```bash
python src/download_sample.py --rows 100000
```

Run the initial readout:

```bash
python src/initial_readout.py
```

Run the baseline uplift workflow:

```bash
python src/uplift_baseline.py
```

## Data Policy

Do not commit full raw Criteo data to GitHub. The full dataset is large, and local iterations should use sampled files under `data/raw/`, which are ignored by git.

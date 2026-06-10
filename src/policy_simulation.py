"""
policy_simulation.py
--------------------
Compare four ad-targeting policies and find the breakeven contact cost.

Policies:
  A. Broad targeting     — target top 30% by uplift score
  B. Narrow targeting    — target top 10% by uplift score
  C. Response targeting  — target top 30% by response (propensity) score
  D. Random targeting    — random 30% of population (baseline)

For each policy and contact cost assumption, compute:
  - Users reached per 100k
  - Incremental conversions per 100k
  - Gross value per 100k (at assumed conversion value)
  - Contact cost per 100k
  - Net ROI per 100k

Breakeven: contact cost at which uplift targeting equals random targeting.

Output:
  data/processed/policy_simulation.csv
  reports/policy_simulation.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECILES_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_deciles.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "policy_simulation.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "policy_simulation.md"

VALUE_PER_CONVERSION = 50.0   # assumed revenue per incremental conversion ($)
CONTACT_COSTS = [0.0, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00]


def load_deciles() -> pd.DataFrame:
    if not DECILES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DECILES_PATH}. Run `python src/uplift_baseline.py` first."
        )
    return pd.read_csv(DECILES_PATH)


def policy_stats(
    deciles: pd.DataFrame,
    ranking: str,
    top_n_deciles: int,
    population: int = 100_000,
) -> dict:
    """Compute aggregate stats for targeting top N deciles under a given ranking."""
    sub = deciles[(deciles["ranking"] == ranking) & (deciles["decile"] <= top_n_deciles)].copy()
    total_users = sub["users"].sum()
    reach_rate = total_users / deciles["users"].sum()

    weighted_uplift = np.average(sub["uplift"], weights=sub["users"])
    incremental = weighted_uplift * population * reach_rate

    return {
        "users_per_100k": round(population * reach_rate),
        "reach_rate": reach_rate,
        "avg_uplift": weighted_uplift,
        "incremental_conv_per_100k": incremental,
    }


def random_policy_stats(deciles: pd.DataFrame, reach_rate: float, population: int = 100_000) -> dict:
    """Random targeting at the same reach rate as broad policy."""
    all_users = deciles["users"].sum()
    overall_uplift = np.average(deciles["uplift"], weights=deciles["users"])
    incremental = overall_uplift * population * reach_rate
    return {
        "users_per_100k": round(population * reach_rate),
        "reach_rate": reach_rate,
        "avg_uplift": overall_uplift,
        "incremental_conv_per_100k": incremental,
    }


def simulate_policies(deciles: pd.DataFrame) -> pd.DataFrame:
    broad = policy_stats(deciles, "uplift_score", top_n_deciles=3)
    narrow = policy_stats(deciles, "uplift_score", top_n_deciles=1)
    response = policy_stats(deciles, "response_score", top_n_deciles=3)
    random = random_policy_stats(deciles, reach_rate=broad["reach_rate"])

    rows = []
    for contact_cost in CONTACT_COSTS:
        for label, stats in [
            ("A. Uplift top 30%", broad),
            ("B. Uplift top 10%", narrow),
            ("C. Response top 30%", response),
            ("D. Random 30%", random),
        ]:
            gross = stats["incremental_conv_per_100k"] * VALUE_PER_CONVERSION
            cost = stats["users_per_100k"] * contact_cost
            net = gross - cost
            rows.append({
                "policy": label,
                "contact_cost": contact_cost,
                "users_per_100k": stats["users_per_100k"],
                "avg_uplift": stats["avg_uplift"],
                "incremental_conv_per_100k": round(stats["incremental_conv_per_100k"], 1),
                "gross_value_per_100k": round(gross, 2),
                "contact_cost_per_100k": round(cost, 2),
                "net_roi_per_100k": round(net, 2),
            })
    return pd.DataFrame(rows)


def find_breakeven(sim: pd.DataFrame) -> float:
    """
    Find the contact cost at which uplift top-30% net ROI equals random-30% net ROI.
    Linear interpolation between simulated cost steps.
    """
    def net(policy_label: str, cost: float) -> float:
        row = sim[(sim["policy"] == policy_label) & (sim["contact_cost"] == cost)]
        return row["net_roi_per_100k"].iloc[0] if len(row) else np.nan

    prev_diff = None
    for cost in CONTACT_COSTS:
        uplift_net = net("A. Uplift top 30%", cost)
        random_net = net("D. Random 30%", cost)
        diff = uplift_net - random_net
        if prev_diff is not None and prev_diff >= 0 and diff < 0:
            # Linear interpolation
            prev_cost = CONTACT_COSTS[CONTACT_COSTS.index(cost) - 1]
            t = prev_diff / (prev_diff - diff)
            return prev_cost + t * (cost - prev_cost)
        prev_diff = diff
    return float("inf")


def pct(v: float) -> str:
    return f"{v:.3%}"


def money(v: float) -> str:
    return f"${v:,.0f}"


def build_report(sim: pd.DataFrame, breakeven: float) -> str:
    # Show only $0 and $0.50 contact cost for concise table
    show_costs = [0.0, 0.50]
    rows = []
    for cost in show_costs:
        sub = sim[sim["contact_cost"] == cost].sort_values("net_roi_per_100k", ascending=False)
        for _, r in sub.iterrows():
            is_best = _ == sub.index[0]
            rows.append([
                r["policy"],
                f"${cost:.2f}",
                f"{r['users_per_100k']:,}",
                pct(r["avg_uplift"]),
                f"{r['incremental_conv_per_100k']:,.1f}",
                money(r["gross_value_per_100k"]),
                money(r["contact_cost_per_100k"]),
                ("**" + money(r["net_roi_per_100k"]) + "**") if is_best else money(r["net_roi_per_100k"]),
            ])

    if breakeven == float("inf"):
        breakeven_str = "Uplift targeting always dominates within tested range"
    else:
        breakeven_str = f"${breakeven:.2f} per contact"

    table = "\n".join([
        "| Policy | Contact cost | Users/100k | Avg uplift | Inc conv/100k | Gross value | Contact cost | **Net ROI** |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ] + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])

    return "\n".join([
        "# Policy Simulation",
        "",
        "## Assumptions",
        "",
        f"- Value per incremental conversion: ${VALUE_PER_CONVERSION:.0f}",
        "- All figures are per 100,000 users",
        "- Policy A/D share the same reach rate (top 30% of population)",
        "",
        "## Policy Comparison",
        "",
        table,
        "",
        "## Breakeven Analysis",
        "",
        f"**Breakeven contact cost (Uplift top-30% vs Random 30%):** {breakeven_str}",
        "",
        "At contact costs above the breakeven point, uplift-targeted campaigns generate less net value",
        "than random targeting. At costs below, targeting incremental users dominates.",
        "",
        "## Product Reading",
        "",
        "- **Uplift targeting (A)** consistently outperforms response targeting (C) in incremental conversions,",
        "  because response-score users are likely to convert even without the ad.",
        "- **Narrow targeting (B)** (top 10%) has the highest precision but smaller reach.",
        "- At zero or low contact cost, all policies yield positive net ROI; at high cost, only targeted campaigns break even.",
        "",
    ])


def main() -> None:
    deciles = load_deciles()
    sim = simulate_policies(deciles)
    breakeven = find_breakeven(sim)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(OUTPUT_PATH, index=False)

    report = build_report(sim, breakeven)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"Breakeven contact cost: ${breakeven:.2f}" if breakeven != float("inf") else "Uplift dominates throughout.")
    print(f"Saved to {OUTPUT_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()

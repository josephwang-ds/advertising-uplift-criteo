"""
segment_uplift.py
-----------------
Segment-level uplift analysis for the Criteo holdout experiment.

Segments are defined by quartiles of f0 (the highest-variance user feature).
For each segment:
  - visit uplift: two-proportion z-test
  - conversion uplift: two-proportion z-test
  - effect size: Cohen's h

Multiple comparisons (8 tests per metric pair) are corrected with
Benjamini-Hochberg FDR at alpha=0.10.

Suppress logic: AND — both visit and conversion must be simultaneously
negative and significant to suppress a segment.
"""
from __future__ import annotations

from math import sqrt, asin
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "criteo_uplift_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "segment_uplift.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "segment_uplift.md"

BH_ALPHA = 0.10


def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run download_sample.py first.")
    df = pd.read_csv(RAW_PATH)
    # Label f0 into segments using rank-based quartiles (handles ties / duplicate edges)
    df["segment"] = pd.qcut(
        df["f0"].rank(method="first"),
        q=4,
        labels=["Q1 (low f0)", "Q2", "Q3", "Q4 (high f0)"],
    )
    return df


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))


def two_prop_z(x_t: int, n_t: int, x_c: int, n_c: int) -> tuple[float, float]:
    """Two-proportion z-test; returns (z, p_value)."""
    p_t = x_t / n_t
    p_c = x_c / n_c
    pooled = (x_t + x_c) / (n_t + n_c)
    se = sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    z = (p_t - p_c) / se if se else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def analyse_segment(seg: pd.DataFrame) -> dict:
    ctrl = seg[seg["treatment"] == 0]
    trt = seg[seg["treatment"] == 1]
    n_c, n_t = len(ctrl), len(trt)

    result: dict = {"n_control": n_c, "n_treatment": n_t}
    for metric in ("visit", "conversion"):
        x_c = ctrl[metric].sum()
        x_t = trt[metric].sum()
        p_c = x_c / n_c if n_c else 0
        p_t = x_t / n_t if n_t else 0
        z, p = two_prop_z(x_t, n_t, x_c, n_c)
        h = cohen_h(max(p_t, 1e-9), max(p_c, 1e-9))
        result[f"{metric}_rate_control"] = p_c
        result[f"{metric}_rate_treatment"] = p_t
        result[f"{metric}_lift"] = p_t - p_c
        result[f"{metric}_lift_per_100k"] = (p_t - p_c) * 100_000
        result[f"{metric}_cohen_h"] = h
        result[f"{metric}_p_raw"] = p
    return result


def apply_bh(df: pd.DataFrame, col: str) -> pd.Series:
    pvals = df[col].values
    rejected = false_discovery_control(pvals, method="bh") <= BH_ALPHA
    # false_discovery_control returns adjusted p-values
    return pd.Series(false_discovery_control(pvals, method="bh"), index=df.index, name=col.replace("_p_raw", "_p_bh"))


def classify(row: pd.Series) -> str:
    """Target / Suppress / Retest using AND logic for suppress."""
    v_pos = row["visit_lift"] > 0 and row["visit_sig_bh"]
    v_neg = row["visit_lift"] < 0 and row["visit_sig_bh"]
    c_pos = row["conversion_lift"] > 0 and row["conversion_sig_bh"]
    c_neg = row["conversion_lift"] < 0 and row["conversion_sig_bh"]

    if c_pos:
        return "Target"
    if v_neg and c_neg:          # AND logic
        return "Suppress"
    return "Retest"


def pct(v: float) -> str:
    return f"{v:.3%}"


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(df: pd.DataFrame) -> str:
    headers = [
        "Segment", "N", "Conv lift", "Conv lift/100k",
        "Cohen h", "BH sig?", "Action",
    ]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["segment"],
            f"{r['n_treatment'] + r['n_control']:,}",
            pct(r["conversion_lift"]),
            f"{r['conversion_lift_per_100k']:,.0f}",
            fmt(r["conversion_cohen_h"]),
            "✓" if r["conversion_sig_bh"] else "—",
            r["action"],
        ])

    visit_rows = []
    for _, r in df.iterrows():
        visit_rows.append([
            r["segment"],
            pct(r["visit_lift"]),
            f"{r['visit_lift_per_100k']:,.0f}",
            fmt(r["visit_cohen_h"]),
            "✓" if r["visit_sig_bh"] else "—",
        ])

    return "\n".join([
        "# Segment Uplift Analysis",
        "",
        "## Method",
        "",
        "- Segments: quartiles of `f0` (highest-variance feature in dataset)",
        "- Tests: two-proportion z-test per metric per segment",
        f"- Multiple testing: Benjamini-Hochberg FDR correction (α = {BH_ALPHA})",
        "- Effect size: Cohen's h",
        "- Suppress logic: AND (both visit AND conversion must be simultaneously negative + significant)",
        "",
        "## Conversion Uplift by Segment",
        "",
        markdown_table(headers, rows),
        "",
        "## Visit Uplift by Segment",
        "",
        markdown_table(
            ["Segment", "Visit lift", "Visit lift/100k", "Cohen h", "BH sig?"],
            visit_rows,
        ),
        "",
        "## Product Reading",
        "",
        "- Segments marked **Target** show statistically significant positive conversion uplift after BH correction.",
        "- Segments marked **Suppress** show negative uplift on BOTH metrics (AND rule — conservative).",
        "- Segments marked **Retest** have inconclusive evidence; recommend holdout replication with more power.",
        "- Cohen's h < 0.2 indicates small effect sizes — business significance depends on volume and cost.",
        "",
    ])


def main() -> None:
    df = load_data()
    segments = []
    for seg_name, grp in df.groupby("segment", observed=True):
        row = {"segment": seg_name}
        row.update(analyse_segment(grp))
        segments.append(row)

    result = pd.DataFrame(segments)

    # BH correction per metric
    result["visit_p_bh"] = false_discovery_control(result["visit_p_raw"].values, method="bh")
    result["conversion_p_bh"] = false_discovery_control(result["conversion_p_raw"].values, method="bh")
    result["visit_sig_bh"] = result["visit_p_bh"] <= BH_ALPHA
    result["conversion_sig_bh"] = result["conversion_p_bh"] <= BH_ALPHA
    result["action"] = result.apply(classify, axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    report = build_report(result)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"Saved segment table to {OUTPUT_PATH}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

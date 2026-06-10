from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "criteo_uplift_sample.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "initial_readout.md"


def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing sample data at {RAW_PATH}. Run `python src/download_sample.py --rows 100000` first."
        )
    return pd.read_csv(RAW_PATH)


def two_prop_test(control: pd.Series, treatment: pd.Series) -> dict[str, float]:
    n_c = control.shape[0]
    n_t = treatment.shape[0]
    x_c = control.sum()
    x_t = treatment.sum()
    p_c = x_c / n_c
    p_t = x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se = sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se if se else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "control_rate": p_c,
        "treatment_rate": p_t,
        "abs_lift": p_t - p_c,
        "p_value": p_value,
    }


def pct(value: float) -> str:
    return f"{value:.3%}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(df: pd.DataFrame) -> str:
    control = df[df["treatment"] == 0]
    treatment = df[df["treatment"] == 1]

    sample_rows = [
        ["Control", f"{len(control):,}", pct(len(control) / len(df))],
        ["Treatment", f"{len(treatment):,}", pct(len(treatment) / len(df))],
    ]

    visit = two_prop_test(control["visit"], treatment["visit"])
    conversion = two_prop_test(control["conversion"], treatment["conversion"])
    exposure_rate = treatment["exposure"].mean()

    metric_rows = [
        [
            "Visit",
            pct(visit["control_rate"]),
            pct(visit["treatment_rate"]),
            pct(visit["abs_lift"]),
            f"{visit['abs_lift'] * 100_000:,.0f}",
            f"{visit['p_value']:.4f}",
        ],
        [
            "Conversion",
            pct(conversion["control_rate"]),
            pct(conversion["treatment_rate"]),
            pct(conversion["abs_lift"]),
            f"{conversion['abs_lift'] * 100_000:,.0f}",
            f"{conversion['p_value']:.4f}",
        ],
    ]

    return "\n".join(
        [
            "# Initial Criteo Uplift Readout",
            "",
            "## Dataset Snapshot",
            "",
            f"- Rows: `{len(df):,}`",
            f"- Columns: `{df.shape[1]}`",
            f"- Treatment exposure rate among treated users: `{pct(exposure_rate)}`",
            "",
            "## Sample Split",
            "",
            markdown_table(["Group", "Users", "Share"], sample_rows),
            "",
            "## Global Treatment Effect",
            "",
            markdown_table(
                ["Metric", "Control", "Treatment", "Absolute Lift", "Lift / 100k", "p-value"],
                metric_rows,
            ),
            "",
            "## Product Reading",
            "",
            "- `visit` is a diagnostic metric for ad-driven return visits.",
            "- `conversion` is the primary business outcome.",
            "- The next step is uplift modeling: rank users by incremental response, not raw conversion probability.",
            "",
        ]
    )


def main() -> None:
    df = load_data()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(df)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

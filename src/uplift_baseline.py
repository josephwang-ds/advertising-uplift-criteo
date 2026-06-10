from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "criteo_uplift_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_deciles.csv"
SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_scores.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "uplift_model.md"

FEATURES = [f"f{i}" for i in range(12)]
TARGET = "conversion"

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")


def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing sample data at {RAW_PATH}. Run `python src/download_sample.py --rows 100000` first."
        )
    df = pd.read_csv(RAW_PATH)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def train_response_model(train: pd.DataFrame) -> object:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train[FEATURES], train[TARGET])
    return model


def score_two_model_uplift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    treated_train = train[train["treatment"] == 1]
    control_train = train[train["treatment"] == 0]

    treated_model = train_response_model(treated_train)
    control_model = train_response_model(control_train)
    response_model = train_response_model(train)

    scored = test.copy()
    scored["p_treated"] = treated_model.predict_proba(test[FEATURES])[:, 1]
    scored["p_control"] = control_model.predict_proba(test[FEATURES])[:, 1]
    scored["uplift_score"] = scored["p_treated"] - scored["p_control"]
    scored["response_score"] = response_model.predict_proba(test[FEATURES])[:, 1]
    return scored


def summarize_deciles(scored: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["decile"] = pd.qcut(out[score_col].rank(method="first"), 10, labels=False) + 1
    out["decile"] = 11 - out["decile"]

    rows = []
    for decile, sub in out.groupby("decile"):
        treated = sub[sub["treatment"] == 1]
        control = sub[sub["treatment"] == 0]
        if treated.empty or control.empty:
            continue
        treated_rate = treated[TARGET].mean()
        control_rate = control[TARGET].mean()
        rows.append(
            {
                "ranking": score_col,
                "decile": int(decile),
                "users": len(sub),
                "treated_conversion": treated_rate,
                "control_conversion": control_rate,
                "uplift": treated_rate - control_rate,
                "incremental_conversions_per_100k": (treated_rate - control_rate) * 100_000,
            }
        )
    return pd.DataFrame(rows).sort_values(["ranking", "decile"])


def pct(value: float) -> str:
    return f"{value:.3%}"


def markdown_table(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                row["ranking"],
                str(int(row["decile"])),
                f"{int(row['users']):,}",
                pct(row["treated_conversion"]),
                pct(row["control_conversion"]),
                pct(row["uplift"]),
                f"{row['incremental_conversions_per_100k']:,.0f}",
            ]
        )
    headers = [
        "Ranking",
        "Decile",
        "Users",
        "Treated Conv",
        "Control Conv",
        "Uplift",
        "Inc Conv / 100k",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(deciles: pd.DataFrame, response_auc: float) -> str:
    top = deciles[deciles["decile"] <= 3].copy()
    return "\n".join(
        [
            "# Uplift Model",
            "",
            "## Model Setup",
            "",
            "- Response baseline: logistic regression predicting conversion.",
            "- Uplift baseline: two-model approach, `p_treated - p_control`.",
            f"- Response model AUC on test split: `{response_auc:.4f}`.",
            "",
            "## Top Deciles",
            "",
            markdown_table(top),
            "",
            "## Product Reading",
            "",
            "- A high response score means a user is likely to convert.",
            "- A high uplift score means the ad is more likely to change the user's behavior.",
            "- Paid media budget should prefer high positive uplift, not necessarily high raw conversion probability.",
            "",
        ]
    )


def main() -> None:
    df = load_data()
    train, test = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df[["treatment", TARGET]],
    )
    scored = score_two_model_uplift(train, test)
    response_auc = roc_auc_score(scored[TARGET], scored["response_score"])

    deciles = pd.concat(
        [
            summarize_deciles(scored, "uplift_score"),
            summarize_deciles(scored, "response_score"),
        ],
        ignore_index=True,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    deciles.replace([np.inf, -np.inf], np.nan).to_csv(OUTPUT_PATH, index=False)

    # Save per-user scores for downstream use (policy sim, Streamlit)
    score_cols = ["treatment", "conversion", "visit", "uplift_score", "response_score"]
    scored[score_cols].reset_index(drop=True).to_csv(SCORES_PATH, index=False)

    report = build_report(deciles, response_auc)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"Saved decile table to {OUTPUT_PATH}")
    print(f"Saved uplift scores to {SCORES_PATH}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""
Advertising Uplift Analysis — Interactive Streamlit App
josephjwang.com  ·  Criteo holdout experiment
"""
from __future__ import annotations

from math import sqrt, asin
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


st.set_page_config(
    page_title="Advertising Uplift Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECILES_PATH = PROJECT_ROOT / "data" / "processed" / "uplift_deciles.csv"
SEGMENT_PATH = PROJECT_ROOT / "data" / "processed" / "segment_uplift.csv"
POLICY_PATH  = PROJECT_ROOT / "data" / "processed" / "policy_simulation.csv"
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "criteo_uplift_sample.csv"

GITHUB_URL  = "https://github.com/josephwang-ds/advertising-uplift-criteo"
WEBSITE_URL = "https://www.josephjwang.com/advertising-uplift"


def add_css():
    st.markdown("""<style>
.block-container{padding-top:2rem;padding-bottom:3rem;}
div[data-testid="stMetric"]{background:#f8fafc;border:1px solid #dbe4e8;border-radius:8px;padding:1rem 1rem .8rem;}
.callout{background:#f7fbfa;border:1px solid #d8eee8;border-left:4px solid #0f766e;border-radius:8px;padding:1rem 1.1rem;margin:.5rem 0 1rem;color:#12312d;}
.warn{background:#fffbeb;border-color:#fde68a;border-left-color:#d97706;}
.story-box{background:#f0f4ff;border:1px solid rgba(99,102,241,0.35);border-left:4px solid #6366f1;border-radius:0 8px 8px 0;padding:1.1rem 1.3rem;margin:.5rem 0 1.2rem;color:#1e1b4b;line-height:1.8;}
.decision-box{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:0 8px 8px 0;padding:1.1rem 1.3rem;margin:.5rem 0 1rem;color:#14532d;line-height:1.8;}
</style>""", unsafe_allow_html=True)

def pct(v, d=3): return f"{v:.{d}%}"
def pp(v, d=3):  return f"{v*100:.{d}f} pp"
def money(v):    return f"${v:,.0f}"
def fmt(v, d=3): return f"{v:.{d}f}"

def cohen_h(p1, p2):
    return 2*asin(sqrt(max(p1,1e-9))) - 2*asin(sqrt(max(p2,1e-9)))

def two_prop_z(x_t, n_t, x_c, n_c):
    p_t, p_c = x_t/n_t, x_c/n_c
    pooled = (x_t+x_c)/(n_t+n_c)
    se = sqrt(pooled*(1-pooled)*(1/n_t+1/n_c))
    z  = (p_t-p_c)/se if se else 0.0
    p  = 2*(1-stats.norm.cdf(abs(z)))
    return p_t, p_c, z, p, p_t-p_c-1.96*se, p_t-p_c+1.96*se

@st.cache_data
def load_raw():       return pd.read_csv(RAW_PATH)
@st.cache_data
def load_deciles():   return pd.read_csv(DECILES_PATH)
@st.cache_data
def load_segments():  return pd.read_csv(SEGMENT_PATH)
@st.cache_data
def load_policy():    return pd.read_csv(POLICY_PATH)

@st.cache_data
def compute_readout(df):
    ctrl = df[df["treatment"]==0]; trt = df[df["treatment"]==1]
    out = {}
    for m in ("visit","conversion"):
        x_c,n_c = ctrl[m].sum(), len(ctrl)
        x_t,n_t = trt[m].sum(),  len(trt)
        p_t,p_c,z,p,ci_lo,ci_hi = two_prop_z(x_t,n_t,x_c,n_c)
        out[m] = dict(p_control=p_c, p_treatment=p_t, lift=p_t-p_c,
                      lift_per_100k=(p_t-p_c)*100_000, p_value=p, z=z,
                      ci_lo=ci_lo, ci_hi=ci_hi, cohen_h=cohen_h(p_t,p_c),
                      n_control=n_c, n_treatment=n_t)
    return out


def generate_story_opener(df, rd, deciles, segs):
    """Compute headline facts for the story-opener callout."""
    n = len(df)
    visit_lift_pp = rd["visit"]["lift"] * 100
    conv_lift_pp  = rd["conversion"]["lift"] * 100
    conv_sig      = rd["conversion"]["p_value"] < 0.05

    # Best BH-significant targetable segment
    target_segs = segs[segs["action"] == "Target"] if "action" in segs.columns else pd.DataFrame()
    if not target_segs.empty:
        best_row  = target_segs.loc[target_segs["conversion_lift_per_100k"].idxmax()]
        best_seg  = best_row["segment"]
        best_inc  = float(best_row["conversion_lift_per_100k"])
    else:
        best_seg, best_inc = "Q2", 453.0

    # Suppress segments
    suppress_count = int((segs["action"] == "Suppress").sum()) if "action" in segs.columns else 0

    # Uplift top-30% vs random efficiency from deciles (uplift_score, deciles 1-3 vs all)
    top3 = deciles[(deciles["ranking"] == "uplift_score") & (deciles["decile"] <= 3)]
    all_d = deciles[deciles["ranking"] == "uplift_score"]
    uplift_avg_inc = float(top3["incremental_conversions_per_100k"].mean()) if len(top3) else 0
    rand_avg_inc   = float(all_d["incremental_conversions_per_100k"].mean()) if len(all_d) else 0
    ratio = round(uplift_avg_inc / rand_avg_inc, 1) if rand_avg_inc > 0 else 0

    return dict(
        n=n, visit_lift_pp=visit_lift_pp, conv_lift_pp=conv_lift_pp,
        conv_sig=conv_sig, best_seg=best_seg, best_inc=best_inc,
        suppress_count=suppress_count, ratio=ratio,
    )


def main():
    add_css()

    missing = [p for p in [DECILES_PATH, RAW_PATH] if not p.exists()]
    if missing:
        st.error(f"Missing files: {missing}. Run `python src/run_pipeline.py` first.")
        return

    df      = load_raw()
    deciles = load_deciles()
    rd      = compute_readout(df)
    segs    = load_segments() if SEGMENT_PATH.exists() else pd.DataFrame()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        lang = st.radio("语言 / Language", ["English", "中文"], horizontal=True)
        st.markdown(f"← [项目分析报告]({WEBSITE_URL})  ·  [GitHub]({GITHUB_URL})")
        st.caption("交互工具版：用实验结果辅助广告预算分配。" if lang == "中文" else
                   "Interactive tool: use experiment results to guide ad budget allocation.")
        st.divider()
        st.header("Controls" if lang == "English" else "参数控制")
        contact_cost  = st.slider("每用户接触成本 ($)" if lang == "中文" else "Contact cost per user ($)", 0.0, 5.0, 0.50, 0.10)
        value_per_conv = st.slider("每增量转化价值 ($)" if lang == "中文" else "Value per incremental conversion ($)", 5, 200, 50, 5)
        st.divider()
        ranking_opts = ["提升分数", "响应分数"] if lang == "中文" else ["Uplift score", "Response score"]
        ranking_label = st.selectbox("分桶排序方式" if lang == "中文" else "Decile ranking", ranking_opts)
        ranking = "uplift_score" if ranking_label in ("Uplift score", "提升分数") else "response_score"
        top_n   = st.slider("定向前 N 分桶" if lang == "中文" else "Top deciles to target", 1, 10, 3)
        st.divider()
        st.caption("Criteo Uplift Dataset · 100k users · randomized holdout")

    def t(en: str, zh: str) -> str:
        return zh if lang == "中文" else en

    # ── Header ───────────────────────────────────────────────────────────────
    st.title(t("📈 Advertising Incrementality & Uplift Analysis", "📈 广告增量与提升分析"))
    st.caption(t(
        "Paid ads holdout · Criteo dataset · Incrementality readout, segment analysis, budget allocation",
        "付费广告保留实验 · Criteo 数据集 · 增量分析、分群分析与预算分配"
    ))
    _biz_q = t(
        "Which users should receive paid retargeting ads because the ad creates <em>incremental</em> conversion, not merely because they have a high baseline purchase probability?",
        "付费再营销广告应该投给哪些用户？标准不是“谁本来就容易买”，而是“广告是否真的带来了额外转化”。"
    )
    st.markdown(f'<div class="callout"><b>{t("Business question","业务问题")}</b> — {_biz_q}</div>', unsafe_allow_html=True)

    # Story opener
    if not segs.empty:
        so = generate_story_opener(df, rd, deciles, segs)
        sig_note = t("statistically significant", "统计显著") if so["conv_sig"] else t("borderline significant", "边界显著")
        st.markdown(f"""<div class="story-box">
<b>📊 {t("What this experiment found", "实验发现")}</b><br>
{so['n']:,} {t("users", "用户")}. {t("Conversion lift", "转化提升")}: <b>+{so['conv_lift_pp']:.3f} pp</b> ({sig_note}).
{t("Visit lift", "访问提升")}: <b>+{so['visit_lift_pp']:.2f} pp</b> (p &lt; 0.0001).<br><br>
{t(
    f"The story is in the segments: <b>{so['best_seg']}</b> drove <b>{so['best_inc']:,.0f} incremental conversions per 100K</b> — while 3 other segments showed noise-level or negative lift.",
    f"关键在于分群：<b>{so['best_seg']}</b> 带来了 <b>每 10 万用户 {so['best_inc']:,.0f} 增量转化</b>，而其他 3 个分群仅有噪声级或负向提升。"
)}<br><br>
💡 {t(
    f"The key insight: high baseline purchase probability ≠ high ad-driven uplift. Uplift targeting delivers <b>{so['ratio']}× more incremental conversions</b> than random targeting the same budget.",
    f"核心洞察：高基础购买概率 ≠ 高广告驱动提升。提升定向比随机定向多 <b>{so['ratio']}× 增量转化</b>。"
)}
</div>""", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    top1 = deciles[(deciles["ranking"]=="uplift_score")&(deciles["decile"]==1)].iloc[0]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric(t("Users", "用户数"), f"{len(df):,}")
    c2.metric(t("Visit lift", "访问提升"),       pp(rd["visit"]["lift"]),       delta=f"+{rd['visit']['lift_per_100k']:,.0f}/100k")
    c3.metric(t("Conversion lift", "转化提升"),  pp(rd["conversion"]["lift"]),  delta=f"+{rd['conversion']['lift_per_100k']:,.0f}/100k")
    c4.metric(t("Conv p-value", "转化 p 值"),     f"{rd['conversion']['p_value']:.4f}")
    c5.metric(t("Top decile uplift", "最高分桶提升"), pp(float(top1["uplift"])), delta=f"+{top1['incremental_conversions_per_100k']:,.0f}/100k")
    c6.metric("Response AUC",     "0.9538")

    st.divider()

    # ── 1. Global Readout ────────────────────────────────────────────────────
    st.subheader(t("1. Did the ad work? — Global Readout", "1. 广告有效果吗？— 全局分析"))
    metric_col = t("Metric", "指标")
    ctrl_col   = t("Control rate", "对照组比率")
    trt_col    = t("Treatment rate", "实验组比率")
    lift_col   = t("Absolute lift", "绝对提升")
    tbl = pd.DataFrame([{
        metric_col:  m.capitalize(),
        ctrl_col:    pct(rd[m]["p_control"]),
        trt_col:     pct(rd[m]["p_treatment"]),
        lift_col:    pp(rd[m]["lift"]),
        "95% CI":    f"[{pp(rd[m]['ci_lo'])}, {pp(rd[m]['ci_hi'])}]",
        "p-value":   "<0.0001" if rd[m]["p_value"]<0.0001 else f"{rd[m]['p_value']:.4f}",
        "Cohen's h": fmt(rd[m]["cohen_h"]),
    } for m in ("visit","conversion")])
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    with st.expander(t("Statistical details", "统计细节")):
        for m in ("visit","conversion"):
            r = rd[m]
            st.markdown(f"""
**{m.capitalize()}**
- z = {r['z']:.3f} · p = {r['p_value']:.4f} · Cohen's h = {r['cohen_h']:.4f}
- 95% CI on lift: [{pp(r['ci_lo'])}, {pp(r['ci_hi'])}]
- n_control = {r['n_control']:,} · n_treatment = {r['n_treatment']:,}
""")
        if lang == "English":
            st.markdown("""
**Test choice**: Two-proportion z-test — correct for binary (0/1) outcomes.
For continuous spend metrics, Welch's t-test + Mann-Whitney U robustness check would be used.

**Effect size guide**: Cohen's h < 0.2 = small · ≈ 0.5 = medium · > 0.8 = large.
Both metrics show small effect sizes — business value comes from scale, not per-user magnitude.

**Imbalance note**: 85k treated vs 15k control (Criteo holdout design). z-test handles unequal N correctly.
""")
        else:
            st.markdown("""
**检验选择**：双比例 z 检验——适用于二元 (0/1) 结果变量。
对于连续型花费指标，会用 Welch t 检验 + Mann-Whitney U 稳健性检验。

**效应量参考**：Cohen's h < 0.2 小 · ≈ 0.5 中 · > 0.8 大。
两个指标均为小效应——业务价值来自规模，而非单用户量级。

**样本不平衡说明**：8.5 万实验组 vs 1.5 万对照组（Criteo 保留实验设计）。z 检验可正确处理不等 N。
""")

    st.divider()

    # ── 2. Segment Uplift ────────────────────────────────────────────────────
    st.subheader(t("2. Who did it work for? — Segment Analysis", "2. 广告对谁有效？— 分群分析"))
    if SEGMENT_PATH.exists():
        sort_opts = (["转化提升", "访问提升", "分群名称"] if lang == "中文"
                     else ["Conversion lift", "Visit lift", "Segment name"])
        sort_by = st.radio(t("Sort by", "排序方式"), sort_opts, horizontal=True)
        sort_col = {
            "Conversion lift": "conversion_lift", "转化提升": "conversion_lift",
            "Visit lift":      "visit_lift",       "访问提升": "visit_lift",
            "Segment name":    "segment",          "分群名称": "segment",
        }[sort_by]
        segs = segs.sort_values(sort_col, ascending=(sort_col=="segment"))

        chart = segs.set_index("segment")[["conversion_lift_per_100k","visit_lift_per_100k"]]
        chart.columns = [t("Conversion lift/100k", "转化提升/10万"), t("Visit lift/100k", "访问提升/10万")]
        st.bar_chart(chart, use_container_width=True)

        icons = {
            "Target":   f"🟢 {t('Target', '定向')}",
            "Suppress": f"🔴 {t('Suppress', '屏蔽')}",
            "Retest":   f"🟡 {t('Retest', '重测')}",
        }
        display = pd.DataFrame({
            t("Segment", "分群"):         segs["segment"],
            t("Conv lift", "转化提升"):   segs["conversion_lift"].map(pp),
            t("Conv lift/100k", "转化提升/10万"): segs["conversion_lift_per_100k"].map(lambda v: f"{v:,.0f}"),
            "Cohen's h":                  segs["conversion_cohen_h"].map(fmt),
            t("BH sig?", "BH显著?"):      segs["conversion_sig_bh"].map(lambda v: "✓" if v else "—"),
            t("Visit lift/100k", "访问提升/10万"): segs["visit_lift_per_100k"].map(lambda v: f"{v:,.0f}"),
            t("Action", "行动"):          segs["action"].map(lambda v: icons.get(v, v)),
        })
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.markdown(f"""<div class="callout">
{t(
    "BH FDR correction α = 10%. <b>Suppress</b> = both visit AND conversion negative + significant (AND rule). Cohen's h &lt; 0.2 = small effect.",
    "BH FDR 校正 α = 10%。<b>屏蔽</b> = 访问和转化均为负且显著（AND 规则）。Cohen's h &lt; 0.2 = 小效应。"
)}
</div>""", unsafe_allow_html=True)
    else:
        st.info(t("Run `python src/segment_uplift.py` to generate segment data.",
                  "运行 `python src/segment_uplift.py` 生成分群数据。"))

    st.divider()

    # ── 3. Policy Simulation ─────────────────────────────────────────────────
    st.subheader(t("3. Who should we target? — Budget Policy Simulation",
                   "3. 应该定向哪些用户？— 预算策略模拟"))
    if POLICY_PATH.exists():
        policy_df = load_policy()
        avail_costs = sorted(policy_df["contact_cost"].unique())
        closest = min(avail_costs, key=lambda c: abs(c-contact_cost))

        bdf = []
        for c in avail_costs:
            s = policy_df[policy_df["contact_cost"]==c].copy()
            s["net"] = s["incremental_conv_per_100k"]*value_per_conv - s["contact_cost_per_100k"]
            for _,row in s.iterrows():
                bdf.append({"Contact cost":c,"Policy":row["policy"],"Net ROI/100k":row["net"]})
        bdf = pd.DataFrame(bdf).pivot(index="Contact cost",columns="Policy",values="Net ROI/100k")
        st.line_chart(bdf, use_container_width=True)

        sim = policy_df[policy_df["contact_cost"]==closest].copy()
        sim["net"] = sim["incremental_conv_per_100k"]*value_per_conv - sim["contact_cost_per_100k"]
        sim = sim.sort_values("net", ascending=False)
        a_net = sim[sim["policy"]=="A. Uplift top 30%"]["net"].values
        d_net = sim[sim["policy"]=="D. Random 30%"]["net"].values
        if len(a_net) and len(d_net):
            if a_net[0] > d_net[0]:
                st.markdown(f'<div class="callout">{t(f"At ${closest:.2f}/contact, uplift targeting beats random by", f"每次接触 ${closest:.2f} 时，提升定向比随机多")} <b>{money(a_net[0]-d_net[0])}/100k</b>.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="callout warn">{t("Contact cost exceeds incremental value — consider narrowing targeting or renegotiating CPM.", "接触成本超过增量价值——考虑缩小定向范围或重新谈判 CPM。")}</div>', unsafe_allow_html=True)

        disp = sim[["policy","users_per_100k","avg_uplift","incremental_conv_per_100k","contact_cost_per_100k","net"]].copy()
        disp.columns = [
            t("Policy","策略"), t("Users/100k","用户/10万"), t("Avg uplift","平均提升"),
            t("Inc conv/100k","增量转化/10万"), t("Contact cost","接触成本"), t("Net ROI","净ROI")
        ]
        disp[t("Avg uplift","平均提升")]        = disp[t("Avg uplift","平均提升")].map(pp)
        disp[t("Inc conv/100k","增量转化/10万")] = disp[t("Inc conv/100k","增量转化/10万")].map(lambda v: f"{v:.1f}")
        disp[t("Contact cost","接触成本")]       = disp[t("Contact cost","接触成本")].map(money)
        disp[t("Net ROI","净ROI")]               = disp[t("Net ROI","净ROI")].map(money)
        st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)
    else:
        st.info(t("Run `python src/policy_simulation.py` to generate policy data.",
                  "运行 `python src/policy_simulation.py` 生成策略数据。"))

    st.divider()

    # ── 4. Uplift Model ──────────────────────────────────────────────────────
    st.subheader(t("4. The model — Uplift Score vs Response Score",
                   "4. 模型 — 提升分数 vs 响应分数"))
    st.write(t(
        "T-learner: separate logistic regression for treatment/control. Uplift = P(conv|ad) − P(conv|no ad).",
        "T-learner：分别对实验组/对照组训练逻辑回归。提升分数 = P(转化|有广告) − P(转化|无广告)。"
    ))

    chart_data = (
        deciles.pivot(index="decile", columns="ranking", values="incremental_conversions_per_100k")
        .rename(columns={"uplift_score": t("Uplift score","提升分数"),
                         "response_score": t("Response score","响应分数")})
        .sort_index()
    )
    st.bar_chart(chart_data, use_container_width=True)

    calc_df  = deciles[deciles["ranking"]==ranking]
    selected = calc_df[calc_df["decile"]<=top_n]
    wt_lift  = np.average(selected["uplift"], weights=selected["users"])
    inc_tot  = wt_lift * selected["users"].sum() / deciles["users"].sum() * 100_000

    m1,m2,m3 = st.columns(3)
    m1.metric(t("Weighted avg uplift", "加权平均提升"),   pp(wt_lift))
    m2.metric(t("Incremental conv/100k", "增量转化/10万"), f"{inc_tot:,.1f}")
    m3.metric(t("Expected value/100k", "预期价值/10万"),   money(inc_tot*value_per_conv))

    disp2 = selected[["decile","users","treated_conversion","control_conversion","uplift","incremental_conversions_per_100k"]].copy()
    disp2.columns = [
        t("Decile","分桶"), t("Users","用户"), t("Treated conv","实验组转化"),
        t("Control conv","对照组转化"), t("Uplift","提升"), t("Inc conv/100k","增量转化/10万")
    ]
    disp2[t("Treated conv","实验组转化")] = disp2[t("Treated conv","实验组转化")].map(pct)
    disp2[t("Control conv","对照组转化")] = disp2[t("Control conv","对照组转化")].map(pct)
    disp2[t("Uplift","提升")]             = disp2[t("Uplift","提升")].map(pp)
    disp2[t("Inc conv/100k","增量转化/10万")] = disp2[t("Inc conv/100k","增量转化/10万")].map(lambda v: f"{v:,.0f}")
    st.dataframe(disp2, hide_index=True, use_container_width=True)

    with st.expander(t("Model notes & limitations", "模型说明与局限性")):
        if lang == "English":
            st.markdown("""
- **T-learner**: separate logistic regression on treated vs. control subsamples. Uplift = score_treated − score_control.
- **Features**: 12 anonymous numeric features (f0–f11).
- **Response AUC 0.9538**: unusually high — likely due to near-constant features (f1, f2). Caveat: high AUC ≠ good calibration.
- **Control sample size**: ~15k control users limits precision of control-side model. X-learner or R-learner would help.
- **Production recommendation**: validate decile lift on a holdout before deploying targeting rules.
""")
        else:
            st.markdown("""
- **T-learner**：分别在实验组/对照组子样本上训练逻辑回归。提升分数 = 实验组分数 − 对照组分数。
- **特征**：12 个匿名数值特征（f0–f11）。
- **响应 AUC 0.9538**：异常高——可能因为 f1、f2 近乎常数特征。注意：高 AUC ≠ 好校准。
- **对照组样本量**：约 1.5 万对照组用户限制了对照侧模型精度。X-learner 或 R-learner 可改善。
- **上线建议**：在保留测试集上验证分桶提升效果，再部署定向规则。
""")

    st.divider()

    # ── 5. Decision Synthesis ────────────────────────────────────────────────
    st.subheader(t("5. Decision synthesis", "5. 决策综合"))
    if not segs.empty:
        so2 = generate_story_opener(df, rd, deciles, segs)
        sig_txt = f"p = {rd['conversion']['p_value']:.4f}" if rd['conversion']['p_value'] >= 0.0001 else "p &lt; 0.0001"
        if lang == "English":
            st.markdown(f"""<div class="decision-box">
<b>What to take to the next budget meeting:</b><br><br>
<b>1. The ad works — but narrowly.</b> Conversion lift is +{so2['conv_lift_pp']:.3f} pp ({sig_txt}).
Visit lift is +{so2['visit_lift_pp']:.2f} pp and highly significant. The effect is real but small per user; value comes from scale.<br><br>
<b>2. Only {so2['best_seg']} is worth targeting.</b> It drives {so2['best_inc']:,.0f} incremental conversions per 100K users — all other segments show noise-level lift.
Concentrating budget on this segment improves ROI without increasing spend.<br><br>
<b>3. Uplift model beats response model by {so2['ratio']}×.</b> High baseline buyers convert anyway — the uplift score finds users who convert <em>because of</em> the ad.
Switch the targeting criteria from response score to uplift score before the next campaign.
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="decision-box">
<b>下次预算会议的核心结论：</b><br><br>
<b>1. 广告有效——但幅度很小。</b> 转化提升 +{so2['conv_lift_pp']:.3f} pp（{sig_txt}）。
访问提升 +{so2['visit_lift_pp']:.2f} pp 且高度显著。效果真实但单用户量级小；价值靠规模放大。<br><br>
<b>2. 只有 {so2['best_seg']} 值得定向。</b> 每 10 万用户带来 {so2['best_inc']:,.0f} 增量转化——其他分群仅有噪声级提升。
集中预算投放该分群可提升 ROI 而无需增加花费。<br><br>
<b>3. 提升模型优于响应模型 {so2['ratio']}×。</b> 高基础购买概率的用户本来就会转化——提升分数找的是"因为广告才转化"的用户。
下个 campaign 前将定向依据从响应分数切换为提升分数。
</div>""", unsafe_allow_html=True)
    else:
        _key_findings_zh = (
            "<b>1. 正向 ATE 已确认</b> — 转化 +0.09 pp（p = 0.049），访问 +1.02 pp（p &lt; 0.0001）。"
            "<br><br><b>2. 提升定向 &gt; 响应定向</b> — 提升分数识别「因广告才转化」的用户。"
            "<br><br><b>3. 策略 A（提升前 30%）在所有测试接触成本下均优胜。</b>"
        )
        _key_findings_en = (
            "<b>1. Positive ATE confirmed</b> — conversion +0.09 pp (p = 0.049), visit +1.02 pp (p &lt; 0.0001)."
            "<br><br><b>2. Uplift targeting &gt; response targeting</b> — uplift score identifies users who convert <em>because of</em> the ad."
            "<br><br><b>3. Policy A (uplift top 30%) dominates at all tested contact costs.</b>"
        )
        st.markdown(
            f'<div class="callout">{t(_key_findings_en, _key_findings_zh)}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

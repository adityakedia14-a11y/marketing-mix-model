"""
Interactive Marketing Mix Model dashboard.

Run:  streamlit run app/app.py
(Needs the outputs/ folder produced by src/train.py and src/simulate.py.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optimizer import channel_revenue, load_params, optimize_budget, revenue_samples  # noqa: E402
from mmm import weekly_response  # noqa: E402

OUT = ROOT / "outputs"
LABELS = {"search": "Google Search", "shopping_pmax": "Google Shopping/PMax",
          "meta": "Meta (FB+IG)", "video": "Video (TikTok+YouTube)"}
COLORS = {"search": "#8b5cf6", "shopping_pmax": "#b45309", "meta": "#ec4899", "video": "#6b7280",
          "baseline": "#3b82f6", "bfcm": "#f97316", "gift_season": "#22c55e", "discount_rate": "#ef4444"}

st.set_page_config(page_title="Marketing Mix Model", page_icon="📊", layout="wide")


@st.cache_data
def load_all():
    weekly = pd.read_csv(ROOT / "data" / "processed" / "weekly.csv", parse_dates=["week"])
    decomp = pd.read_csv(OUT / "weekly_decomposition.csv", parse_dates=["week"])
    roi = pd.read_csv(OUT / "roi_by_channel.csv")
    metrics = json.load(open(OUT / "metrics.json"))
    sim = pd.read_csv(OUT / "simulation_recovery.csv") if (OUT / "simulation_recovery.csv").exists() else None
    return weekly, decomp, roi, metrics, sim


@st.cache_resource
def params():
    return load_params()


weekly, decomp, roi, metrics, sim = load_all()
P = params()
CH = list(P["channels"])
current = P["avg_weekly_spend"]

st.title("📊 Marketing Mix Model — US Food & Drink eCommerce brand")
st.caption("Real, anonymised data (Conjura MMM dataset, CC BY 4.0) · Bayesian MMM built from scratch in PyMC · "
           f"{len(weekly)} weeks, {weekly.week.min():%b %Y} – {weekly.week.max():%b %Y}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Channel ROI", "Response curves", "Budget planner", "Validation"])

# ------------------------------------------------------------------ Overview
with tab1:
    spend = weekly[CH].sum().sum()
    rev = weekly.revenue.sum()
    media = decomp[CH].sum().sum()
    c = st.columns(5)
    c[0].metric("Total ad spend", f"${spend / 1e6:.1f}M")
    c[1].metric("Total revenue", f"${rev / 1e6:.1f}M")
    c[2].metric("Blended ROAS", f"{rev / spend:.2f}")
    c[3].metric("Revenue driven by paid media", f"{media / rev:.0%}")
    c[4].metric("Model fit (R²)", f"{metrics['in_sample']['r2']:.2f}")

    st.subheader("Where weekly revenue came from")
    parts = ["baseline", "bfcm", "gift_season", "discount_rate"] + CH
    names = {"baseline": "Baseline (no ads)", "bfcm": "Black Friday/Cyber Monday",
             "gift_season": "December gift season", "discount_rate": "Discounts", **LABELS}
    fig = go.Figure()
    for p_ in parts:
        fig.add_trace(go.Scatter(x=decomp.week, y=decomp[p_].clip(lower=0), stackgroup="one",
                                 name=names[p_], line=dict(width=0.5, color=COLORS[p_])))
    fig.add_trace(go.Scatter(x=decomp.week, y=decomp.actual, name="Actual revenue",
                             line=dict(color="black", width=2)))
    fig.update_layout(height=430, yaxis_title="USD / week", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.info("**How to read this:** the model splits each week's revenue into what would have happened "
            "anyway (baseline), seasonal events, and the extra revenue each ad channel created.")

# ------------------------------------------------------------------ ROI
with tab2:
    st.subheader("Return on investment by channel")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=roi.label, y=roi.roi, name="Average ROI",
                         marker_color=[COLORS[c] for c in roi.channel],
                         error_y=dict(type="data", symmetric=False,
                                      array=roi.roi_high - roi.roi, arrayminus=roi.roi - roi.roi_low)))
    fig.add_hline(y=1, line_dash="dash", line_color="grey", annotation_text="break-even (revenue)")
    fig.update_layout(height=380, yaxis_title="USD revenue per USD spent", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    margin = st.slider("Gross margin % (to convert revenue ROI into profit ROI)", 10, 90, 40, 5) / 100
    t = roi[["label", "spend_share", "contribution_share", "roi", "roi_low", "roi_high", "marginal_roi"]].copy()
    t["profit_roi"] = t.roi * margin - 1
    t.columns = ["Channel", "Share of spend", "Share of revenue", "ROI", "ROI low (3%)", "ROI high (97%)",
                 "Marginal ROI", "Profit per $1 (after margin)"]
    st.dataframe(t.style.format({"Share of spend": "{:.0%}", "Share of revenue": "{:.0%}", "ROI": "{:.2f}",
                                 "ROI low (3%)": "{:.2f}", "ROI high (97%)": "{:.2f}",
                                 "Marginal ROI": "{:.2f}", "Profit per $1 (after margin)": "{:+.2f}"}),
                 hide_index=True, width="stretch")
    st.info("**ROI** = total revenue the channel created ÷ what we spent on it. "
            "**Marginal ROI** = revenue from the *next* dollar — this is what matters for reallocating budget. "
            "Bars show the 94% uncertainty range: wide bars mean the data can't pin that channel down precisely.")

# ------------------------------------------------------------------ Response curves
with tab3:
    st.subheader("Response curves — weekly spend vs. weekly revenue")
    ch = st.selectbox("Channel", CH, format_func=lambda c: LABELS[c])
    i = CH.index(ch)
    grid = np.linspace(0, P["x_scale"][i] * 1.5, 120)
    resp = np.array([weekly_response(s, P["lam"][:, i], P["beta"][:, i], P["x_scale"][i], P["y_scale"]) for s in grid])
    mean, lo, hi = resp.mean(1), np.quantile(resp, 0.03, axis=1), np.quantile(resp, 0.97, axis=1)
    fig = go.Figure([
        go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0),
                   fillcolor="rgba(59,130,246,0.2)", name="94% range"),
        go.Scatter(x=grid, y=mean, line=dict(color="#1d4ed8", width=3), name="Expected revenue"),
        go.Scatter(x=grid, y=grid, line=dict(color="grey", dash="dash"), name="Break-even (revenue = spend)"),
    ])
    fig.add_vline(x=current[i], line_dash="dot", annotation_text="current avg spend")
    fig.add_vline(x=weekly[ch].max(), line_dash="dot", line_color="red", annotation_text="max ever spent",
                  annotation_position="bottom right")
    fig.update_layout(height=430, xaxis_title="Weekly spend (USD)", yaxis_title="Weekly incremental revenue (USD)")
    st.plotly_chart(fig, width="stretch")
    st.info("The curve bends because of **diminishing returns**. Beyond the red line the model is "
            "extrapolating — it has never seen spend that high, so treat that region with caution.")

# ------------------------------------------------------------------ Budget planner
with tab4:
    st.subheader("Budget planner")
    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**1. Try your own split** (weekly spend, USD)")
        plan = np.array([st.slider(LABELS[c], 0, int(weekly[c].max()), int(current[k]), 500, key=f"s_{c}")
                         for k, c in enumerate(CH)], dtype=float)
        today = np.array([int(x) for x in current], dtype=float)   # same rounding as the sliders
        base = revenue_samples(today, P)
        mine = revenue_samples(plan, P)
        d = mine - base
        st.metric("Media-driven revenue / week", f"${mine.mean():,.0f}",
                  f"{d.mean():+,.0f} vs today (spend {plan.sum() - today.sum():+,.0f})")
        if np.allclose(plan, today):
            st.caption("Move the sliders to compare a new split against today's average weekly spend.")
        else:
            st.caption(f"94% range of the change: ${np.quantile(d, .03):,.0f} to ${np.quantile(d, .97):,.0f} · "
                       f"chance it beats today: {(d > 0).mean():.0%}")

        st.markdown("---\n**2. Let the optimizer choose**")
        budget = st.number_input("Weekly budget (USD)", 10_000, 300_000, int(round(current.sum(), -3)), 1000)
        max_change = st.slider("Max change per channel (guardrail)", 0.1, 1.0, 0.3, 0.05,
                               help="Limits how far each channel can move from today's split. "
                                    "Keeps recommendations within the range the model has evidence for.")
        run = st.button("Optimize split", type="primary")

    with right:
        if run:
            r = optimize_budget(P, current, budget=budget, max_change=max_change)
            res = pd.DataFrame({"Channel": [LABELS[c] for c in CH],
                                "Today (scaled to budget)": r["current_split"],
                                "Recommended": r["optimal_split"]})
            res["Change"] = res.Recommended / res["Today (scaled to budget)"] - 1
            fig = go.Figure([
                go.Bar(x=res.Channel, y=res["Today (scaled to budget)"], name="Today", marker_color="#9ca3af"),
                go.Bar(x=res.Channel, y=res.Recommended, name="Recommended", marker_color="#2563eb")])
            fig.update_layout(barmode="group", height=360, yaxis_title="Weekly spend (USD)")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(res.style.format({"Today (scaled to budget)": "${:,.0f}", "Recommended": "${:,.0f}",
                                           "Change": "{:+.0%}"}), hide_index=True, width="stretch")
            m = st.columns(3)
            m[0].metric("Extra revenue / week", f"${r['uplift_mean']:,.0f}")
            m[1].metric("Extra revenue / year", f"${r['uplift_mean'] * 52 / 1e6:,.2f}M")
            m[2].metric("Chance it beats today's split", f"{r['prob_better']:.0%}")
            st.caption(f"94% range of weekly uplift: ${r['uplift_low']:,.0f} to ${r['uplift_high']:,.0f}")
        else:
            st.info("Set a budget and press **Optimize split**. The optimizer maximises expected revenue "
                    "averaged over 4,000 posterior samples, so the plan is robust to model uncertainty.")

# ------------------------------------------------------------------ Validation
with tab5:
    st.subheader("Can we trust it?")
    h = metrics["holdout"]
    c = st.columns(4)
    c[0].metric("Holdout weeks (never seen)", metrics["holdout_weeks"])
    c[1].metric("Holdout error (MAPE)", f"{h['test']['mape_pct']:.1f}%")
    c[2].metric("In-sample R²", f"{metrics['in_sample']['r2']:.2f}")
    c[3].metric("Sampler r-hat (≈1 is good)", f"{metrics['diagnostics']['max_r_hat']:.3f}")
    st.image(str(OUT / "holdout_fit.png"), caption="Trained on weeks before the dashed line, forecasting the rest.")
    if sim is not None:
        st.markdown("**Parameter recovery on simulated data** — real spend patterns, invented 'true' ROI. "
                    "Did the model find the truth?")
        st.image(str(OUT / "simulation_recovery.png"))
        st.dataframe(sim[["channel", "true_roi", "est_roi", "roi_low", "roi_high", "roi_recovered"]]
                     .round(2), hide_index=True)

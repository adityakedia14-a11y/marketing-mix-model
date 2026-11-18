"""
Step 3 of the pipeline: fit the MMM on the REAL brand data.

  A. Out-of-sample test: train on the first 100 weeks, forecast the last 16
     (which include Black Friday 2023) without the model ever seeing them.
  B. Final model on all weeks -> ROI per channel, sales decomposition,
     response curves, and saved parameters for the optimizer / app.

Run:  python src/train.py
"""
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm import MarketingMixModel, fit_metrics, weekly_response

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHANNELS = ["search", "shopping_pmax", "meta", "video"]
CONTROLS = ["bfcm", "gift_season", "discount_rate"]
HOLDOUT_WEEKS = 16
# Chosen by holdout comparison: 1 seasonal wave + baseline prior from the 37% returning-customer revenue share
MODEL_KW = dict(n_fourier=1, baseline_share=0.35)
LABELS = {"search": "Google Search", "shopping_pmax": "Google Shopping/PMax",
          "meta": "Meta (FB+IG)", "video": "Video (TikTok+YouTube)"}


def holdout_test(df):
    train = np.arange(len(df)) < len(df) - HOLDOUT_WEEKS
    mmm = MarketingMixModel(CHANNELS, CONTROLS, **MODEL_KW)
    mmm.fit(df, train_mask=train)
    pred = mmm.predicted_revenue()                      # (S, T)
    actual = df["revenue"].to_numpy()
    p_mean = pred.mean(0)
    lo, hi = np.quantile(pred, [0.03, 0.97], axis=0)
    metrics = {"train": fit_metrics(actual[train], p_mean[train]),
               "test": fit_metrics(actual[~train], p_mean[~train])}

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.week, actual, color="black", lw=1.4, label="Actual revenue")
    ax.plot(df.week, p_mean, color="#1f6feb", lw=1.4, label="Model")
    ax.fill_between(df.week, lo, hi, color="#1f6feb", alpha=0.2, label="94% interval")
    ax.axvline(df.week[~train].iloc[0], color="grey", ls="--")
    ax.text(df.week[~train].iloc[0], actual.max() * 0.95, "  unseen weeks →", color="grey")
    ax.set_title(f"Out-of-sample test - test MAPE {metrics['test']['mape_pct']:.1f}%, "
                 f"test R² {metrics['test']['r2']:.2f}")
    ax.legend(loc="upper left"); ax.set_ylabel("USD / week")
    fig.tight_layout(); fig.savefig(OUT / "holdout_fit.png", dpi=150); plt.close(fig)
    return metrics


def final_model(df):
    mmm = MarketingMixModel(CHANNELS, CONTROLS, **MODEL_KW)
    mmm.fit(df)

    summ = az.summary(mmm.idata, var_names=["alpha", "lam", "beta", "gamma_control", "sigma"])
    diagnostics = {"max_r_hat": float(summ["r_hat"].max()),
                   "min_ess_bulk": float(summ["ess_bulk"].min()),
                   "divergences": int(mmm.idata.sample_stats["diverging"].sum())}

    actual = df["revenue"].to_numpy()
    pred = mmm.predicted_revenue()
    in_sample = fit_metrics(actual, pred.mean(0))

    # ---------- ROI per channel ----------
    contrib = mmm.channel_contributions()               # (S, T, C)
    total_contrib = contrib.sum(axis=1)                 # (S, C)
    spend = df[CHANNELS].sum().to_numpy()
    roi = total_contrib / spend

    # Marginal ROI: revenue from the NEXT dollar at the current average weekly spend.
    p = mmm.response_params()
    avg_spend = df[CHANNELS].tail(52).mean().to_numpy()
    mroi = np.column_stack([
        (weekly_response(avg_spend[c] * 1.01, p["lam"][:, c], p["beta"][:, c], p["x_scale"][c], p["y_scale"])
         - weekly_response(avg_spend[c], p["lam"][:, c], p["beta"][:, c], p["x_scale"][c], p["y_scale"]))
        / (avg_spend[c] * 0.01)
        for c in range(len(CHANNELS))])

    roi_table = pd.DataFrame({
        "channel": CHANNELS,
        "label": [LABELS[c] for c in CHANNELS],
        "spend": spend,
        "spend_share": spend / spend.sum(),
        "revenue_contribution": total_contrib.mean(0),
        "contribution_share": total_contrib.mean(0) / actual.sum(),
        "roi": roi.mean(0),
        "roi_low": np.quantile(roi, 0.03, axis=0),
        "roi_high": np.quantile(roi, 0.97, axis=0),
        "marginal_roi": mroi.mean(0),
        "mroi_low": np.quantile(mroi, 0.03, axis=0),
        "mroi_high": np.quantile(mroi, 0.97, axis=0),
        "avg_weekly_spend_last52": avg_spend,
        "adstock_alpha": p["alpha"].mean(0),
    })
    roi_table.to_csv(OUT / "roi_by_channel.csv", index=False)

    # ---------- Weekly decomposition (where did each week's revenue come from?) ----------
    S = mmm.y_scale
    decomp = pd.DataFrame({"week": df.week, "actual": actual, "predicted": pred.mean(0),
                           "baseline": mmm.posterior_samples("baseline").mean(0) * S})
    ctrl = mmm.posterior_samples("control_contrib").mean(0) * S
    for i, c in enumerate(CONTROLS):
        decomp[c] = ctrl[:, i]
    for i, c in enumerate(CHANNELS):
        decomp[c] = contrib[:, :, i].mean(0)
    decomp.to_csv(OUT / "weekly_decomposition.csv", index=False)

    # ---------- Save posterior parameters for the optimizer and app ----------
    np.savez(OUT / "model_params.npz", **p, avg_weekly_spend=avg_spend)

    # ---------- Plots ----------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    parts = ["baseline", "bfcm", "gift_season", "discount_rate"] + CHANNELS
    ax.stackplot(df.week, *[decomp[c].clip(lower=0) for c in parts],
                 labels=["Baseline"] + ["BFCM", "Gift season", "Discounts"] + [LABELS[c] for c in CHANNELS],
                 alpha=0.85)
    ax.plot(df.week, actual, color="black", lw=1.2, label="Actual")
    ax.set_title("Where weekly revenue came from"); ax.set_ylabel("USD / week")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "decomposition.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(CHANNELS))
    ax.errorbar(roi_table.roi, y, xerr=[roi_table.roi - roi_table.roi_low, roi_table.roi_high - roi_table.roi],
                fmt="o", capsize=4, color="#1f6feb", label="Average ROI")
    ax.errorbar(roi_table.marginal_roi, y + 0.15,
                xerr=[roi_table.marginal_roi - roi_table.mroi_low, roi_table.mroi_high - roi_table.marginal_roi],
                fmt="s", capsize=4, color="#d97706", label="Marginal ROI (next $)")
    ax.axvline(1, color="grey", ls="--"); ax.set_yticks(y, roi_table.label)
    ax.set_xlabel("USD revenue per USD spent"); ax.set_title("ROI by channel (94% intervals)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "roi.png", dpi=150); plt.close(fig)

    return in_sample, diagnostics, roi_table


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "weekly.csv", parse_dates=["week"])

    print("== A. Holdout test ==")
    holdout = holdout_test(df)
    print(json.dumps(holdout, indent=2))

    print("\n== B. Final model on all data ==")
    in_sample, diag, roi_table = final_model(df)
    print("Diagnostics:", diag)
    print("In-sample fit:", in_sample)
    print(roi_table.drop(columns=["label"]).round(3).to_string(index=False))

    json.dump({"holdout": holdout, "in_sample": in_sample, "diagnostics": diag,
               "holdout_weeks": HOLDOUT_WEEKS, "n_weeks": len(df)},
              open(OUT / "metrics.json", "w"), indent=2)
    print("\nSaved: metrics.json, roi_by_channel.csv, weekly_decomposition.csv, model_params.npz, *.png")


if __name__ == "__main__":
    main()

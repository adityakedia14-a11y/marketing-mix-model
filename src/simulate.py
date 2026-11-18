"""
Step 2 of the pipeline: PARAMETER-RECOVERY TEST.

Real data has no answer key - nobody knows the true ROI of each channel.
So before trusting the model on real data we:
  1. take the brand's REAL weekly spend patterns,
  2. invent "true" channel effects (adstock, saturation, ROI) ourselves,
  3. generate fake revenue from them (+ noise),
  4. fit the same model and check it recovers the truth.

Run:  python src/simulate.py
Output: outputs/simulation_recovery.csv, outputs/simulation_recovery.png
"""
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm import MarketingMixModel, fourier_features, geometric_adstock, logistic_saturation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHANNELS = ["search", "shopping_pmax", "meta", "video"]
CONTROLS = ["bfcm", "gift_season", "discount_rate"]

# ---------------- The "truth" we hide inside the fake data ----------------
TRUE = {
    "alpha": np.array([0.10, 0.30, 0.45, 0.20]),   # carry-over per channel
    "lam":   np.array([3.0, 2.0, 1.5, 4.0]),       # saturation steepness
    "beta":  np.array([0.10, 0.12, 0.30, 0.03]),   # max effect (share of max revenue)
}
L_MAX = 6
Y_SCALE = 750_000          # roughly the brand's peak weekly revenue
NOISE = 0.03               # noise sd in scaled units (~$22K/week)


def simulate(df: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    X = df[CHANNELS].to_numpy() / df[CHANNELS].max().to_numpy()
    T = len(df)
    t = np.linspace(0, 1, T)

    contrib = np.column_stack([
        TRUE["beta"][c] * logistic_saturation(geometric_adstock(X[:, c], TRUE["alpha"][c], L_MAX),
                                              TRUE["lam"][c])
        for c in range(len(CHANNELS))
    ])
    F = fourier_features(df["week"], 2)
    baseline = 0.20 + 0.05 * t + F @ np.array([0.02, 0.03, 0.0, 0.01])
    controls = 0.15 * df["bfcm"].to_numpy() + 0.05 * df["gift_season"].to_numpy()
    y = baseline + contrib.sum(1) + controls + rng.normal(0, NOISE, T)

    sim = df.copy()
    sim["revenue"] = y * Y_SCALE
    spend = df[CHANNELS].sum().to_numpy()
    true_roi = contrib.sum(0) * Y_SCALE / spend
    return sim, {"roi": true_roi, "alpha": TRUE["alpha"]}


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "weekly.csv", parse_dates=["week"])
    sim, truth = simulate(df)

    mmm = MarketingMixModel(CHANNELS, CONTROLS, l_max=L_MAX)
    mmm.fit(sim)

    # Sampler health check: r_hat ~ 1.00 means the 4 chains agree.
    summ = az.summary(mmm.idata, var_names=["alpha", "lam", "beta"])
    print("Max r_hat:", summ["r_hat"].max(), " | min effective sample size:", summ["ess_bulk"].min())

    contrib = mmm.channel_contributions().sum(axis=1)            # (S, C) total USD
    roi = contrib / sim[CHANNELS].sum().to_numpy()
    alpha = mmm.posterior_samples("alpha")

    rows = []
    for i, ch in enumerate(CHANNELS):
        lo, hi = np.quantile(roi[:, i], [0.03, 0.97])
        alo, ahi = np.quantile(alpha[:, i], [0.03, 0.97])
        rows.append({
            "channel": ch,
            "true_roi": truth["roi"][i], "est_roi": roi[:, i].mean(), "roi_low": lo, "roi_high": hi,
            "roi_recovered": lo <= truth["roi"][i] <= hi,
            "true_alpha": truth["alpha"][i], "est_alpha": alpha[:, i].mean(),
            "alpha_low": alo, "alpha_high": ahi,
            "alpha_recovered": alo <= truth["alpha"][i] <= ahi,
        })
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "simulation_recovery.csv", index=False)
    print(res.round(3).to_string(index=False))

    # Plot: true ROI vs estimated ROI with 94% interval
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(CHANNELS))
    ax.errorbar(res.est_roi, y, xerr=[res.est_roi - res.roi_low, res.roi_high - res.est_roi],
                fmt="o", color="#1f6feb", capsize=4, label="Model estimate (94% interval)")
    ax.scatter(res.true_roi, y, marker="x", s=90, color="#d1242f", zorder=3, label="True ROI")
    ax.set_yticks(y, CHANNELS)
    ax.set_xlabel("ROI (USD revenue per USD spent)")
    ax.set_title("Parameter recovery on simulated data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "simulation_recovery.png", dpi=150)
    print("Saved outputs/simulation_recovery.csv and .png")


if __name__ == "__main__":
    main()

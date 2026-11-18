"""
Step 4 of the pipeline: BUDGET OPTIMIZER.

Question it answers: "Same weekly budget - how should we split it across channels
to get the most revenue?"

How:
  * For each channel the model gives a response curve: weekly spend -> weekly revenue.
  * We maximise the SUM of the curves subject to:
      - total spend == budget
      - each channel stays within +/- max_change of today's spend (guardrails:
        the model has only seen spend levels close to history, so we don't let
        the optimizer jump far outside what it has evidence for).
  * We average over all posterior samples, so the plan is good on average across
    every plausible version of the model - and we report the probability that the
    new plan beats the current one.

Run:  python src/optimizer.py
Output: outputs/optimized_plan.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mmm import weekly_response

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "outputs" / "model_params.npz"


def load_params(path=PARAMS) -> dict:
    p = dict(np.load(path, allow_pickle=True))
    p["y_scale"] = float(p["y_scale"])
    return p


def revenue_samples(spend: np.ndarray, p: dict) -> np.ndarray:
    """Posterior samples (S,) of total weekly media-driven revenue for a spend plan (C,)."""
    total = 0.0
    for c in range(len(p["channels"])):
        total = total + weekly_response(spend[c], p["lam"][:, c], p["beta"][:, c], p["x_scale"][c], p["y_scale"])
    return total


def channel_revenue(spend: np.ndarray, p: dict) -> np.ndarray:
    """Expected weekly revenue per channel (C,)."""
    return np.array([
        weekly_response(spend[c], p["lam"][:, c], p["beta"][:, c], p["x_scale"][c], p["y_scale"]).mean()
        for c in range(len(p["channels"]))])


def optimize_budget(p: dict, current: np.ndarray, budget: float | None = None,
                    max_change: float = 0.3) -> dict:
    current = np.asarray(current, dtype=float)
    budget = float(current.sum() if budget is None else budget)
    scale = budget / current.sum()           # if the budget changes, guardrails move with it
    lo = current * scale * (1 - max_change)
    hi = current * scale * (1 + max_change)
    # Work in thousands of dollars so the solver's steps and tolerances are well-scaled.
    K = 1000.0
    bounds = list(zip(lo / K, hi / K))
    obj = lambda k: -revenue_samples(k * K, p).mean() / K
    cons = [{"type": "eq", "fun": lambda k: k.sum() - budget / K}]
    x0 = current * scale / K
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    new = res.x * K

    base_rev = revenue_samples(current * scale, p)    # same budget, today's split
    new_rev = revenue_samples(new, p)
    uplift = new_rev - base_rev
    return {
        "success": bool(res.success),
        "budget": budget,
        "current_split": current * scale,
        "optimal_split": new,
        "current_media_revenue": float(base_rev.mean()),
        "optimal_media_revenue": float(new_rev.mean()),
        "uplift_mean": float(uplift.mean()),
        "uplift_low": float(np.quantile(uplift, 0.03)),
        "uplift_high": float(np.quantile(uplift, 0.97)),
        "prob_better": float((uplift > 0).mean()),
    }


def main():
    p = load_params()
    current = p["avg_weekly_spend"]
    r = optimize_budget(p, current, max_change=0.3)

    plan = pd.DataFrame({
        "channel": p["channels"],
        "current_weekly_spend": r["current_split"],
        "optimal_weekly_spend": r["optimal_split"],
    })
    plan["change_pct"] = (plan.optimal_weekly_spend / plan.current_weekly_spend - 1) * 100
    plan["current_weekly_revenue"] = channel_revenue(r["current_split"], p)
    plan["optimal_weekly_revenue"] = channel_revenue(r["optimal_split"], p)
    plan.to_csv(ROOT / "outputs" / "optimized_plan.csv", index=False)

    print(plan.round(0).to_string(index=False))
    print(f"\nWeekly budget: ${r['budget']:,.0f} (unchanged)")
    print(f"Media-driven revenue: ${r['current_media_revenue']:,.0f} -> ${r['optimal_media_revenue']:,.0f} per week")
    print(f"Uplift: ${r['uplift_mean']:,.0f}/week  (94% range ${r['uplift_low']:,.0f} to ${r['uplift_high']:,.0f})"
          f"  ~ ${r['uplift_mean'] * 52:,.0f}/year")
    print(f"Probability the new plan beats the current one: {r['prob_better']:.0%}")


if __name__ == "__main__":
    main()

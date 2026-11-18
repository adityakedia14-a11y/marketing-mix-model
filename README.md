# Bayesian Marketing Mix Model & Budget Optimizer

**Which marketing channels actually drive sales — and how should the budget be re-split to get more revenue for the same spend?**

A Bayesian Marketing Mix Model (MMM) built from scratch in PyMC on **real, anonymised data** from a US Food & Drink eCommerce brand, with a validated budget optimizer and an interactive Streamlit planner.

## Headline result

> With the **same weekly budget (~$83K)**, shifting spend from Video and Google Search into Google Shopping/Performance Max is expected to add **~$30K revenue per week (~$1.5M per year)**. The model gives that plan a **~99% probability** of beating the current split. The plan is held within ±30% guardrails per channel.

| Channel | Share of spend | Share of revenue driven | ROI ($ revenue per $1) | 94% interval |
|---|---|---|---|---|
| Google Shopping / PMax | 16% | 37% | **6.7** | 3.0 – 10.4 |
| Meta (Facebook + Instagram) | **57%** | 27% | 1.4 | 0.7 – 2.2 |
| Google Search (non-brand) | 17% | 5% | 0.8 | 0.03 – 2.5 |
| Video (TikTok + YouTube) | 10% | 1% | 0.4 | 0.01 – 1.4 |

**Insight:** most of the budget goes to Meta, but Shopping/PMax returns roughly 5× more per dollar. Search and Video are at or below revenue break-even at current spend.

## Data

[Conjura Multi-Region MMM Dataset](https://doi.org/10.6084/m9.figshare.25314841) (Anderson, 2024, CC BY 4.0) contains daily marketing and sales data for ~100 anonymised eCommerce brands.

* **Brand selection:** I screened all 143 timeseries and rejected brands where:
  * channels were highly collinear (e.g. a beauty brand whose 3 channels were 95% correlated, so their effects can't be separated), or
  * revenue grew explosively (100×), so growth would be confused with ad effect.

  The chosen brand has a maximum channel correlation of about 0.5 and stable growth.
* **Preparation:** 116 weeks (Oct 2021 – Jan 2024), $9.0M spend, $26.8M revenue.
* **Channel grouping:**
  * Shopping + PMax are merged into one channel, because Google migrated Shopping campaigns into PMax in 2022 and the data shows one replacing the other.
  * Facebook + Instagram are merged into Meta.
  * TikTok + YouTube are merged into Video.
* **Controls:** Black Friday/Cyber Monday weeks, December gift season, discount rate, and yearly seasonality.

## Method

```
revenue[t] = baseline (intercept + trend + seasonality)
           + Σ_channels  β_c · saturation( adstock(spend_c) )
           + Σ_controls  γ_k · control_k
           + noise
```

* **Adstock (carry-over):** geometric decay over up to 6 weeks. An ad keeps working after the week it runs.
* **Saturation (diminishing returns):** logistic curve. Doubling spend does not double sales.
* **Bayesian inference** (PyMC, NUTS sampler, 4 chains × 1,000 draws):
  * Every estimate comes with an uncertainty interval.
  * Business knowledge enters through priors. Example: 37% of revenue comes from returning customers, so the baseline prior is centred at ~35% of revenue.
* **Optimizer** (SciPy SLSQP):
  * Maximises expected revenue **averaged over all 4,000 posterior samples**, so the plan is robust to model uncertainty.
  * Keeps total spend fixed.
  * Applies ±30% guardrails per channel, so it never recommends spend levels the model has no evidence for.

## Validation — "how do you know it's right?"

1. **Parameter recovery:**
   * I kept the brand's real spend patterns, invented "true" channel ROIs, and generated synthetic revenue from them.
   * The model recovered **all 4 true ROIs and all 4 true adstock rates inside its 94% intervals**.
   * The intervals are wide. With about 2 years of weekly data, the model ranks channels reliably but cannot give precise ROI values.
2. **Out-of-sample forecast:**
   * Trained on the first 100 weeks and forecast the last 16, including Black Friday 2023, which it had never seen.
   * Holdout error (MAPE) was 23.7%. It captured the timing of the holiday peak but under-predicted its record height.
3. **In-sample fit:** R² = 0.84, MAPE = 15%.
4. **Sampler diagnostics:** r-hat ≈ 1.00 and 0 divergences, so the 4 independent chains agree.

## Limitations (and what I'd do next)

* **PMax ROI is probably overstated.** PMax also serves branded-search and remarketing ads, which reach people who would likely have bought anyway. **Next step:** a geo-holdout lift test on PMax, with the result fed back into the model as a calibration prior.
* **Media takes credit for ~70% of revenue.** That's high. Better baseline controls (organic traffic, email, price, competitor activity) would sharpen the split.
* **Wide intervals for Search and Video.** More history, or deliberate spend variation (pulsing), would narrow them.
* **Revenue ≠ profit.** The app converts ROI to profit using a gross-margin input, but true margins are unknown in an anonymised dataset.
* **Scope.** Digital channels only (no TV or offline), and one brand. The pipeline is written to be re-run per brand (change `TIMESERIES_ID`).

## Project structure

```
src/data_prep.py   raw daily data -> clean weekly table for one brand
src/mmm.py         the Bayesian MMM (adstock, saturation, priors, likelihood)
src/simulate.py    parameter-recovery test on synthetic data
src/train.py       holdout test + final fit -> ROI, decomposition, saved posterior
src/optimizer.py   budget optimizer with guardrails and uncertainty
app/app.py         Streamlit dashboard: overview, ROI, response curves, budget planner, validation
outputs/           all results and charts
run_all.sh         reproduces everything (~30 seconds)
```

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh                         # downloads data, validates, fits, optimises
.venv/bin/streamlit run app/app.py   # interactive dashboard
```

**Tech:** Python, PyMC, ArviZ, NumPy, pandas, SciPy, Plotly, Streamlit.

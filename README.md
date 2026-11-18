# Marketing Mix Model

I built this project to answer a question most marketing teams have: which ad channels are actually bringing in sales, and could the same budget be split better?

It's a Bayesian Marketing Mix Model (MMM) written in PyMC. I trained it on two years of real weekly data from an anonymised US food & drink eCommerce brand, then used the model to build a budget optimizer and a small Streamlit dashboard for trying out different spend plans.

## Results

The brand spent about $9M on ads over the period. The model estimates the return on each channel as follows:

| Channel | Share of spend | ROI | 94% interval |
|---|---|---|---|
| Google Shopping / Performance Max | 16% | 6.7 | 3.0 to 10.4 |
| Meta (Facebook + Instagram) | 57% | 1.4 | 0.7 to 2.2 |
| Google Search | 17% | 0.8 | 0.03 to 2.5 |
| Video (TikTok + YouTube) | 10% | 0.4 | 0.01 to 1.4 |

ROI here means dollars of revenue per dollar spent.

The main finding is that more than half the budget goes to Meta, while Shopping/PMax brings back several times more per dollar. Search and Video look like they're around break-even or below at current spend levels.

Keeping the weekly budget the same (around $83K), the optimizer suggests moving money out of Search and Video and into Shopping/PMax. The model estimates this would add roughly $30K in revenue a week, or about $1.5M a year, and says there's around a 99% chance the new split beats the current one. No channel is allowed to move more than 30% from where it is today, because the model hasn't seen spend levels far outside that range.

These are model estimates, not measured results. The plan would need to be tested before anyone relied on it.

## Data

I used the [Conjura MMM dataset](https://doi.org/10.6084/m9.figshare.25314841) (Anderson, 2024, CC BY 4.0), which has daily ad spend and sales for around 100 eCommerce brands.

Choosing the brand took some care. My first pick was a beauty brand, but its three main channels moved together almost perfectly (about 0.95 correlation), so there was no way to separate their effects. Its revenue also grew about 100x over the period, which the model would mistake for ad impact. I then screened all 143 timeseries for low correlation between channels, steady growth and enough active weeks, and picked a brand that met all three.

Data preparation:
- Aggregated daily data into 116 weeks (Oct 2021 to Jan 2024).
- Combined Shopping and PMax into a single channel. Google moved Shopping campaigns over to PMax during 2022, and you can see one replacing the other in the data.
- Grouped Facebook and Instagram as Meta, and TikTok and YouTube as Video.
- Added controls for Black Friday / Cyber Monday, the December gift season, discount rate and yearly seasonality.

## How the model works

Weekly revenue is modelled as a baseline (what the brand would sell without ads) plus a contribution from each channel, plus the seasonal and promotional controls.

Each channel's contribution includes two effects:

- **Adstock.** An ad doesn't only work in the week it runs; some of the effect carries over into later weeks. I used geometric decay over up to 6 weeks.
- **Saturation.** Each extra dollar does less than the one before it, so doubling spend doesn't double sales. I used a logistic curve for this.

I chose a Bayesian approach so every estimate comes with an uncertainty range instead of a single number. It also lets me put business knowledge in through priors. For example, about 37% of this brand's revenue comes from returning customers, so I centred the baseline prior around 35% of revenue. The model was fit with the NUTS sampler (4 chains, 1,000 draws each).

The optimizer uses SciPy (SLSQP). It searches for the split that gives the highest expected revenue averaged across all 4,000 posterior samples, so the plan holds up across the range of plausible models instead of being tuned to one point estimate.

## Validation

I checked the model in four ways:

1. **Simulated data.** I kept the brand's real spend patterns, made up "true" ROIs, and generated fake revenue from them. The model recovered all four true ROIs within its 94% intervals. The intervals were wide, though. With two years of weekly data, it can rank channels reliably but can't pin down exact ROI values.
2. **Holdout test.** I trained on the first 100 weeks and forecast the last 16, which include Black Friday 2023. The error on those weeks was 23.7% MAPE. It caught the timing of the holiday spike but underestimated how high it went.
3. **In-sample fit.** R² is 0.84 and MAPE is 15%.
4. **Sampler checks.** r-hat is about 1.00 with no divergences, which means the four independent chains agree.

## Limitations

- **PMax's ROI is probably too high.** PMax also runs branded search and remarketing ads, which reach people who were likely to buy anyway. The right next step would be a geo lift test on PMax, with the result used to calibrate the model.
- **Media takes credit for about 70% of revenue,** which feels high. Data on organic traffic, email, pricing or competitors would help separate the baseline better.
- **Search and Video estimates are uncertain.** More history, or deliberately varying spend on those channels, would narrow the ranges.
- **Revenue isn't profit.** The dashboard lets you enter a gross margin to convert ROI into profit, but the real margins for this brand aren't known.
- **The scope is limited:** digital channels only and a single brand. To run it on another brand, change `TIMESERIES_ID` in `src/data_prep.py`.

## Project structure

```
src/data_prep.py   cleans the raw daily data into a weekly table for one brand
src/mmm.py         the model itself (adstock, saturation, priors, likelihood)
src/simulate.py    parameter recovery test on simulated data
src/train.py       holdout test, final fit, ROI and revenue breakdown
src/optimizer.py   budget optimizer
app/app.py         Streamlit dashboard
outputs/           saved results and charts
run_all.sh         runs the whole pipeline
```

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
.venv/bin/streamlit run app/app.py
```

`run_all.sh` downloads the dataset, runs the validation, fits the model and runs the optimizer. It takes about 30 seconds.

Built with Python, PyMC, ArviZ, pandas, NumPy, SciPy, Plotly and Streamlit.

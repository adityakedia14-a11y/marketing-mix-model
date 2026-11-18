"""
The Bayesian Marketing Mix Model, written from scratch in PyMC.

    revenue[t] = baseline[t]                      (intercept + trend + yearly seasonality)
               + sum_c  beta_c * saturation( adstock(spend_c)[t] )   (paid media)
               + sum_k  gamma_k * control_k[t]                       (BFCM, gift season, discounts)
               + noise

Two marketing "physics" ideas:
  * ADSTOCK   - an ad keeps working after the week it ran; its effect decays by a
                factor alpha each week (alpha=0: no memory, alpha=0.8: long memory).
  * SATURATION - diminishing returns: doubling spend does NOT double sales.
                 We use the logistic curve  (1 - e^(-lam*x)) / (1 + e^(-lam*x)).

Everything is learned with Bayesian inference (MCMC), so each number comes with
an uncertainty range instead of a single point estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# --------------------------------------------------------------------------- #
# The two transformations, in plain numpy (used for simulation, optimizer, app)
# --------------------------------------------------------------------------- #

def adstock_weights(alpha: float, l_max: int) -> np.ndarray:
    """Weights for 'this week, last week, 2 weeks ago...', normalised to sum to 1."""
    w = alpha ** np.arange(l_max)
    return w / w.sum()


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """Carry-over: today's effective spend = weighted sum of recent weeks' spend."""
    w = adstock_weights(alpha, l_max)
    out = np.zeros_like(x, dtype=float)
    for lag, wl in enumerate(w):
        out[lag:] += wl * x[: len(x) - lag]
    return out


def logistic_saturation(x, lam):
    """Diminishing returns. x is spend scaled to ~[0, 1]; output is in [0, 1)."""
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def lagged_matrix(X: np.ndarray, l_max: int) -> np.ndarray:
    """(T, C) -> (T, l_max, C) where [t, l, c] = X[t - l, c] (0 before the start)."""
    T, C = X.shape
    out = np.zeros((T, l_max, C))
    for lag in range(l_max):
        out[lag:, lag, :] = X[: T - lag]
    return out


def fourier_features(dates: pd.Series, n_order: int) -> np.ndarray:
    """Smooth yearly seasonality as sin/cos waves of the day of year."""
    doy = pd.to_datetime(dates).dt.dayofyear.to_numpy()
    feats = []
    for k in range(1, n_order + 1):
        feats += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(feats)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

class MarketingMixModel:
    def __init__(self, channels, controls, date_col="week", target_col="revenue",
                 l_max: int = 6, n_fourier: int = 2, baseline_share: float | None = None):
        """
        baseline_share: optional business prior - roughly what share of revenue would
        happen WITHOUT paid media (e.g. returning-customer share). None = uninformed.
        """
        self.baseline_share = baseline_share
        self.channels = list(channels)
        self.controls = list(controls)
        self.date_col = date_col
        self.target_col = target_col
        self.l_max = l_max
        self.n_fourier = n_fourier
        self.idata = None

    # ---- scaling: put every variable on a ~0-1 scale so priors are sensible ----
    def _prepare(self, df: pd.DataFrame):
        self.x_scale = df[self.channels].max().to_numpy().astype(float)
        self.y_scale = float(df[self.target_col].max())
        self.c_scale = df[self.controls].abs().max().replace(0, 1).to_numpy().astype(float)

        X = df[self.channels].to_numpy() / self.x_scale
        y = df[self.target_col].to_numpy() / self.y_scale
        Z = df[self.controls].to_numpy() / self.c_scale
        t = np.linspace(0, 1, len(df))
        F = fourier_features(df[self.date_col], self.n_fourier)
        return X, y, Z, t, F

    def build(self, df: pd.DataFrame, train_mask: np.ndarray | None = None) -> pm.Model:
        X, y, Z, t, F = self._prepare(df)
        T, C = X.shape
        if train_mask is None:
            train_mask = np.ones(T, dtype=bool)
        self.train_mask = train_mask
        X_lag = lagged_matrix(X, self.l_max)

        coords = {"week": np.arange(T), "channel": self.channels,
                  "control": self.controls, "fourier": np.arange(F.shape[1])}

        with pm.Model(coords=coords) as model:
            # ---------------- Priors (our beliefs BEFORE seeing data) ----------------
            if self.baseline_share is None:
                intercept = pm.Normal("intercept", 0.2, 0.2)
            else:
                # centre the baseline on (share x average revenue), +/- ~30%
                mu0 = self.baseline_share * y.mean()
                intercept = pm.Normal("intercept", mu0, mu0 * 0.3)
            trend = pm.Normal("trend", 0, 0.2)
            season = pm.Normal("season_coef", 0, 0.1, dims="fourier")
            gamma = pm.Normal("gamma_control", 0, 0.5, dims="control")

            # adstock decay: Beta(1,3) -> mostly short memory (digital ads), but data can say otherwise
            alpha = pm.Beta("alpha", 1, 3, dims="channel")
            # saturation steepness
            lam = pm.Gamma("lam", 3, 1, dims="channel")
            # channel effect size: positive only (ads can't reduce sales), same prior for every channel
            beta = pm.HalfNormal("beta", 0.5, dims="channel")
            sigma = pm.HalfNormal("sigma", 0.1)

            # ---------------- Transformations ----------------
            # adstock: weights[l, c] = alpha_c^l / sum_l alpha_c^l
            lags = pt.arange(self.l_max)[:, None]
            w = alpha[None, :] ** lags
            w = w / w.sum(axis=0, keepdims=True)
            x_adstock = (X_lag * w[None, :, :]).sum(axis=1)                      # (T, C)
            x_sat = (1 - pt.exp(-lam * x_adstock)) / (1 + pt.exp(-lam * x_adstock))

            channel_contrib = pm.Deterministic("channel_contrib", beta * x_sat, dims=("week", "channel"))
            control_contrib = pm.Deterministic("control_contrib", Z * gamma, dims=("week", "control"))
            baseline = pm.Deterministic(
                "baseline", intercept + trend * t + pt.dot(F, season), dims="week")

            mu = pm.Deterministic(
                "mu", baseline + channel_contrib.sum(axis=1) + control_contrib.sum(axis=1), dims="week")

            # ---------------- Likelihood: only on training weeks ----------------
            pm.Normal("y_obs", mu=mu[train_mask], sigma=sigma, observed=y[train_mask])

        self.model = model
        return model

    def fit(self, df, train_mask=None, draws=1000, tune=1500, chains=4, seed=42, **kw):
        self.build(df, train_mask)
        with self.model:
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                                   target_accept=0.95, progressbar=False, **kw)
        return self.idata

    # ---------------------------- Outputs ---------------------------- #
    def posterior_samples(self, var):
        """Stack chains x draws into one 'sample' axis."""
        return self.idata.posterior[var].stack(sample=("chain", "draw")).transpose("sample", ...).values

    def predicted_revenue(self):
        """Posterior samples of expected revenue per week, in USD. Shape (S, T)."""
        return self.posterior_samples("mu") * self.y_scale

    def channel_contributions(self):
        """USD revenue driven by each channel each week. Shape (S, T, C)."""
        return self.posterior_samples("channel_contrib") * self.y_scale

    def response_params(self):
        """Everything needed to rebuild response curves outside PyMC."""
        return dict(
            channels=np.array(self.channels),
            alpha=self.posterior_samples("alpha"),
            lam=self.posterior_samples("lam"),
            beta=self.posterior_samples("beta"),
            x_scale=self.x_scale,
            y_scale=np.array(self.y_scale),
            l_max=np.array(self.l_max),
        )


def weekly_response(spend, lam, beta, x_scale, y_scale):
    """
    Expected weekly incremental revenue if a channel spends `spend` EVERY week
    (steady state: adstock weights sum to 1, so adstocked spend == spend).
    Works with posterior arrays: lam, beta shape (S,) -> returns (S,).
    """
    return y_scale * beta * logistic_saturation(spend / x_scale, lam)


def fit_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    resid = actual - predicted
    r2 = 1 - (resid ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    mape = np.mean(np.abs(resid) / actual) * 100
    nrmse = np.sqrt(np.mean(resid ** 2)) / (actual.max() - actual.min())
    return {"r2": float(r2), "mape_pct": float(mape), "nrmse": float(nrmse)}

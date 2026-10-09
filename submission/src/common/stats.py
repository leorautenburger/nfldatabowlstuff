"""Shared modelling / uncertainty helpers.

Cross-fitting protocol (use for every expected-value model):
  * fit/tune only on TRAIN_WEEKS
  * train rows get out-of-fold predictions via GroupKFold(gameId)
  * test rows get predictions from a model refit on all train rows
  * report model quality on the TEST rows only
This gives every play an honest expectation and keeps weeks 7-8 untouched for evaluation.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .config import N_FOLDS


def cross_fit(model, df: pd.DataFrame, features, target, is_train_col="is_train",
              group_col="gameId", proba=False):
    """Return (predictions aligned to df.index, fitted_full_train_model)."""
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    tr = df[df[is_train_col]]
    te = df[~df[is_train_col]]
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fit_idx, oof_idx in gkf.split(tr, groups=tr[group_col]):
        m = clone(model).fit(tr.iloc[fit_idx][features], tr.iloc[fit_idx][target])
        X = tr.iloc[oof_idx][features]
        pred.loc[tr.index[oof_idx]] = m.predict_proba(X)[:, 1] if proba else m.predict(X)
    full = clone(model).fit(tr[features], tr[target])
    if len(te):
        pred.loc[te.index] = full.predict_proba(te[features])[:, 1] if proba else full.predict(te[features])
    return pred, full


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (vectorized)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def mean_ci(x, z=1.96):
    """(mean, lo, hi, n) normal-approx CI for a mean."""
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return pd.Series({"mean": x.mean() if n else np.nan, "lo": np.nan, "hi": np.nan, "n": n})
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    return pd.Series({"mean": m, "lo": m - z * se, "hi": m + z * se, "n": n})


def beta_shrink(k, n, prior_mean, prior_strength):
    """Empirical-Bayes shrinkage of a rate toward prior_mean with prior_strength pseudo-trials."""
    return (np.asarray(k) + prior_mean * prior_strength) / (np.asarray(n) + prior_strength)


def pooled_within_sd(df, player_col, value_col):
    """Pooled within-player SD (preferred input to eb_shrink_mean)."""
    d = df[[player_col, value_col]].dropna()
    dev = d[value_col] - d.groupby(player_col)[value_col].transform("mean")
    dof = len(d) - d[player_col].nunique()
    return float(np.sqrt((dev**2).sum() / max(dof, 1)))


def eb_shrink_mean(means, ns, sds, min_n_hyper=10):
    """Shrink per-player means toward the grand mean (normal-normal EB).

    sds: scalar pooled within-player SD (recommended, see pooled_within_sd) or per-player SDs.
    Hyper-parameters (grand mean, tau^2) are estimated by method of moments from players with
    n >= min_n_hyper only, then applied to everyone; tiny-n players otherwise dominate var(means).
    """
    means, ns = np.asarray(means, float), np.asarray(ns, float)
    sds = np.broadcast_to(np.asarray(sds, float), means.shape)
    ok = np.isfinite(means) & (ns > 0)
    se2 = np.where(ok, sds**2 / np.maximum(ns, 1), np.inf)
    h = ok & (ns >= min_n_hyper) & np.isfinite(se2)
    if h.sum() < 3:
        h = ok & np.isfinite(se2)
    grand = np.sum(means[h] * ns[h]) / np.sum(ns[h])
    tau2 = max(np.var(means[h]) - np.mean(se2[h]), 1e-6)
    w = np.where(ok, tau2 / (tau2 + se2), 0.0)
    return np.where(ok, w * np.nan_to_num(means) + (1 - w) * grand, grand)


def split_half_reliability(df, player_col, value_col, half_col="is_train", min_n=10):
    """Spearman correlation of player-level means between train weeks and test weeks."""
    g = df.groupby([player_col, half_col])[value_col].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= min_n]
    w = g.pivot(index=player_col, columns=half_col, values="mean")
    if True not in w.columns or False not in w.columns:
        return np.nan, 0
    w = w.dropna()
    if len(w) < 5:
        return np.nan, len(w)
    return stats.spearmanr(w[True], w[False]).correlation, len(w)


def odd_even_reliability(df, player_col, value_col, game_col="gameId", min_n=10):
    """Spearman of player means between alternating games (each player's 1st,3rd,.. vs 2nd,4th,..).

    Complements split_half_reliability: weeks 7-8 alone is a small half; odd/even games balance it.
    """
    d = df[[player_col, game_col, value_col]].dropna().copy()
    d["_half"] = d.groupby(player_col)[game_col].rank(method="dense") % 2 == 0
    return split_half_reliability(d, player_col, value_col, half_col="_half", min_n=min_n)

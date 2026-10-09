"""Edge Seal Sustainability (ESS) for tight ends.

For every blocker -> assigned-rusher pair (PFF pff_nflIdBlockedPlayer), seal time is the time from
the snap until the rusher first enters the QB threat zone:
    breach = dist(rusher, QB) <= R   OR   (rusher_x <= qb_x + 0.5  AND  |rusher_y - qb_y| <= L)
Seal time is right-censored at the end of the dropback (and at the TE's release for chip-release
rows). Per-TE observed Kaplan-Meier survival is compared with a context-only expected survival from
a discrete-time hazard model (0.5 s bins, HistGradientBoosting, cross-fit by game on weeks 1-6).

    value = observed KM S(2.5 s) - expected S(2.5 s)          (higher is better)

Run from the project root:
    .venv/bin/python -m src.metrics.edge_seal_sustainability
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.common import io, stats
from src.common.config import FPS, OUT, SEED, TEST_WEEKS, TRAIN_WEEKS

SLUG = "edge_seal_sustainability"
OUT_DIR = OUT / SLUG

# ---- breach definition -------------------------------------------------------------------------
R_BASE = 2.5             # yd, radial QB threat zone
R_SENS = [2.0, 3.0]      # mini-sensitivity
L_LAT = 3.0              # yd, lateral half-width of the QB depth-plane gate
DEPTH_PLANE = 0.5        # yd, rusher x <= qb_x + 0.5 counts as reaching the QB depth plane
# ---- survival / hazard --------------------------------------------------------------------------
BIN_W = 0.5              # s, discrete-time hazard bin width; bin k covers (0.5k, 0.5k+0.5]
MAX_BIN_FEAT = 9         # time_bin feature capped (>= 4.5 s pooled)
GRID_BINS = 7            # predict hazards for bins 0..6 for every pair (needed for S(3.5))
PARTIAL_MIN_FRAMES = 3   # a censored partial bin counts as at-risk if >= 3 of its 5 frames observed
EVAL_T = [1.5, 2.5, 3.5]
VALUE_T = 2.5
MIN_TOP = 20             # TE pass-block snaps required for the top list / rank correlations
MIN_REL = 10             # per-half minimum for reliability
EDGE_POS = {"LEO", "REO", "LOLB", "ROLB", "LE", "RE"}

CAT_FEATS = ["rusher_pos", "dropBackType", "offenseFormation"]
NUM_FEATS = ["time_bin", "rusher_pr", "r_dx", "r_dy", "b_dx", "b_dy", "b_xrel", "br_dx", "br_out",
             "br_dist", "pff_playAction", "n_rushers", "n_pass_blockers"]
FEATURES = NUM_FEATS + CAT_FEATS


# =================================================================================================
# data
# =================================================================================================
def load_pairs() -> tuple[pd.DataFrame, dict]:
    pp = io.player_plays()
    pl = io.plays()
    info = {}
    base_cols = ["gameId", "playId", "nflId", "displayName", "officialPosition", "alignment",
                 "x_rel_snap", "y_snap", "pff_nflIdBlockedPlayer", "pff_hitAllowed",
                 "pff_hurryAllowed", "pff_sackAllowed"]
    blk = pp.loc[pp.pff_role.eq("Pass Block") & pp.pff_nflIdBlockedPlayer.notna(), base_cols].copy()
    blk["kind"] = "pass_block"
    blk["release_t"] = np.nan
    te = io.te_plays()
    chip = te.loc[te.assignment.eq("chip_release") & te.pff_nflIdBlockedPlayer.notna(),
                  base_cols + ["release_t"]].copy()
    chip["kind"] = "chip_release"
    pairs = pd.concat([blk, chip], ignore_index=True)
    pairs["rusher_id"] = pairs.pff_nflIdBlockedPlayer.astype("int64")
    pairs = pairs.rename(columns={"x_rel_snap": "b_x", "y_snap": "b_y"})
    pairs["blocker_pressure_allowed"] = (
        pairs[["pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"]].fillna(0).sum(axis=1) > 0
    ).astype(int)
    info["candidate_pairs"] = {k: int(v) for k, v in pairs.kind.value_counts().items()}

    rush = pp.loc[pp.pff_role.eq("Pass Rush"),
                  ["gameId", "playId", "nflId", "pff_positionLinedUp", "x_rel_snap", "y_snap",
                   "pff_hit", "pff_hurry", "pff_sack", "week"]].copy()
    rush["rush_pressure"] = (rush[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).sum(axis=1) > 0).astype(int)
    r = rush.rename(columns={"nflId": "rusher_id", "pff_positionLinedUp": "rusher_pos",
                             "x_rel_snap": "r_x", "y_snap": "r_y"})
    pairs = pairs.merge(r[["gameId", "playId", "rusher_id", "rusher_pos", "r_x", "r_y"]],
                        on=["gameId", "playId", "rusher_id"], how="inner")
    info["pairs_after_requiring_blocked_player_is_pass_rusher"] = {
        k: int(v) for k, v in pairs.kind.value_counts().items()}

    qb = pp.loc[pp.pff_role.eq("Pass"), ["gameId", "playId", "nflId", "x_rel_snap", "y_snap"]]
    qb = qb.drop_duplicates(["gameId", "playId"]).rename(
        columns={"nflId": "qb_id", "x_rel_snap": "qb_x0", "y_snap": "qb_y0"})
    pairs = pairs.merge(qb, on=["gameId", "playId"], how="inner")

    pcols = ["gameId", "playId", "week", "is_train", "time_to_end", "dropBackType", "pff_playAction",
             "n_rushers", "n_pass_blockers", "offenseFormation", "pressure"]
    pairs = pairs.merge(pl[pcols], on=["gameId", "playId"], how="inner")
    pairs = pairs.loc[pairs.time_to_end.notna()].copy()

    # observation window
    pairs["window_end"] = pairs.time_to_end
    chip_rel = pairs.kind.eq("chip_release") & pairs.release_t.notna()
    pairs.loc[chip_rel, "window_end"] = np.minimum(pairs.loc[chip_rel, "release_t"],
                                                   pairs.loc[chip_rel, "time_to_end"])
    pairs["chip_no_release"] = pairs.kind.eq("chip_release") & pairs.release_t.isna()
    pairs["window_end"] = pairs.window_end.round(1)

    # snap geometry (context only: offsets from QB, blocker alignment relative to QB and rusher)
    pairs["r_dx"] = pairs.r_x - pairs.qb_x0
    pairs["r_dy"] = (pairs.r_y - pairs.qb_y0).abs()
    pairs["b_dx"] = pairs.b_x - pairs.qb_x0
    pairs["b_dy"] = (pairs.b_y - pairs.qb_y0).abs()
    pairs["b_xrel"] = pairs.b_x
    pairs["br_dx"] = pairs.r_x - pairs.b_x
    pairs["br_out"] = pairs.r_dy - pairs.b_dy          # + = rusher outside the blocker
    pairs["br_dist"] = np.hypot(pairs.r_x - pairs.b_x, pairs.r_y - pairs.b_y)
    pairs["pff_playAction"] = pairs.pff_playAction.astype(float)

    pairs = pairs.merge(rusher_quality(rush, pairs), on=["gameId", "playId", "rusher_id"], how="left")
    info["usable_pairs"] = {k: int(v) for k, v in pairs.kind.value_counts().items()}
    return pairs.reset_index(drop=True), info


def rusher_quality(rush: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """PFF pressure rate per rusher from train weeks only; leave-one-game-out for train rows;
    beta-shrunk toward the league train rate (prior strength by method of moments)."""
    tr = rush.loc[rush.week.isin(TRAIN_WEEKS)].rename(columns={"nflId": "rusher_id"})
    g = tr.groupby(["rusher_id", "gameId"]).rush_pressure.agg(k="sum", n="count").reset_index()
    tot = g.groupby("rusher_id")[["k", "n"]].sum()
    p0 = tot.k.sum() / tot.n.sum()
    big = tot.loc[tot.n >= 50]
    rates = big.k / big.n
    var_between = max(rates.var() - np.mean(p0 * (1 - p0) / big.n), 1e-6)
    strength = float(np.clip(p0 * (1 - p0) / var_between - 1, 10, 1000))
    rusher_quality.params = {"league_train_pressure_rate": float(p0), "beta_prior_strength": strength}

    keys = pairs[["gameId", "playId", "rusher_id", "is_train"]].drop_duplicates(
        ["gameId", "playId", "rusher_id"])
    keys = keys.merge(tot.rename(columns={"k": "K", "n": "N"}), left_on="rusher_id",
                      right_index=True, how="left")
    keys = keys.merge(g, on=["rusher_id", "gameId"], how="left")
    keys[["K", "N", "k", "n"]] = keys[["K", "N", "k", "n"]].fillna(0)
    own = keys.is_train.astype(bool)
    k = np.where(own, keys.K - keys.k, keys.K)
    n = np.where(own, keys.N - keys.n, keys.N)
    keys["rusher_pr"] = stats.beta_shrink(k, n, p0, strength)
    keys["rusher_n_train"] = n
    return keys[["gameId", "playId", "rusher_id", "rusher_pr", "rusher_n_train"]]


def breach_times(pairs: pd.DataFrame) -> pd.DataFrame:
    """First breach time per (play, rusher) for each radius, within snap -> end of dropback."""
    radii = sorted(set([R_BASE] + R_SENS))
    out = []
    for gid, gp in pairs.groupby("gameId"):
        rp = gp[["playId", "rusher_id", "qb_id", "time_to_end"]].drop_duplicates(["playId", "rusher_id"])
        ids = sorted(set(rp.rusher_id.tolist()) | set(rp.qb_id.tolist()))
        trk = io.tracking(columns=["playId", "nflId", "t", "x", "y"],
                          filters=[("gameId", "==", int(gid)), ("nflId", "in", ids)])
        trk["nflId"] = trk.nflId.astype("int64")
        trk["t"] = trk.t.astype(float).round(1)
        q = trk.merge(rp[["playId", "qb_id"]].drop_duplicates(), left_on=["playId", "nflId"],
                      right_on=["playId", "qb_id"])[["playId", "t", "x", "y"]]
        q = q.rename(columns={"x": "qx", "y": "qy"})
        r = trk.merge(rp, left_on=["playId", "nflId"], right_on=["playId", "rusher_id"])
        r = r.loc[r.t <= r.time_to_end + 1e-6, ["playId", "rusher_id", "t", "x", "y"]]
        f = r.merge(q, on=["playId", "t"])
        dist = np.hypot(f.x - f.qx, f.y - f.qy)
        plane = (f.x <= f.qx + DEPTH_PLANE) & ((f.y - f.qy).abs() <= L_LAT)
        res = f.groupby(["playId", "rusher_id"]).t.max().rename("track_t_max").to_frame()
        res["plane_breach_t"] = f.loc[plane].groupby(["playId", "rusher_id"]).t.min()
        keys = [f.playId, f.rusher_id]
        for R in radii:
            cond = ((dist <= R) | plane).to_numpy()
            # breach = zone ENTRY: first in-zone frame after the rusher has been outside the zone
            # (under-center snaps can start an interior rusher inside R; that is not a breach)
            t_out = pd.Series(np.where(~cond, f.t, np.inf), index=f.index).groupby(keys).transform("min")
            entry = cond & (f.t > t_out).to_numpy()
            res[f"breach_t_R{R}"] = f.loc[entry].groupby(["playId", "rusher_id"]).t.min()
            res[f"in_zone_at_snap_R{R}"] = (pd.Series(cond & (f.t == 0).to_numpy(), index=f.index)
                                            .groupby(keys).any())
        res = res.reset_index()
        res.insert(0, "gameId", gid)
        out.append(res)
    return pd.concat(out, ignore_index=True)


# =================================================================================================
# survival
# =================================================================================================
def survival_data(pairs: pd.DataFrame, R: float) -> tuple[np.ndarray, np.ndarray]:
    fb = pairs[f"breach_t_R{R}"].to_numpy(float)
    w = pairs.window_end.to_numpy(float)
    E = np.nan_to_num(fb, nan=np.inf) <= w + 1e-6
    T = np.round(np.where(E, fb, w), 1)
    return T, E


def km(T, E, taus):
    """Kaplan-Meier survival at taus with Greenwood variance and log-log 95% CI (numpy only)."""
    T, E = np.asarray(T, float), np.asarray(E, bool)
    res = {}
    if len(T) == 0:
        return {tau: dict(S=np.nan, se=np.nan, lo=np.nan, hi=np.nan, at_risk=0) for tau in taus}
    ut, d = np.unique(T[E], return_counts=True)
    Ts = np.sort(T)
    n = len(T) - np.searchsorted(Ts, ut - 1e-9, side="left")
    S = np.cumprod(1 - d / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(np.where(n > d, d / (n * (n - d)), np.inf))
    for tau in taus:
        at_risk = int((T >= tau - 1e-9).sum())
        if T.max() < tau - 1e-9 and (len(S) == 0 or S[-1] > 0):
            res[tau] = dict(S=np.nan, se=np.nan, lo=np.nan, hi=np.nan, at_risk=at_risk)
            continue
        j = np.searchsorted(ut, tau + 1e-9, side="right") - 1
        s, v = (1.0, 0.0) if j < 0 else (float(S[j]), float(gw[j]))
        se = s * np.sqrt(v) if np.isfinite(v) else np.nan
        if 0 < s < 1 and np.isfinite(v):
            sl = np.sqrt(v) / abs(np.log(s))
            lo, hi = s ** np.exp(1.96 * sl), s ** np.exp(-1.96 * sl)
        else:
            lo, hi = s, s
        res[tau] = dict(S=s, se=se, lo=lo, hi=hi, at_risk=at_risk)
    return res


def time_bin(t):
    return np.maximum(np.ceil(np.round(np.asarray(t, float) / BIN_W, 6)) - 1, 0).astype(int)


def person_period(pairs: pd.DataFrame, T, E) -> pd.DataFrame:
    """Discrete-time rows: observed (at-risk) bins get y in {0,1}; extra grid bins (up to bin 6)
    get y = -1 so they are predicted (out-of-fold) but never fitted."""
    kb = time_bin(T)
    frames_in_bin = np.round((T - BIN_W * kb) * FPS).astype(int) + (kb == 0)
    last_obs = np.where(E | (frames_in_bin >= PARTIAL_MIN_FRAMES), kb, kb - 1)
    nrows = np.maximum(last_obs, GRID_BINS - 1) + 1
    rep = np.repeat(np.arange(len(pairs)), nrows)
    k = np.arange(nrows.sum()) - np.repeat(np.cumsum(nrows) - nrows, nrows)
    obs = k <= last_obs[rep]
    ev = obs & E[rep] & (k == kb[rep])
    pp = pairs.iloc[rep][["gameId", "is_train"] + [c for c in FEATURES if c != "time_bin"]].reset_index(drop=True)
    pp["pair"] = rep
    pp["k"] = k
    pp["time_bin"] = np.minimum(k, MAX_BIN_FEAT)
    pp["y"] = np.where(obs, ev.astype(int), -1)
    return pp


class MaskedHazardHGB(ClassifierMixin, BaseEstimator):
    """HistGradientBoostingClassifier that fits only rows with y >= 0 (observed person-periods)
    but predicts every row, so stats.cross_fit yields OOF hazards for unobserved grid bins too."""

    def __init__(self, categorical_features=None, max_iter=300, learning_rate=0.06,
                 max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=SEED):
        self.categorical_features = categorical_features
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def fit(self, X, y):
        m = np.asarray(y) >= 0
        self.model_ = HistGradientBoostingClassifier(
            categorical_features=self.categorical_features, max_iter=self.max_iter,
            learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            random_state=self.random_state).fit(X[m], np.asarray(y)[m])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def fit_hazard(pairs: pd.DataFrame, R: float) -> dict:
    """Cross-fit the hazard model for radius R and return pair-level expectations + test metrics."""
    T, E = survival_data(pairs, R)
    pp = person_period(pairs, T, E)
    model = MaskedHazardHGB(categorical_features=CAT_FEATS)
    pp["h"], full = stats.cross_fit(model, pp, FEATURES, "y", proba=True)

    # pair-level expected survival and expected breaches in the observed window
    grid = pp.loc[pp.k < GRID_BINS].pivot(index="pair", columns="k", values="h").sort_index()
    cs = np.cumprod(1 - grid.to_numpy(), axis=1)
    exp = pd.DataFrame(index=grid.index)
    for tau in EVAL_T:
        exp[f"exp_S_{tau}"] = cs[:, int(round(tau / BIN_W)) - 1]
    exp["exp_breaches"] = pp.loc[pp.y >= 0].groupby("pair").h.sum()
    exp = exp.reindex(np.arange(len(pairs)))
    exp["exp_breaches"] = exp.exp_breaches.fillna(0.0)
    exp["T"], exp["E"] = T, E.astype(int)

    # test-set model quality on observed person-periods vs time-bin-only baseline
    obs = pp.loc[pp.y >= 0]
    trn, tst = obs.loc[obs.is_train], obs.loc[~obs.is_train]
    base_h = trn.groupby("time_bin").y.mean()
    b = tst.time_bin.map(base_h).fillna(trn.y.mean()).clip(1e-6, 1 - 1e-6).to_numpy()
    p = tst.h.clip(1e-6, 1 - 1e-6).to_numpy()
    y = tst.y.to_numpy()
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin")
             .agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index())
    metrics = {
        "unit": "test person-period rows (pair x 0.5 s bin while at risk), weeks 7-8",
        "n_rows": int(len(tst)), "n_events": int(y.sum()),
        "model": {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
                  "brier": float(brier_score_loss(y, p))},
        "baseline_time_bin_only": {"auc": float(roc_auc_score(y, b)), "log_loss": float(log_loss(y, b)),
                                   "brier": float(brier_score_loss(y, b))},
        "calibration_5bin": calib.round(4).to_dict(orient="records"),
    }
    # league expected vs observed survival on test pairs
    te_mask = ~pairs.is_train.to_numpy()
    base_S = {tau: float(np.prod(1 - base_h.reindex(range(int(round(tau / BIN_W)))).to_numpy()))
              for tau in EVAL_T}
    kmt = km(T[te_mask], E[te_mask], EVAL_T)
    metrics["test_survival_expected_vs_observed"] = {
        str(tau): {"observed_km": kmt[tau]["S"], "km_lo": kmt[tau]["lo"], "km_hi": kmt[tau]["hi"],
                   "expected_model_mean": float(exp.loc[te_mask, f"exp_S_{tau}"].mean()),
                   "expected_baseline": base_S[tau]} for tau in EVAL_T}
    metrics["test_expected_vs_observed_breaches_per_pair"] = {
        "expected": float(exp.loc[te_mask, "exp_breaches"].mean()), "observed": float(E[te_mask].mean())}
    return {"exp": exp, "metrics": metrics, "n_person_period_rows": int((pp.y >= 0).sum()),
            "n_iter_full": int(full.model_.n_iter_)}


# =================================================================================================
# player tables
# =================================================================================================
def te_value_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nid, name), g in d.groupby(["nflId", "displayName"]):
        k = km(g["T"], g["E"].astype(bool), EVAL_T)
        row = {"nflId": nid, "displayName": name, "n": len(g), "n_breaches": int(g.E.sum())}
        for tau in EVAL_T:
            row[f"km_S_{tau}"] = k[tau]["S"]
            row[f"km_lo_{tau}"] = k[tau]["lo"]
            row[f"km_hi_{tau}"] = k[tau]["hi"]
            row[f"exp_S_{tau}"] = g[f"exp_S_{tau}"].mean()
        row["km_se_2.5"] = k[VALUE_T]["se"]
        row["sustained_rate"] = 1 - g.E.mean()
        row["exp_minus_obs_breaches_per_snap"] = (g.exp_breaches - g.E).mean()
        row["inline_share"] = g.alignment.eq("inline").mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t["value"] = t[f"km_S_{VALUE_T}"] - t[f"exp_S_{VALUE_T}"]
    return t


def km_value_reliability(d: pd.DataFrame, half_col: str) -> tuple[float, int]:
    vals = {}
    for h in (True, False):
        sub = d.loc[d[half_col] == h]
        t = te_value_table(sub)
        vals[h] = t.loc[t.n >= MIN_REL].set_index("nflId").value
    j = pd.concat(vals, axis=1).dropna()
    if len(j) < 5:
        return np.nan, len(j)
    return float(sps.spearmanr(j[True], j[False]).correlation), len(j)


def binary_validation(flag: pd.Series, outcome: pd.Series) -> dict:
    flag, outcome = flag.astype(bool), outcome.astype(int)
    a = outcome[flag]
    b = outcome[~flag]
    phi = float(np.corrcoef(flag.astype(int), outcome)[0, 1]) if outcome.std() > 0 else np.nan
    return {"n": int(len(flag)), "share_early_breach": float(flag.mean()),
            "outcome_rate_if_early_breach": float(a.mean()) if len(a) else None,
            "outcome_rate_if_no_early_breach": float(b.mean()) if len(b) else None,
            "relative_risk": float(a.mean() / b.mean()) if len(a) and b.mean() > 0 else None,
            "phi": phi,
            "auc_of_early_breach_flag": float(roc_auc_score(outcome, flag)) if outcome.nunique() > 1 else None,
            "share_of_outcomes_with_early_breach": float(flag[outcome == 1].mean()) if outcome.sum() else None}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# =================================================================================================
def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs, info = load_pairs()
    pairs["rusher_pos_label"] = pairs.rusher_pos.fillna("UNK").astype(str)
    for c in CAT_FEATS:
        codes = pd.Categorical(pairs[c].fillna("UNK").astype(str)).codes.astype(float)
        pairs[c] = codes
    print(f"[{time.time()-t0:5.1f}s] pairs: {info['usable_pairs']}")

    bt = breach_times(pairs)
    pairs = pairs.merge(bt, on=["gameId", "playId", "rusher_id"], how="inner")
    print(f"[{time.time()-t0:5.1f}s] breach times computed for {len(pairs)} pairs")
    early = {f"R{R}": {"in_zone_at_snap": int(pairs[f"in_zone_at_snap_R{R}"].sum()),
                       "entry_breach_t<=0.3s": int((pairs[f"breach_t_R{R}"] <= 0.3).sum())}
             for R in [R_BASE] + R_SENS}

    res = {}
    for R in [R_BASE] + R_SENS:
        res[R] = fit_hazard(pairs, R)
        print(f"[{time.time()-t0:5.1f}s] hazard model R={R}: test AUC "
              f"{res[R]['metrics']['model']['auc']:.3f} vs base {res[R]['metrics']['baseline_time_bin_only']['auc']:.3f}")

    base = pd.concat([pairs, res[R_BASE]["exp"].reset_index(drop=True)], axis=1)
    is_te = base.officialPosition.eq("TE")
    te_pb = base.loc[is_te & base.kind.eq("pass_block")].copy()
    te_ch = base.loc[base.kind.eq("chip_release")].copy()

    # ---- TE table (pass-block snaps) --------------------------------------------------------
    players = te_value_table(te_pb)
    league_te = km(te_pb["T"], te_pb["E"].astype(bool), [VALUE_T])[VALUE_T]["S"]
    sd = np.sqrt(league_te * (1 - league_te))
    players["value_shrunk"] = stats.eb_shrink_mean(players.value, players.n, sd)
    players["lo"] = players[f"km_lo_{VALUE_T}"] - players[f"exp_S_{VALUE_T}"]
    players["hi"] = players[f"km_hi_{VALUE_T}"] - players[f"exp_S_{VALUE_T}"]
    s_lo, s_hi = stats.wilson_ci(te_pb.groupby("nflId").E.apply(lambda e: (1 - e).sum()).reindex(players.nflId).to_numpy(),
                                 players.n.to_numpy())
    players["sustained_lo"], players["sustained_hi"] = s_lo, s_hi
    players["higher_is_better"] = True
    players["qualifies_top"] = players.n >= MIN_TOP

    # ---- chip-release rows, reported separately --------------------------------------------
    ch = te_ch.groupby(["nflId", "displayName"]).agg(
        chip_n=("E", "size"), chip_breaches=("E", "sum"), chip_window_median=("T", "median"),
        chip_exp_breaches=("exp_breaches", "sum")).reset_index()
    ch["chip_breach_rate"] = ch.chip_breaches / ch.chip_n
    ch["chip_exp_minus_obs_breaches_per_snap"] = (ch.chip_exp_breaches - ch.chip_breaches) / ch.chip_n
    players = players.merge(ch.drop(columns="chip_exp_breaches"), on=["nflId", "displayName"], how="outer")
    team = (io.player_plays()[["gameId", "playId", "nflId"]]
            .merge(io.plays()[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"])
            .loc[lambda d: d.nflId.isin(players.nflId)]
            .groupby("nflId").possessionTeam.agg(lambda s: s.mode().iloc[0]).rename("team"))
    players = players.merge(team, left_on="nflId", right_index=True, how="left")
    players["n"] = players.n.fillna(0).astype(int)
    players["higher_is_better"] = True
    players["qualifies_top"] = players.n >= MIN_TOP
    players = players.sort_values(["qualifies_top", "value_shrunk"], ascending=[False, False],
                                  na_position="last")
    first = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "higher_is_better",
             "qualifies_top", "km_S_1.5", "km_S_2.5", "km_S_3.5", "exp_S_1.5", "exp_S_2.5", "exp_S_3.5",
             "sustained_rate", "sustained_lo", "sustained_hi", "exp_minus_obs_breaches_per_snap", "n_breaches"]
    players = players[first + [c for c in players.columns if c not in first]]
    players.to_csv(OUT_DIR / "players.csv", index=False, float_format="%.4f")

    # ---- plays.csv ---------------------------------------------------------------------------
    te_rows = base.loc[is_te | base.kind.eq("chip_release")].copy()
    te_rows["seal_time"] = te_rows["T"]
    te_rows["breached"] = te_rows["E"]
    te_rows["exp_minus_obs_breaches"] = te_rows.exp_breaches - te_rows.E
    pcols = ["gameId", "playId", "nflId", "displayName", "week", "is_train", "kind", "alignment",
             "rusher_id", "rusher_pos_label", "rusher_pr", "window_end", "chip_no_release", "seal_time",
             "breached", "exp_S_1.5", "exp_S_2.5", "exp_S_3.5", "exp_breaches", "exp_minus_obs_breaches",
             "plane_breach_t", "breach_t_R2.0", "breach_t_R2.5", "breach_t_R3.0",
             "blocker_pressure_allowed", "pressure"]
    te_rows.sort_values(["gameId", "playId", "nflId"])[pcols].to_csv(
        OUT_DIR / "plays.csv", index=False, float_format="%.4f")

    # ---- validation ---------------------------------------------------------------------------
    pb = base.loc[base.kind.eq("pass_block")]
    early_flag = (pb.E == 1) & (pb["T"] <= VALUE_T + 1e-9)
    tpb = pb.officialPosition.eq("TE")
    validation = {
        "early_breach_definition": f"breach at or before {VALUE_T} s (R={R_BASE})",
        "vs_pff_pressure_allowed_by_blocker": {
            "all_blockers": binary_validation(early_flag, pb.blocker_pressure_allowed),
            "te_blockers": binary_validation(early_flag[tpb], pb.blocker_pressure_allowed[tpb])},
        "vs_play_pressure": {
            "all_blockers": binary_validation(early_flag, pb.pressure),
            "te_blockers": binary_validation(early_flag[tpb], pb.pressure[tpb])},
    }
    edge = pb.loc[pb.rusher_pos_label.isin(EDGE_POS)]
    grp = {}
    for lab, pos in [("TE", "TE"), ("OT", "T")]:
        g = edge.loc[edge.officialPosition.eq(pos)]
        k = km(g["T"], g["E"].astype(bool), EVAL_T)
        grp[lab] = {"n": int(len(g)), **{f"km_S_{tau}": k[tau]["S"] for tau in EVAL_T},
                    **{f"km_ci_{tau}": [k[tau]["lo"], k[tau]["hi"]] for tau in EVAL_T},
                    "expected_S_2.5": float(g["exp_S_2.5"].mean()),
                    "obs_minus_exp_S_2.5": k[VALUE_T]["S"] - float(g["exp_S_2.5"].mean())}
    validation["te_vs_ot_vs_edge_rushers"] = {"edge_alignments": sorted(EDGE_POS), **grp}

    # ---- reliability ----------------------------------------------------------------------
    te_pb["resid"] = te_pb.exp_breaches - te_pb.E
    rel = {}
    r, n = stats.split_half_reliability(te_pb, "nflId", "resid", min_n=MIN_REL)
    rel["split_half_weeks1-6_vs_7-8_exp_minus_obs_breaches"] = {"spearman": r, "n_players": n}
    r, n = stats.odd_even_reliability(te_pb, "nflId", "resid", min_n=MIN_REL)
    rel["odd_even_games_exp_minus_obs_breaches"] = {"spearman": r, "n_players": n}
    r, n = km_value_reliability(te_pb, "is_train")
    rel["split_half_weeks1-6_vs_7-8_value"] = {"spearman": r, "n_players": n}
    te_pb["_odd"] = te_pb.groupby("nflId").gameId.rank(method="dense") % 2 == 0
    r, n = km_value_reliability(te_pb, "_odd")
    rel["odd_even_games_value"] = {"spearman": r, "n_players": n}
    rel["min_snaps_per_half"] = MIN_REL

    # ---- sensitivity -------------------------------------------------------------------------
    sens = {}
    base_rank = players.loc[players.qualifies_top].set_index("nflId")
    for R in [R_BASE] + R_SENS:
        d = pd.concat([pairs, res[R]["exp"].reset_index(drop=True)], axis=1)
        dte = d.loc[d.officialPosition.eq("TE") & d.kind.eq("pass_block")]
        kl = km(d.loc[d.kind.eq("pass_block"), "T"], d.loc[d.kind.eq("pass_block"), "E"].astype(bool), EVAL_T)
        kt = km(dte["T"], dte["E"].astype(bool), EVAL_T)
        entry = {"league_all_blockers_km": {str(t): kl[t]["S"] for t in EVAL_T},
                 "league_te_km": {str(t): kt[t]["S"] for t in EVAL_T},
                 "test_auc": res[R]["metrics"]["model"]["auc"],
                 "test_auc_baseline": res[R]["metrics"]["baseline_time_bin_only"]["auc"]}
        if R != R_BASE:
            t = te_value_table(dte).set_index("nflId")
            common = base_rank.index.intersection(t.index[t.n >= MIN_TOP])
            entry["spearman_te_value_vs_base"] = float(sps.spearmanr(base_rank.loc[common, "value"],
                                                                     t.loc[common, "value"]).correlation)
            entry["spearman_te_km_S2.5_vs_base"] = float(sps.spearmanr(base_rank.loc[common, "km_S_2.5"],
                                                                       t.loc[common, "km_S_2.5"]).correlation)
            entry["n_te"] = int(len(common))
        sens[f"R={R}"] = entry

    # ---- summary ---------------------------------------------------------------------------------
    top5 = players.loc[players.qualifies_top].head(5)[
        ["displayName", "team", "n", "value", "value_shrunk", "km_S_2.5", "exp_S_2.5"]]
    summary = {
        "metric": "Edge Seal Sustainability (ESS)", "slug": SLUG,
        "definition": (
            "Blocker -> assigned rusher (PFF pff_nflIdBlockedPlayer) seal time = time from snap to the first "
            f"frame where the rusher is within R={R_BASE} yd of the QB, or rusher x <= qb_x + {DEPTH_PLANE} "
            f"while |rusher y - qb y| <= {L_LAT} yd, counted as zone entry (the rusher must first be "
            "observed outside the zone; under-center snaps can place an interior rusher inside R at t=0). "
            "Right-censored at end of dropback (throw/sack/other stop); "
            "chip-release rows are censored at the TE's release. value = TE observed Kaplan-Meier S(2.5 s) "
            "on pass-block snaps minus mean context-expected S(2.5 s) from a discrete-time hazard model."),
        "parameters": {"R_yards": R_BASE, "L_yards": L_LAT, "depth_plane_yards": DEPTH_PLANE,
                       "bin_width_s": BIN_W, "time_bin_feature_cap": MAX_BIN_FEAT,
                       "partial_censored_bin_min_frames": PARTIAL_MIN_FRAMES, "eval_times_s": EVAL_T,
                       "value_time_s": VALUE_T, "min_te_snaps_top": MIN_TOP,
                       "rusher_quality": rusher_quality.params, "seed": SEED,
                       "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS},
        "population": {**info, "pairs_with_rusher_and_qb_tracking": int(len(pairs)),
                       "te_pass_block_pairs": int(len(te_pb)), "te_chip_release_pairs": int(len(te_ch)),
                       "chip_rows_never_released_censored_at_dropback_end": int(te_ch.chip_no_release.sum()),
                       "snap_zone_diagnostics": early},
        "expected_model": {
            "type": "discrete-time hazard, 0.5 s bins, HistGradientBoostingClassifier wrapped to fit only "
                    "at-risk person-periods; stats.cross_fit(proba=True), GroupKFold by game on weeks 1-6, "
                    "full-train refit for weeks 7-8. Fit on all pairs (OL/TE/RB pass blocks + TE chips).",
            "features": FEATURES, "categorical": CAT_FEATS,
            "n_person_period_rows": res[R_BASE]["n_person_period_rows"],
            "full_model_iterations": res[R_BASE]["n_iter_full"],
            "test_metrics": res[R_BASE]["metrics"]},
        "validation": validation,
        "reliability": rel,
        "sensitivity_radius": sens,
        "top5_min20": top5.to_dict(orient="records"),
        "caveats": [
            "dropBackType includes SCRAMBLE labels that are partly a consequence of pressure; it is used as "
            "requested but makes the expectation slightly outcome-aware.",
            "Seal breach is a geometric proxy; a rusher can reach the zone after the blocker passes him off "
            "or on a QB drift into him. Assignment comes from a single PFF blocked-player label.",
            "Tracking ends ~0.5 s after the throw; S(3.5) is estimated from few at-risk pairs on quick games.",
            "Chip-release rows are short windows (median release ~1.8 s); they are reported separately and do "
            "not enter value.",
            "Expected survival is the mean of per-pair model curves; observed is KM. Both assume censoring "
            "independent of seal quality given context.",
            "lo/hi = KM log-log Greenwood CI minus expected (expectation uncertainty ignored). value_shrunk uses "
            "normal EB with a binomial-style SD at league TE S(2.5).",
        ],
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(f"[{time.time()-t0:5.1f}s] wrote {len(players)} TEs, {len(te_rows)} TE pair rows to {OUT_DIR}")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()

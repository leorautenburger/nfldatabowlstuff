"""Middle-of-Field Access (MOFA) for tight ends.

At fixed checkpoints t in {1.5, 2.5, 3.5} s after the snap, a route runner offers a viable
middle-of-field (MOF) window if, at the receiver_frames frame nearest t:
    MOF_Y[0] <= y <= MOF_Y[1]           (between the numbers, y in [12, 41.3])
    MOF_X_REL[0] <= x_rel <= MOF_X_REL[1] (0-20 yd past the LOS)
    sep >= VIABLE_MIN_SEP                (nearest defender)
    lane_clear >= VIABLE_MIN_LANE        (QB -> receiver segment)

Live dropback: checkpoint t is "live" iff t <= t_end (end of dropback = throw frame on throw plays,
sack/scramble/other stop frame otherwise; routes.t_end).

Primary:       P(viable at t | TE on route or chip_release, dropback live at t)
Unconditional: P(viable at t | TE on route or chip_release)  (not live -> not viable)
Player value:  equal-weight mean of the three per-checkpoint rates (value = raw, value_shrunk =
               mean of the three beta-shrunk rates; prior = TE league live rate per checkpoint).
Context model: P(viable | t, alignment, snap position, receiver slot, play context) fit on ALL route
               runners' live checkpoints (position is NOT a feature); MOFA-OE = observed - expected.

Run: .venv/bin/python -m src.metrics.middle_of_field_access
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

from src.common import io, stats  # noqa: E402
from src.common.config import (OUT, SEED, TEST_WEEKS, TRAIN_WEEKS, VIABLE_MIN_LANE,  # noqa: E402
                               VIABLE_MIN_SEP)

warnings.filterwarnings("ignore", category=FutureWarning)

SLUG = "middle_of_field_access"
OUT_DIR = OUT / SLUG
KEY = ["gameId", "playId", "nflId"]
EPS = 1e-6

CHECKPOINTS = [1.5, 2.5, 3.5]
CP_TAG = {1.5: "t15", 2.5: "t25", 3.5: "t35"}
FRAME_TOL = 0.05 + EPS                 # nearest frame must be within half a frame of the checkpoint
MOF_Y = (12.0, 41.3)                   # between the numbers (default)
MOF_Y_HASH_WIDE = (18.0, 35.3)         # "between the hashes", widened (sensitivity)
MOF_X_REL = (0.0, 20.0)                # yards past LOS
ROUTE_ASSIGN = ["route", "chip_release"]
MIN_N_RANK = 40                        # min TE route snaps for top-5 / sensitivity Spearman
MIN_N_PRIOR = 20                       # min live snaps at a checkpoint for prior-strength estimation
N_BOOT = 2000

SENS_BANDS = {"numbers_12_41.3": MOF_Y, "hashes_wide_18_35.3": MOF_Y_HASH_WIDE}
SENS_SEP = [2.0, 2.5, 3.0]

CAT_FEATS = ["alignment", "offenseFormation", "personnelO", "pff_passCoverage", "pff_passCoverageType",
             "dropBackType"]
NUM_FEATS = ["t", "x_rel_snap", "abs_lat", "receiver_num", "n_rec_side", "pff_playAction", "down",
             "yardsToGo", "red_zone"]
FEATURES = NUM_FEATS[:1] + CAT_FEATS[:1] + NUM_FEATS[1:5] + CAT_FEATS[1:] + NUM_FEATS[5:]
HGB_GRID = [
    dict(learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=50, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=400, max_depth=3, min_samples_leaf=100, l2_regularization=1.0),
    dict(learning_rate=0.03, max_iter=500, max_depth=6, min_samples_leaf=80, l2_regularization=3.0),
]


# --------------------------------------------------------------------------- data
def load():
    p = io.plays()
    ctx = p[["gameId", "playId", "possessionTeam", "week", "offenseFormation", "personnelO",
             "pff_passCoverage", "pff_passCoverageType", "pff_playAction", "dropBackType", "down",
             "yardsToGo", "red_zone"]]
    r = io.routes()[KEY + ["officialPosition", "alignment", "x_rel_snap", "y_snap", "abs_lat", "receiver_num",
                           "n_rec_side", "release_t", "t_end", "end_event", "time_to_throw", "is_target",
                           "is_catch", "is_train"]]
    r = r.merge(ctx, on=["gameId", "playId"], how="left")
    te = io.te_plays()
    te_r = te[te.assignment.isin(ROUTE_ASSIGN)]
    n_te_route_all = len(te_r)
    r = r.merge(te_r[KEY + ["assignment", "displayName"]], on=KEY, how="left")
    r["is_te"] = r.assignment.notna()          # TE on route / chip_release with tracking
    r = r.reset_index(drop=True)

    rf = io.receiver_frames(columns=KEY + ["t", "y", "x_rel", "sep", "lane_clear"],
                            filters=[("t", ">=", min(CHECKPOINTS) - 0.06), ("t", "<=", max(CHECKPOINTS) + 0.06)])
    parts = []
    for c in CHECKPOINTS:
        s = rf[(rf.t - c).abs() <= FRAME_TOL].copy()
        s["d"] = (s.t - c).abs()
        s = s.sort_values("d").drop_duplicates(KEY).drop(columns="d")
        s["cp"] = c
        parts.append(s.rename(columns={"t": "t_frame"}))
    fr = pd.concat(parts, ignore_index=True)

    # long table: one row per route runner x checkpoint
    cp = r.assign(_k=1).merge(pd.DataFrame({"cp": CHECKPOINTS, "_k": 1}), on="_k").drop(columns="_k")
    cp = cp.merge(fr, on=KEY + ["cp"], how="left")
    cp["t"] = cp.cp
    cp["live"] = cp.cp <= cp.t_end + EPS
    cp["frame_missing"] = cp.live & cp.t_frame.isna()
    cp["los_side_ok"] = cp.x_rel.between(*MOF_X_REL)
    return p, r, cp, n_te_route_all


def viable_mask(cp, band=MOF_Y, sep=VIABLE_MIN_SEP, lane=VIABLE_MIN_LANE):
    m = (cp.live & cp.y.between(*band) & cp.x_rel.between(*MOF_X_REL)
         & (cp.sep >= sep) & (cp.lane_clear >= lane))
    return m.fillna(False).values.astype(bool)


# --------------------------------------------------------------------------- shrinkage helpers
def beta_prior_strength(k, n, min_n=MIN_N_PRIOR):
    """Method-of-moments beta-binomial prior strength from players with n >= min_n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    sel = n >= min_n
    k, n = k[sel], n[sel]
    m = k.sum() / n.sum()
    r = k / n
    tau2 = np.var(r, ddof=1) - np.mean(m * (1 - m) / n)
    if tau2 <= 1e-6:
        return 1000.0, float(tau2), int(sel.sum())
    return float(min(max(m * (1 - m) / tau2 - 1, 5.0), 1000.0)), float(tau2), int(sel.sum())


def priors_for(te_cp, flag):
    out = {}
    for c in CHECKPOINTS:
        s = te_cp[(te_cp.cp == c) & te_cp.live]
        g = s.groupby("nflId")[flag].agg(["sum", "count"])
        strength, tau2, npl = beta_prior_strength(g["sum"], g["count"])
        out[c] = {"mean": float(s[flag].mean()), "strength": strength, "between_player_var": tau2,
                  "n_players_used": npl, "n_live": int(len(s))}
    return out


def eb_toward(means, ns, grand, s2_within, min_n_mask):
    """Normal-normal EB: shrink player mean residuals toward `grand` (TE pooled mean residual).
    se2 = pooled within-player residual variance / n; tau2 by method of moments over players in
    min_n_mask. Undefined means (n=0) -> grand."""
    means, ns = np.asarray(means, float), np.asarray(ns, float)
    ok = np.isfinite(means) & (ns > 0)
    se2 = np.where(ok, s2_within / np.maximum(ns, 1), np.inf)
    sel = ok & min_n_mask
    tau2 = max(np.var(means[sel], ddof=1) - np.mean(se2[sel]), 1e-6) if sel.sum() > 2 else 1e-6
    w = np.where(ok, tau2 / (tau2 + se2), 0.0)
    return np.where(ok, w * np.nan_to_num(means) + (1 - w) * grand, grand), float(tau2)


# --------------------------------------------------------------------------- player table
def to_wide(te_cp, cols):
    """One row per TE route snap (index KEY), columns (col, checkpoint); bools become float."""
    d = te_cp[KEY + ["cp"] + cols].copy()
    for c in cols:
        if d[c].dtype == bool:
            d[c] = d[c].astype(float)
    return d.set_index(KEY + ["cp"])[cols].unstack("cp")


def player_rates(te_cp, flag, prior):
    """Per-player per-checkpoint k, n_live, raw rate, Wilson CI, beta-shrunk; plus summaries."""
    g = te_cp.groupby(["nflId", "cp"]).agg(k=(flag, "sum"), n=("live", "sum")).unstack("cp")
    out = pd.DataFrame(index=g.index)
    n_route = te_cp[te_cp.cp == CHECKPOINTS[0]].groupby("nflId").size()
    out["n"] = n_route.reindex(out.index).astype(int)
    raw, shr = [], []
    for c in CHECKPOINTS:
        tg = CP_TAG[c]
        k, n = g[("k", c)].fillna(0).astype(int), g[("n", c)].fillna(0).astype(int)
        out[f"n_live_{tg}"], out[f"k_{tg}"] = n, k
        out[f"rate_{tg}"] = k / n.replace(0, np.nan)
        out[f"rate_{tg}_lo"], out[f"rate_{tg}_hi"] = stats.wilson_ci(k, n)
        out[f"rate_{tg}_shrunk"] = stats.beta_shrink(k, n, prior[c]["mean"], prior[c]["strength"])
        out[f"uncond_{tg}"] = k / out.n
        raw.append(out[f"rate_{tg}"])
        shr.append(out[f"rate_{tg}_shrunk"])
    out["value"] = pd.concat(raw, axis=1).mean(axis=1, skipna=False)
    out["value_shrunk"] = pd.concat(shr, axis=1).mean(axis=1)
    out["uncond_mean"] = out[[f"uncond_{CP_TAG[c]}" for c in CHECKPOINTS]].mean(axis=1)
    return out


def bootstrap_player(te_cp, prior, rng):
    """Play-level bootstrap within player (keeps the 3 checkpoints of a play together)."""
    w = to_wide(te_cp, ["live", "viable", "resid"])
    res = {}
    m = np.array([prior[c]["mean"] for c in CHECKPOINTS])
    s = np.array([prior[c]["strength"] for c in CHECKPOINTS])
    for pid, sub in w.groupby(level="nflId"):
        L = sub["live"][CHECKPOINTS].to_numpy(float)
        V = sub["viable"][CHECKPOINTS].to_numpy(float) * L
        R = np.nan_to_num(sub["resid"][CHECKPOINTS].to_numpy(float)) * L
        n = len(sub)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
        K, N, RS = W @ V, W @ L, W @ R
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (K / np.where(N > 0, N, np.nan)).mean(axis=1)
            shr = ((K + m * s) / (N + s)).mean(axis=1)
            oe_c = RS / np.where(N > 0, N, np.nan)
        oe = oe_c.mean(axis=1)
        res[pid] = {
            "lo": np.nanpercentile(raw, 2.5) if np.isfinite(raw).mean() > 0.9 else np.nan,
            "hi": np.nanpercentile(raw, 97.5) if np.isfinite(raw).mean() > 0.9 else np.nan,
            "value_shrunk_lo": np.percentile(shr, 2.5), "value_shrunk_hi": np.percentile(shr, 97.5),
            "mofa_oe_lo": np.nanpercentile(oe, 2.5) if np.isfinite(oe).mean() > 0.9 else np.nan,
            "mofa_oe_hi": np.nanpercentile(oe, 97.5) if np.isfinite(oe).mean() > 0.9 else np.nan,
        }
    return pd.DataFrame.from_dict(res, orient="index")


def player_table(te_cp, r_te, prior, rng):
    out = player_rates(te_cp, "viable", prior)
    meta = r_te.groupby("nflId").agg(displayName=("displayName", lambda s: s.mode().iat[0]),
                                     team=("possessionTeam", lambda s: s.mode().iat[0]),
                                     n_chip_release=("assignment", lambda s: int((s == "chip_release").sum())),
                                     share_inline=("alignment", lambda s: (s == "inline").mean()),
                                     share_wing=("alignment", lambda s: (s == "wing").mean()),
                                     share_detached=("alignment", lambda s: (s == "detached").mean()),
                                     share_backfield=("alignment", lambda s: (s == "backfield").mean()))
    out = meta.join(out)
    bt = bootstrap_player(te_cp, prior, rng)
    out = out.join(bt)

    # context-adjusted: per-checkpoint mean expected and observed-minus-expected (live rows)
    live = te_cp[te_cp.live]
    ge = live.groupby(["nflId", "cp"]).agg(expected=("expected", "mean"), oe=("resid", "mean")).unstack("cp")
    big = (out.n >= MIN_N_RANK).values
    oe_shr, eb_info = [], {}
    for c in CHECKPOINTS:
        tg = CP_TAG[c]
        out[f"expected_{tg}"] = ge[("expected", c)].reindex(out.index)
        out[f"oe_{tg}"] = ge[("oe", c)].reindex(out.index)
        lc = live[live.cp == c]
        grand = float(lc.resid.mean())
        s2_within = float((lc.resid - lc.groupby("nflId").resid.transform("mean")).pow(2).sum()
                          / max(len(lc) - lc.nflId.nunique(), 1))
        shr, tau2 = eb_toward(out[f"oe_{tg}"], out[f"n_live_{tg}"], grand, s2_within, big)
        out[f"oe_{tg}_shrunk"] = shr
        oe_shr.append(out[f"oe_{tg}_shrunk"])
        eb_info[c] = {"grand_mean_te_resid": round(grand, 5), "within_player_resid_var": round(s2_within, 5),
                      "tau2": tau2, "implied_strength_n_for_half_weight": round(s2_within / tau2, 1)}
    out["mofa_expected"] = out[[f"expected_{CP_TAG[c]}" for c in CHECKPOINTS]].mean(axis=1, skipna=False)
    out["mofa_oe"] = out[[f"oe_{CP_TAG[c]}" for c in CHECKPOINTS]].mean(axis=1, skipna=False)
    out["mofa_oe_shrunk"] = pd.concat(oe_shr, axis=1).mean(axis=1)

    # alignment split (pooled live checkpoints) - reporting principle
    for a in ["inline", "wing", "detached", "backfield"]:
        s = live[live.alignment == a].groupby("nflId").viable.agg(["mean", "count"]).reindex(out.index)
        out[f"n_live_cp_{a}"] = s["count"].fillna(0).astype(int)
        out[f"pooled_rate_{a}"] = s["mean"]
    out["higher_is_better"] = True
    out = out.reset_index().rename(columns={"index": "nflId"}).sort_values("value_shrunk", ascending=False)
    first = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi",
             "value_shrunk_lo", "value_shrunk_hi", "higher_is_better"]
    return out[first + [c for c in out.columns if c not in first]], eb_info


# --------------------------------------------------------------------------- model
def prep_features(cp):
    X = cp.copy()
    for c in CAT_FEATS:
        X[c] = X[c].fillna("NA").astype(str).astype("category")
    for c in NUM_FEATS:
        X[c] = X[c].astype(float)
    X["viable_i"] = X.viable.astype(int)
    return X


def _clf_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": round(float(roc_auc_score(y, p)), 5) if len(np.unique(y)) > 1 else None,
            "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5),
            "brier": round(float(brier_score_loss(y, p)), 5)}


def fit_expected(cp_live):
    X = prep_features(cp_live).reset_index(drop=True)
    tr = X[X.is_train]
    tuning, best = [], None
    for i, params in enumerate(HGB_GRID):
        m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED,
                                           early_stopping=False, **params)
        pred, _ = stats.cross_fit(m, tr, FEATURES, "viable_i", proba=True)
        ll = log_loss(tr.viable_i, pred.clip(1e-6, 1 - 1e-6))
        tuning.append({"config": i, **params, "train_oof_logloss": round(float(ll), 5)})
        if best is None or ll < best[0]:
            best = (ll, params)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED,
                                           early_stopping=False, **best[1])
    pred, _ = stats.cross_fit(model, X, FEATURES, "viable_i", proba=True)
    X["expected"] = pred.values

    base = float(tr.viable_i.mean())
    base_t = tr.groupby("t").viable_i.mean()
    test = X[~X.is_train]
    metrics = {"n_train_rows": int(len(tr)), "n_test_rows": int(len(test)), "train_base_rate": round(base, 5),
               "train_base_rate_by_t": {str(k): round(float(v), 5) for k, v in base_t.items()},
               "selected_config": best[1], "tuning_train_oof": tuning, "test": {}}
    for name, sub in [("all_route_runners", test), ("te_only", test[test.is_te])]:
        y, pt = sub.viable_i.values, sub.expected.values
        d = {"n": int(len(sub)), "obs_rate": round(float(y.mean()), 5), "mean_pred": round(float(pt.mean()), 5),
             "model": _clf_metrics(y, pt),
             "baseline_train_rate": _clf_metrics(y, np.full(len(y), base)),
             "baseline_train_rate_by_t": _clf_metrics(y, sub.t.map(base_t).values)}
        d["by_checkpoint"] = {}
        for c in CHECKPOINTS:
            s = sub[sub.t == c]
            d["by_checkpoint"][str(c)] = {"n": int(len(s)), "obs_rate": round(float(s.viable_i.mean()), 5),
                                          "mean_pred": round(float(s.expected.mean()), 5),
                                          "model": _clf_metrics(s.viable_i.values, s.expected.values),
                                          "baseline_train_rate_t": _clf_metrics(
                                              s.viable_i.values, np.full(len(s), base_t[c]))}
        bins = pd.qcut(pt, 5, labels=False, duplicates="drop")
        cal = pd.DataFrame({"bin": bins, "pred": pt, "obs": y}).groupby("bin").agg(
            n=("obs", "size"), mean_pred=("pred", "mean"), obs_rate=("obs", "mean")).reset_index()
        d["calibration_5bin"] = cal.round(4).to_dict("records")
        metrics["test"][name] = d
    return X["expected"].values, metrics


# --------------------------------------------------------------------------- validation
def last_checkpoint_table(cp, flag_values, sel):
    """Per route-runner snap on throw plays: viability at the last live checkpoint (cp <= t_end)."""
    d = cp.loc[sel, KEY + ["cp", "live", "end_event", "is_target", "is_catch", "is_train", "is_te"]].copy()
    d["flag"] = flag_values[sel]
    d = d[d.live & d.end_event.eq("throw")]
    d = d.sort_values("cp").groupby(KEY).tail(1)
    return d


def rate_rows(d, label):
    rows = []
    for split, s0 in [("all", d), ("train", d[d.is_train]), ("test", d[~d.is_train])]:
        for flag in (True, False):
            s = s0[s0.flag == flag]
            n, kt, kc = len(s), int(s.is_target.sum()), int(s.is_catch.sum())
            tlo, thi = stats.wilson_ci(kt, n)
            clo, chi = stats.wilson_ci(kc, kt)
            rows.append({"population": label, "split": split, "viable_mof_last_cp": flag, "n_snaps": n,
                         "targets": kt, "catches": kc, "target_rate": kt / n if n else np.nan,
                         "target_lo": float(tlo), "target_hi": float(thi),
                         "completion_rate": kc / kt if kt else np.nan, "completion_lo": float(clo),
                         "completion_hi": float(chi), "catch_per_snap": kc / n if n else np.nan})
    return rows


def validation(cp, flag_values):
    te_d = last_checkpoint_table(cp, flag_values, cp.is_te.values)
    all_d = last_checkpoint_table(cp, flag_values, np.ones(len(cp), bool))
    rows = rate_rows(te_d, "TE route/chip snaps") + rate_rows(all_d, "all route runners")
    for c in CHECKPOINTS:
        rows += [dict(r, population=f"TE, last cp = {c}") for r in rate_rows(te_d[te_d.cp == c], "")
                 if r["split"] == "all"]
    v = pd.DataFrame(rows)
    lift = {}
    for pop in v.population.unique():
        s = v[(v.population == pop) & (v.split == "all")].set_index("viable_mof_last_cp")
        if True in s.index and False in s.index and s.loc[False, "target_rate"] > 0:
            lift[pop] = round(float(s.loc[True, "target_rate"] / s.loc[False, "target_rate"]), 3)
    n_te_throw = int(cp[cp.is_te & cp.end_event.eq("throw") & (cp.cp == CHECKPOINTS[0])].shape[0])
    info = {"te_throw_snaps": n_te_throw, "te_throw_snaps_with_checkpoint": int(len(te_d)),
            "te_throw_snaps_thrown_before_1.5s_excluded": n_te_throw - int(len(te_d)),
            "target_rate_lift_viable_vs_not": lift}
    return v.round(4), info, te_d


def target_lift(cp, flag_values):
    d = last_checkpoint_table(cp, flag_values, cp.is_te.values)
    a, b = d[d.flag].is_target.mean(), d[~d.flag].is_target.mean()
    return float(a / b) if b > 0 else np.nan


# --------------------------------------------------------------------------- sensitivity
def sensitivity(cp, te_mask, r_te, ref_shrunk):
    rows = []
    for bname, band in SENS_BANDS.items():
        for sep in SENS_SEP:
            v = viable_mask(cp, band=band, sep=sep)
            tcp = cp[te_mask].assign(viable=v[te_mask])
            pr = priors_for(tcp, "viable")
            pl = player_rates(tcp, "viable", pr)
            pl = pl[pl.n >= MIN_N_RANK]
            rho = sps.spearmanr(pl.value_shrunk, ref_shrunk.reindex(pl.index)).correlation
            row = {"band": bname, "y_band": list(band), "min_sep": sep,
                   "is_default": band == MOF_Y and sep == VIABLE_MIN_SEP,
                   "spearman_value_shrunk_vs_default": round(float(rho), 4), "n_players": int(len(pl)),
                   "te_target_lift_last_cp": round(target_lift(cp, v), 3)}
            for c in CHECKPOINTS:
                tg = CP_TAG[c]
                live_c = tcp[(tcp.cp == c) & tcp.live]
                row[f"te_live_rate_{tg}"] = round(float(live_c.viable.mean()), 4)
                row[f"te_uncond_rate_{tg}"] = round(float(tcp[tcp.cp == c].viable.mean()), 4)
                row[f"all_live_rate_{tg}"] = round(float(v[(cp.cp == c).values & cp.live.values].mean()), 4)
                row[f"prior_strength_{tg}"] = round(pr[c]["strength"], 1)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reliability
def split_half(te_cp, prior):
    rel = {}
    for col, lab in [("viable", "pooled_live_viable"), ("resid", "pooled_live_oe")]:
        d = te_cp[te_cp.live].assign(viable=lambda x: x.viable.astype(float))
        for mn in (10, 25):
            r, n = stats.split_half_reliability(d, "nflId", col, min_n=mn)
            rel[f"{lab}_min{mn}"] = {"spearman": None if pd.isna(r) else round(float(r), 4), "n_players": int(n)}
    # the actual player summary (equal-weight mean of shrunk checkpoint rates), each half separately
    halves = {h: player_rates(te_cp[te_cp.is_train == h], "viable", prior) for h in (True, False)}
    for col in ("value_shrunk", "value"):
        for mn in (10, 20):
            a, b = halves[True], halves[False]
            ids = a.index[(a.n >= mn)].intersection(b.index[b.n >= mn])
            x, y = a.loc[ids, col], b.loc[ids, col]
            ok = x.notna() & y.notna()
            rho = sps.spearmanr(x[ok], y[ok]).correlation if ok.sum() >= 5 else np.nan
            rel[f"summary_{col}_min{mn}_route_snaps_per_half"] = {
                "spearman": None if pd.isna(rho) else round(float(rho), 4), "n_players": int(ok.sum())}
    return rel


# --------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    p, r, cp, n_te_route_all = load()
    cp["viable"] = viable_mask(cp)
    te_mask = cp.is_te.values

    # ----- expected model on ALL route runners' live checkpoints (position not a feature)
    live_idx = np.flatnonzero(cp.live.values)
    exp, model_metrics = fit_expected(cp.iloc[live_idx])
    cp["expected"] = np.nan
    cp.loc[cp.index[live_idx], "expected"] = exp
    cp["resid"] = np.where(cp.live, cp.viable.astype(float) - cp.expected, np.nan)

    te_cp = cp[te_mask].copy()
    r_te = r[r.is_te]
    prior = priors_for(te_cp, "viable")
    players, eb_info = player_table(te_cp, r_te, prior, rng)

    # ----- league tables
    league = {"te": {}, "by_position_live": {}, "te_by_alignment_live": {}, "te_by_assignment_live": {}}
    for c in CHECKPOINTS:
        s = te_cp[te_cp.cp == c]
        sl = s[s.live]
        k, n = int(sl.viable.sum()), int(len(sl))
        lo, hi = stats.wilson_ci(k, n)
        league["te"][str(c)] = {
            "n_te_route_snaps": int(len(s)), "n_live": n, "live_share": round(n / len(s), 4),
            "p_viable_given_live": round(k / n, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_viable_unconditional": round(float(s.viable.mean()), 4),
            "mean_expected_live": round(float(sl.expected.mean()), 4),
            "te_mean_oe_live": round(float(sl.resid.mean()), 4),
            "component_rates_live": {
                "in_y_band": round(float(sl.y.between(*MOF_Y).mean()), 4),
                "in_x_rel_band": round(float(sl.x_rel.between(*MOF_X_REL).mean()), 4),
                "sep_ok": round(float((sl.sep >= VIABLE_MIN_SEP).mean()), 4),
                "lane_ok": round(float((sl.lane_clear >= VIABLE_MIN_LANE).mean()), 4),
                "in_mof_zone_y_and_x": round(float((sl.y.between(*MOF_Y) & sl.x_rel.between(*MOF_X_REL)).mean()), 4),
            },
        }
        allc = cp[(cp.cp == c) & cp.live]
        league["by_position_live"][str(c)] = allc.groupby("officialPosition").viable.agg(
            ["mean", "count"]).query("count >= 100").round(4).to_dict("index")
        league["te_by_alignment_live"][str(c)] = sl.groupby("alignment").viable.agg(["mean", "count"]).round(4).to_dict("index")
        league["te_by_assignment_live"][str(c)] = sl.groupby("assignment").viable.agg(["mean", "count"]).round(4).to_dict("index")
    league["te_summary_equal_weight_live"] = round(float(np.mean(
        [league["te"][str(c)]["p_viable_given_live"] for c in CHECKPOINTS])), 4)

    # ----- validation, sensitivity, reliability
    val, val_info, _ = validation(cp, cp.viable.values)
    ref = players.set_index("nflId").value_shrunk
    sens = sensitivity(cp, te_mask, r_te, ref)
    rel = split_half(te_cp, prior)

    # ----- play-level output (one row per TE route snap, checkpoints wide)
    wcols = ["live", "viable", "t_frame", "y", "x_rel", "sep", "lane_clear", "expected"]
    w = to_wide(te_cp, wcols)
    w.columns = [f"{a}_{CP_TAG[b]}" for a, b in w.columns]
    w = w.reset_index()
    last = last_checkpoint_table(cp, cp.viable.values, te_mask)[KEY + ["cp", "flag"]].rename(
        columns={"cp": "last_cp_before_throw", "flag": "viable_last_cp"})
    base_cols = KEY + ["displayName", "possessionTeam", "week", "is_train", "assignment", "alignment",
                       "x_rel_snap", "y_snap", "abs_lat", "receiver_num", "n_rec_side", "release_t", "t_end",
                       "end_event", "is_target", "is_catch", "offenseFormation", "personnelO",
                       "pff_passCoverage", "pff_passCoverageType", "pff_playAction", "dropBackType", "down",
                       "yardsToGo", "red_zone"]
    plays_out = r_te[base_cols].merge(w, on=KEY, how="left").merge(last, on=KEY, how="left")
    for c in ("live", "viable"):
        for cc in CHECKPOINTS:
            col = f"{c}_{CP_TAG[cc]}"
            plays_out[col] = plays_out[col].astype(bool)
    plays_out.to_csv(OUT_DIR / "plays.csv", index=False, float_format="%.4f")
    players.to_csv(OUT_DIR / "players.csv", index=False, float_format="%.5g")
    sens.to_csv(OUT_DIR / "sensitivity.csv", index=False)
    val.to_csv(OUT_DIR / "validation.csv", index=False)

    top = players[players.n >= MIN_N_RANK].head(5)
    top_oe = players[players.n >= MIN_N_RANK].sort_values("mofa_oe_shrunk", ascending=False).head(5)
    summary = {
        "metric": "Middle-of-Field Access", "slug": SLUG, "higher_is_better": True,
        "definition": {
            "checkpoints_s": CHECKPOINTS,
            "frame_rule": f"receiver_frames frame nearest each checkpoint (|t - checkpoint| <= {FRAME_TOL:.3f} s; "
                          "frames are on an exact 0.1 s grid so this is the frame at t)",
            "viable_mof_window_if_all": {
                "y_band_between_numbers": list(MOF_Y), "x_rel_band_yd_past_los": list(MOF_X_REL),
                "min_sep_yd (config.VIABLE_MIN_SEP)": VIABLE_MIN_SEP,
                "min_lane_clear_yd (config.VIABLE_MIN_LANE)": VIABLE_MIN_LANE,
                "missing sep/lane": "treated as not viable",
            },
            "live_rule": "checkpoint t is live iff t <= t_end (routes.t_end = end of dropback: throw frame on "
                         "throw plays, sack/scramble/other stop frame otherwise)",
            "population": "TE snaps with te_plays.assignment in {route, chip_release} that have tracking",
            "primary": "P(viable at t | TE route/chip_release, live at t), per checkpoint",
            "unconditional": "P(viable at t | TE route/chip_release); not live counted as not viable",
            "player_value": "value = equal-weight mean of the 3 raw per-checkpoint live rates (NaN if a checkpoint "
                            "has 0 live snaps); value_shrunk = equal-weight mean of the 3 beta-shrunk rates "
                            "(prior = TE league live rate at that checkpoint, strength by method of moments)",
            "intervals": "per-checkpoint: Wilson 95%; lo/hi (raw value), value_shrunk_lo/hi, mofa_oe_lo/hi: "
                         f"play-level bootstrap within player ({N_BOOT} reps, the 3 checkpoints of a play "
                         "resampled together; priors held fixed)",
            "n_column": "n = TE route/chip_release snaps with tracking (live counts per checkpoint in n_live_t*)",
            "context_adjusted": "expected = P(viable | context) at each live checkpoint from a cross-fitted HGB "
                                "classifier fit on ALL route runners (position not a feature); oe_t* = mean(viable "
                                "- expected) over the TE's live snaps at t; mofa_oe = equal-weight mean of oe_t*; "
                                "oe_t*_shrunk = normal-normal EB toward the TE pooled mean residual at t "
                                "(se2 = pooled within-player residual variance / n_live, tau2 by MoM over TEs "
                                "with n >= 40); "
                                "mofa_oe_shrunk = mean of the three",
        },
        "parameters": {"min_n_rank_top5_spearman": MIN_N_RANK, "min_n_prior_estimation": MIN_N_PRIOR,
                       "n_boot": N_BOOT, "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS, "seed": SEED},
        "sample": {
            "te_route_chip_snaps_total": int(n_te_route_all),
            "te_route_chip_snaps_used": int(len(r_te)),
            "dropped_no_tracking": int(n_te_route_all - len(r_te)),
            "n_route_chip": r_te.assignment.value_counts().to_dict(),
            "n_tes": int(r_te.nflId.nunique()), "n_tes_min40": int((players.n >= MIN_N_RANK).sum()),
            "all_route_runner_snaps": int(len(r)),
            "live_rows_all_runners": int(cp.live.sum()),
            "frames_missing_on_live_checkpoints": int(cp.frame_missing.sum()),
            "te_live_n_by_checkpoint": {str(c): league["te"][str(c)]["n_live"] for c in CHECKPOINTS},
        },
        "league": league,
        "shrinkage_prior_by_checkpoint": {str(c): v for c, v in prior.items()},
        "oe_eb_by_checkpoint": {str(c): v for c, v in eb_info.items()},
        "expected_model": {"features": FEATURES, "categorical": CAT_FEATS,
                           "target": "viable MOF window at a live checkpoint",
                           "rows": "all route runners x live checkpoints",
                           "estimator": "HistGradientBoostingClassifier (categorical from dtype); stats.cross_fit "
                                        "GroupKFold(gameId) on train weeks, refit on all train for test weeks",
                           **model_metrics},
        "validation": {**val_info, "rule": "throw plays; viability at the last checkpoint with checkpoint <= "
                                           "t_end (throw frame); snaps thrown before 1.5 s excluded",
                       "table": val.to_dict("records")},
        "sensitivity": sens.to_dict("records"),
        "sensitivity_summary": {
            "spearman_min": float(sens.spearman_value_shrunk_vs_default.min()),
            "te_live_rate_t25_range": [float(sens.te_live_rate_t25.min()), float(sens.te_live_rate_t25.max())],
        },
        "split_half_reliability": rel,
        "top5_min40": top[["nflId", "displayName", "team", "n", "value", "value_shrunk", "value_shrunk_lo",
                           "value_shrunk_hi", "rate_t15", "rate_t25", "rate_t35", "n_live_t35",
                           "mofa_oe_shrunk"]].round(4).to_dict("records"),
        "top5_oe_min40": top_oe[["nflId", "displayName", "team", "n", "mofa_expected", "mofa_oe",
                                 "mofa_oe_shrunk", "mofa_oe_lo", "mofa_oe_hi"]].round(4).to_dict("records"),
        "caveats": [
            "3.5 s checkpoint is live on only ~27% of TE route snaps; it is the noisiest component yet gets equal "
            "weight in the summary (shrinkage pulls low-n 3.5 s rates strongly toward the league prior)",
            "live-conditioning selects on play length: long-developing dropbacks (more often play action / "
            "deep shots / pressure-free) dominate 3.5 s; compare checkpoints with that in mind",
            "viability = nearest-defender separation and a straight QB->TE lane at one frame; no ball-flight, "
            "defender closing speed, or QB sightline/progression model",
            "MOF band is a fixed absolute y band (numbers); it does not move with hash/ball position",
            "x_rel >= 0 excludes flat/behind-LOS checkdowns, which are a real part of the TE checkdown role",
            "expected model has no position feature, so TE mean OE vs all route runners partly reflects TE "
            "alignment/route mix not captured by the snap-alignment features",
            "route vs block assignment is a play-call choice; chip_release snaps are included in the denominator",
            "test weeks are only 2 weeks, so split-half reliability rests on small test-week samples",
            "split-half reliability is ~0 for every variant (see split_half_reliability) and beta prior strengths "
            "are large (100-530 pseudo-snaps): in 8 weeks MOFA mostly reflects scheme/context, not a stable "
            "player trait; down-weight it in any composite",
        ],
        "shared_code_notes": [
            f"{n_te_route_all - len(r_te)} te_plays route/chip rows have no snap frame / tracking and no routes row; dropped",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(json.dumps({"sample": summary["sample"], "league_te": league["te"],
                      "prior": summary["shrinkage_prior_by_checkpoint"], "reliability": rel}, indent=1, default=str))
    print(json.dumps({k: model_metrics[k] for k in ["selected_config", "train_base_rate", "test"]}, indent=1,
                     default=str))
    print(val.to_string())
    print(json.dumps(val_info, indent=1))
    print(sens.to_string())
    cols = ["displayName", "team", "n", "value", "value_shrunk", "value_shrunk_lo", "value_shrunk_hi",
            "rate_t15", "rate_t25", "rate_t35", "n_live_t35", "mofa_oe_shrunk"]
    print(top[cols].to_string())
    print(top_oe[["displayName", "team", "n", "mofa_expected", "mofa_oe", "mofa_oe_shrunk", "mofa_oe_lo",
                  "mofa_oe_hi"]].to_string())


if __name__ == "__main__":
    main()

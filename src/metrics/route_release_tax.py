"""Route Release Tax (RRT) for tight ends.

Definition (tight-end-metrics.md): time from the ball snap to the first frame in which the TE
begins his route. Release uses the shared definition (config.RELEASE_*): first frame where
displacement from the snap position (downfield or lateral) >= 1.5 yd AND speed >= 2.5 yd/s held
for 3 frames.

Population: TE plays with assignment in {route, chip_release}.
  * released plays          -> tax = release_t
  * never released (NaN)    -> right-censored at t_end (end of dropback); tax = t_end, flagged
  * no tracking row at all  -> excluded from tax, counted and reported
Per-play tax is winsorized at TAX_CAP seconds (a handful of scramble plays reach 8-16 s).

Context-adjusted: release tax over expected (RTOE) = tax - E[tax | assignment, alignment, play
action, dropback type, formation, receiver_num, n_rec_side, x_rel_snap, abs lateral from ball],
HistGradientBoostingRegressor cross-fit on train weeks (stats.cross_fit), evaluated on test weeks.

Lower tax = faster release (higher_is_better=False). A delayed release is NOT automatically bad:
context tables (assignment, alignment, play action, route depth, target/catch outcomes) are written
alongside the player numbers.

Run: .venv/bin/python -m src.metrics.route_release_tax
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.common import io, stats  # noqa: E402
from src.common.config import (N_FOLDS, OUT, RELEASE_MIN_DISP, RELEASE_MIN_SPEED,  # noqa: E402
                               RELEASE_SUSTAIN, SEED, TEST_WEEKS, TRAIN_WEEKS)

SLUG = "route_release_tax"
OUT_DIR = OUT / SLUG
ASSIGNMENTS = ["route", "chip_release"]
TAX_CAP = 5.0          # seconds, winsorization of per-play tax
MIN_N_TOP = 30         # min route+chip plays for top lists / sensitivity rank correlations
EB_HYPER_MIN_N = 10    # players used to estimate EB hyper-parameters
SENS_DISP = [1.0, 1.5, 2.0]
SENS_SPEED = [2.0, 2.5, 3.0]
SENS_SUSTAIN = 3
KEY = ["gameId", "playId", "nflId"]

CAT_FEATS = ["assignment", "alignment", "play_action", "dropBackType", "offenseFormation"]
NUM_FEATS = ["receiver_num", "n_rec_side", "x_rel_snap", "abs_lat"]
FEATURES = CAT_FEATS + NUM_FEATS

DEPTH_BINS = [-np.inf, 0, 5, 10, 15, 20, np.inf]
DEPTH_LABELS = ["<0", "0-5", "5-10", "10-15", "15-20", "20+"]
TAX_BINS = [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, np.inf]
TAX_LABELS = ["<=0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", ">3.0"]

PARAM_GRID = [
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1.0),
    dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=3.0),
    dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=0.0),
]


# --------------------------------------------------------------------------- data
def load_population():
    te = io.te_plays()
    te = te[te.assignment.isin(ASSIGNMENTS)].copy()
    rt = io.routes()[KEY + ["t_end", "abs_lat", "end_event"]]
    p = io.plays()[["gameId", "playId", "possessionTeam", "pff_playAction", "dropBackType",
                    "offenseFormation", "pressure", "time_to_throw"]]
    df = te.merge(rt, on=KEY, how="left").merge(p, on=["gameId", "playId"], how="left")
    df["has_tracking"] = df.t_end.notna()
    df["released"] = df.release_t.notna()
    df["censored"] = df.has_tracking & ~df.released
    df["tax_uncapped"] = np.where(df.released, df.release_t, df.t_end)
    df["tax"] = np.minimum(df.tax_uncapped, TAX_CAP)
    df["capped"] = df.tax_uncapped > TAX_CAP
    df["play_action"] = df.pff_playAction.map({1: "PA", 0: "no_PA"}).fillna("NA")
    df["dropBackType"] = df.dropBackType.fillna("NA")
    df["offenseFormation"] = df.offenseFormation.fillna("NA")
    df["depth_bin"] = pd.cut(df.max_depth, DEPTH_BINS, labels=DEPTH_LABELS, right=False)
    df["tax_bin"] = pd.cut(df.release_t, TAX_BINS, labels=TAX_LABELS, right=True).astype(object)
    df.loc[df.censored, "tax_bin"] = "never_released"
    df.loc[~df.has_tracking, "tax_bin"] = "no_tracking"
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- context tables
def ctx_table(df, by):
    d = df[df.has_tracking]
    g = d.groupby(by, observed=True)
    t = g.agg(n=("tax", "size"), mean_tax=("tax", "mean"), median_tax=("tax", "median"),
              sd_tax=("tax", "std"), never_released_rate=("censored", "mean"),
              mean_release_t_released=("release_t", "mean"), mean_max_depth=("max_depth", "mean"),
              targets=("is_target", "sum"), catches=("is_catch", "sum"))
    t["target_rate"] = t.targets / t.n
    t["target_lo"], t["target_hi"] = stats.wilson_ci(t.targets, t.n)
    t["catch_rate"] = t.catches / t.n
    t["catch_per_target"] = t.catches / t.targets.replace(0, np.nan)
    return t.reset_index().round(4)


# --------------------------------------------------------------------------- model
def _prep_X(df):
    X = df.copy()
    for c in CAT_FEATS:
        X[c] = pd.Categorical(X[c].astype(str))
    return X


def _hgb(params):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED,
                                         early_stopping=False, **params)


def tune(train):
    """Pick hyper-params by GroupKFold(game) OOF RMSE on train weeks only."""
    res = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for i, prm in enumerate(PARAM_GRID):
        oof = np.full(len(train), np.nan)
        for fi, oi in gkf.split(train, groups=train.gameId):
            m = _hgb(prm).fit(train.iloc[fi][FEATURES], train.iloc[fi]["tax"])
            oof[oi] = m.predict(train.iloc[oi][FEATURES])
        res.append({"config": i, **prm, "oof_rmse": float(np.sqrt(np.mean((oof - train.tax) ** 2)))})
    best = min(res, key=lambda r: r["oof_rmse"])
    return PARAM_GRID[best["config"]], res


def fit_expected(df):
    m = df[df.has_tracking].copy()
    X = _prep_X(m)
    tr = X[X.is_train]
    best, grid = tune(tr)
    pred, full = stats.cross_fit(_hgb(best), X, FEATURES, "tax")
    m["expected_tax"] = pred
    m["rtoe"] = m.tax - m.expected_tax

    te = m[~m.is_train]
    y = te.tax
    base = np.full(len(te), tr.tax.mean())
    asg_mean = tr.groupby("assignment", observed=True).tax.mean()
    base_asg = te.assignment.map(asg_mean).astype(float).values

    def _m(pr):
        return {"rmse": float(np.sqrt(mean_squared_error(y, pr))), "mae": float(mean_absolute_error(y, pr)),
                "r2": float(r2_score(y, pr))}

    # also evaluate on released-only test rows (uncensored targets)
    rel = te.released.values
    metrics = {
        "n_train": int(len(tr)), "n_test": int(len(te)), "best_params": best, "tuning_grid": grid,
        "test_model": _m(te.expected_tax.values), "test_baseline_train_mean": _m(base),
        "test_baseline_assignment_mean": _m(base_asg),
        "test_model_released_only": {
            "rmse": float(np.sqrt(mean_squared_error(y[rel], te.expected_tax[rel]))),
            "r2": float(r2_score(y[rel], te.expected_tax[rel]))},
        "oof_train_model": {"rmse": float(np.sqrt(mean_squared_error(m[m.is_train].tax,
                                                                       m[m.is_train].expected_tax))),
                            "r2": float(r2_score(m[m.is_train].tax, m[m.is_train].expected_tax))},
    }
    # 5-bin calibration of expected vs observed on test
    q = pd.qcut(te.expected_tax, 5, duplicates="drop")
    cal = te.groupby(q, observed=True).agg(n=("tax", "size"), mean_expected=("expected_tax", "mean"),
                                           mean_observed=("tax", "mean")).reset_index(drop=True)
    metrics["test_calibration_5bin"] = cal.round(4).to_dict("records")
    df = df.merge(m[KEY + ["expected_tax", "rtoe"]], on=KEY, how="left")
    return df, metrics


# --------------------------------------------------------------------------- player table
def _eb(means, ns, pooled_sd):
    """Normal-normal EB, same formula as stats.eb_shrink_mean, but hyper-parameters (grand mean,
    tau^2) estimated from players with n >= EB_HYPER_MIN_N and then applied to every player.

    Work-around: stats.eb_shrink_mean estimates tau^2 = var(means) - mean(se^2) over ALL players;
    a few n=1-3 players with extreme values make that negative for RTOE, so it floors tau^2 at 1e-6
    and shrinks every player fully to the grand mean. Restricting the moment estimate to n >= 10
    gives a positive, stable tau^2 for both tax and RTOE.
    """
    means, ns = np.asarray(means, float), np.asarray(ns, float)
    se2 = pooled_sd ** 2 / np.maximum(ns, 1)
    h = ns >= EB_HYPER_MIN_N
    grand = np.sum(means[h] * ns[h]) / np.sum(ns[h])
    tau2 = max(np.var(means[h]) - np.mean(se2[h]), 1e-6)
    w = tau2 / (tau2 + se2)
    return w * means + (1 - w) * grand, {"grand_mean": float(grand), "tau2": float(tau2),
                                         "n_players_hyper": int(h.sum())}


def _shared_eb(means, ns, pooled_sd):
    return stats.eb_shrink_mean(means, ns, np.full(len(means), pooled_sd))


def player_table(df):
    d = df[df.has_tracking]
    g = d.groupby("nflId")
    pt = g.agg(n=("tax", "size"), value=("tax", "mean"), median=("tax", "median"), sd=("tax", "std"),
               n_route=("assignment", lambda s: (s == "route").sum()),
               n_chip=("assignment", lambda s: (s == "chip_release").sum()),
               n_released=("released", "sum"), n_never_released=("censored", "sum"),
               mean_release_t_released_only=("release_t", "mean"),
               mean_expected=("expected_tax", "mean"), rtoe=("rtoe", "mean"), rtoe_sd=("rtoe", "std"),
               inline_share=("alignment", lambda s: (s == "inline").mean()),
               detached_share=("alignment", lambda s: (s == "detached").mean()),
               pa_share=("play_action", lambda s: (s == "PA").mean()),
               target_rate=("is_target", "mean"), catch_rate=("is_catch", "mean"),
               mean_max_depth=("max_depth", "mean"))
    pt["chip_share"] = pt.n_chip / pt.n
    pt["never_release_rate"] = pt.n_never_released / pt.n
    pt["n_no_tracking"] = df[~df.has_tracking].groupby("nflId").size().reindex(pt.index).fillna(0).astype(int)
    se = pt.sd / np.sqrt(pt.n)
    pt["lo"], pt["hi"] = pt.value - 1.96 * se, pt.value + 1.96 * se
    rse = pt.rtoe_sd / np.sqrt(pt.n)
    pt["rtoe_lo"], pt["rtoe_hi"] = pt.rtoe - 1.96 * rse, pt.rtoe + 1.96 * rse
    # EB with pooled within-player SD (per-player SDs are unstable / undefined for tiny n)
    pooled_tax = np.sqrt((d.tax - d.groupby("nflId").tax.transform("mean")).pow(2).sum() / (len(d) - len(pt)))
    pooled_rtoe = np.sqrt((d.rtoe - d.groupby("nflId").rtoe.transform("mean")).pow(2).sum() / (len(d) - len(pt)))
    pt["value_shrunk"], hyp_tax = _eb(pt.value, pt.n, pooled_tax)
    pt["rtoe_shrunk"], hyp_rtoe = _eb(pt.rtoe, pt.n, pooled_rtoe)
    shared_rtoe = _shared_eb(pt.rtoe, pt.n, pooled_rtoe)
    shared_tax = _shared_eb(pt.value, pt.n, pooled_tax)
    eb_info = {"pooled_within_player_sd_tax": float(pooled_tax),
               "pooled_within_player_sd_rtoe": float(pooled_rtoe),
               "hyper_min_n": EB_HYPER_MIN_N, "tax": hyp_tax, "rtoe": hyp_rtoe,
               "shared_eb_shrink_mean_rtoe_sd_of_shrunk": float(np.std(shared_rtoe)),
               "shared_eb_shrink_mean_tax_spearman_vs_local": float(
                   sps.spearmanr(shared_tax, pt.value_shrunk).correlation)}
    names = df.groupby("nflId").displayName.first()
    team = df.groupby("nflId").possessionTeam.agg(lambda s: s.value_counts().index[0])
    pt.insert(0, "displayName", names.reindex(pt.index))
    pt.insert(1, "team", team.reindex(pt.index))
    pt["higher_is_better"] = False
    pt["meets_min_n"] = pt.n >= MIN_N_TOP
    pt = pt.reset_index().sort_values("value_shrunk", ascending=True)
    cols = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "median", "sd",
            "rtoe", "rtoe_shrunk", "rtoe_lo", "rtoe_hi", "mean_expected", "mean_release_t_released_only",
            "n_route", "n_chip", "chip_share", "inline_share", "detached_share", "pa_share",
            "n_released", "n_never_released", "never_release_rate", "n_no_tracking",
            "target_rate", "catch_rate", "mean_max_depth", "higher_is_better", "meets_min_n"]
    return pt[cols], eb_info


# --------------------------------------------------------------------------- sensitivity
def recompute_releases(df):
    """Recompute release_t from receiver_frames for every (disp, speed) setting."""
    ids = df.nflId.dropna().unique().tolist()
    rf = io.receiver_frames(columns=["gameId", "playId", "nflId", "frameId", "t", "x_rel", "y", "s"],
                            filters=[("nflId", "in", ids)])
    rf = rf.merge(df.loc[df.has_tracking, KEY], on=KEY, how="inner")
    rf = rf.sort_values(KEY + ["frameId"]).reset_index(drop=True)
    grp = rf.groupby(KEY, sort=False)
    snap_x = grp.x_rel.transform("first")
    snap_y = grp.y.transform("first")
    disp = np.maximum(np.maximum(rf.x_rel - snap_x, 0), (rf.y - snap_y).abs())
    # flag where each row's group differs from the row k ahead (to block cross-group shifts)
    gid = grp.ngroup().values
    out = {}
    for spd in SENS_SPEED:
        fast = (rf.s >= spd).values
        sus = fast.copy()
        for k in range(1, SENS_SUSTAIN):
            nxt = np.zeros(len(fast), bool)
            same = np.zeros(len(fast), bool)
            nxt[:-k] = fast[k:]
            same[:-k] = gid[k:] == gid[:-k]
            sus &= nxt & same
        for dsp in SENS_DISP:
            cond = sus & (disp.values >= dsp)
            first = rf.loc[cond].groupby(KEY, sort=False).t.first()
            out[(dsp, spd)] = first.rename(f"release_t_d{dsp}_s{spd}")
    return out


def sensitivity(df, rel_by_setting):
    d = df[df.has_tracking].copy()
    base_key = (RELEASE_MIN_DISP, RELEASE_MIN_SPEED)
    rows, taxes = [], {}
    for (dsp, spd), s in rel_by_setting.items():
        r = d[KEY].merge(s.reset_index(), on=KEY, how="left")[s.name].values
        d[s.name] = r
        tax = np.minimum(np.where(np.isnan(r), d.t_end, r), TAX_CAP)
        taxes[(dsp, spd)] = tax
        rows.append({"disp_yd": dsp, "speed_yds": spd, "sustain_frames": SENS_SUSTAIN,
                     "mean_tax": float(np.mean(tax)), "median_tax": float(np.median(tax)),
                     "mean_tax_route": float(np.mean(tax[d.assignment.values == "route"])),
                     "mean_tax_chip": float(np.mean(tax[d.assignment.values == "chip_release"])),
                     "n_never_released": int(np.isnan(r).sum()),
                     "never_release_rate": float(np.isnan(r).mean())})
    # reproduction check of the shared default
    b = d[rel_by_setting[base_key].name].values
    match = np.isclose(np.nan_to_num(b, nan=-1), np.nan_to_num(d.release_t.values, nan=-1), atol=1e-4)
    d["_pid"] = d.nflId
    cnt = d.groupby("_pid").size()
    keep = cnt[cnt >= MIN_N_TOP].index
    base_rank = pd.Series(taxes[base_key], index=d.index).groupby(d._pid).mean().loc[keep]
    for row in rows:
        t = pd.Series(taxes[(row["disp_yd"], row["speed_yds"])], index=d.index).groupby(d._pid).mean().loc[keep]
        row["spearman_player_mean_vs_default"] = float(sps.spearmanr(t, base_rank).correlation)
        row["n_players"] = int(len(keep))
        row["is_default"] = (row["disp_yd"], row["speed_yds"]) == base_key
    alt_cols = [s.name for s in rel_by_setting.values()]
    return pd.DataFrame(rows), float(match.mean()), d[KEY + alt_cols]


# --------------------------------------------------------------------------- main
def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Interval):
        return str(o)
    return str(o)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_population()
    df, model = fit_expected(df)

    # context tables
    ctx = {
        "by_assignment": ctx_table(df, ["assignment"]),
        "by_alignment": ctx_table(df, ["alignment"]),
        "by_play_action": ctx_table(df, ["play_action"]),
        "by_dropback": ctx_table(df, ["dropBackType"]),
        "by_assignment_alignment": ctx_table(df, ["assignment", "alignment"]),
        "by_assignment_play_action": ctx_table(df, ["assignment", "play_action"]),
        "by_depth_bin": ctx_table(df, ["depth_bin"]),
        "by_assignment_depth_bin": ctx_table(df, ["assignment", "depth_bin"]),
        "by_tax_bin": ctx_table(df, ["tax_bin"]),
        "by_assignment_tax_bin": ctx_table(df, ["assignment", "tax_bin"]),
    }
    m = df[df.has_tracking].copy()
    m["rtoe_quintile"] = pd.qcut(m.rtoe, 5, labels=["Q1_fastest", "Q2", "Q3", "Q4", "Q5_slowest"])
    ctx["by_rtoe_quintile"] = ctx_table(m, ["rtoe_quintile"])
    tax_order = TAX_LABELS + ["never_released"]
    for k in ("by_tax_bin", "by_assignment_tax_bin"):
        t = ctx[k]
        t["_o"] = t.tax_bin.map({v: i for i, v in enumerate(tax_order)})
        ctx[k] = t.sort_values([c for c in ["assignment", "_o"] if c in t]).drop(columns="_o")
    for k, t in ctx.items():
        t.to_csv(OUT_DIR / f"context_{k}.csv", index=False)

    # players
    pt, eb_info = player_table(df)
    pt.round(4).to_csv(OUT_DIR / "players.csv", index=False)
    rel = {}
    for col in ("tax", "rtoe"):
        for mn in (10, 15):
            r, n = stats.split_half_reliability(m, "nflId", col, min_n=mn)
            rel[f"{col}_spearman_min{mn}_each_half"] = r
            rel[f"{col}_n_players_min{mn}_each_half"] = n
    # descriptive: how much of raw-tax stability is usage (chip share)?
    r_chip, n_chip = stats.split_half_reliability(m.assign(chip=(m.assignment == "chip_release").astype(float)),
                                                  "nflId", "chip")
    rel["chip_share_spearman_min10_each_half"] = r_chip
    r_rt, n_rt = stats.split_half_reliability(m[m.assignment == "route"], "nflId", "tax")
    rel["tax_route_only_spearman_min10_each_half"] = r_rt
    rel["tax_route_only_n_players"] = n_rt

    # sensitivity
    rel_by_setting = recompute_releases(df)
    sens, repro, alt = sensitivity(df, rel_by_setting)
    sens.round(4).to_csv(OUT_DIR / "sensitivity.csv", index=False)

    # plays
    play_cols = KEY + ["displayName", "possessionTeam", "week", "split", "is_train", "assignment",
                       "pff_blockType", "alignment", "play_action", "dropBackType", "offenseFormation",
                       "receiver_num", "n_rec_side", "x_rel_snap", "abs_lat", "has_tracking", "released",
                       "censored", "release_t", "t_end", "end_event", "tax_uncapped", "tax", "capped",
                       "expected_tax", "rtoe", "max_depth", "depth_bin", "tax_bin", "is_target",
                       "is_catch", "pressure", "time_to_throw"]
    plays_out = df[play_cols].merge(alt, on=KEY, how="left")
    plays_out.round(4).to_csv(OUT_DIR / "plays.csv", index=False)

    top = pt[pt.meets_min_n]
    top5_raw = top.nsmallest(5, "value_shrunk")[["displayName", "team", "n", "value", "value_shrunk",
                                                  "median", "chip_share", "rtoe_shrunk"]]
    top5_rtoe = top.nsmallest(5, "rtoe_shrunk")[["displayName", "team", "n", "rtoe", "rtoe_shrunk",
                                                  "value", "chip_share"]]
    n_trk = int(df.has_tracking.sum())
    summary = {
        "metric": "Route Release Tax",
        "slug": SLUG,
        "definition": ("Seconds from snap (t=0) to the first frame where the TE's displacement from his snap "
                       "position (max of downfield gain, |lateral|) >= RELEASE_MIN_DISP yd AND speed >= "
                       "RELEASE_MIN_SPEED yd/s for RELEASE_SUSTAIN consecutive frames. Never-released plays "
                       "are right-censored at t_end (end of dropback/throw frame); per-play tax winsorized at "
                       f"{TAX_CAP} s. Player value = mean per-play tax (lower = faster release). RTOE = tax - "
                       "cross-fit E[tax | context]."),
        "higher_is_better": False,
        "parameters": {"release_min_disp_yd": RELEASE_MIN_DISP, "release_min_speed_yds": RELEASE_MIN_SPEED,
                       "release_sustain_frames": RELEASE_SUSTAIN, "tax_cap_s": TAX_CAP,
                       "min_n_top_list": MIN_N_TOP, "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS,
                       "model_features": FEATURES, "depth_bins_yd": DEPTH_LABELS, "tax_bins_s": TAX_LABELS,
                       "ci": "normal approx mean +/- 1.96*sd/sqrt(n) (per-player sd)",
                       "empirical_bayes": eb_info},
        "sample": {
            "te_route_or_chip_plays": int(len(df)),
            "by_assignment": df.assignment.value_counts().to_dict(),
            "with_tracking": n_trk,
            "no_tracking_excluded": int((~df.has_tracking).sum()),
            "released": int(df.released.sum()),
            "never_released_censored_at_t_end": int(df.censored.sum()),
            "never_released_by_assignment": df[df.censored].assignment.value_counts().to_dict(),
            "never_released_mean_t_end": float(df[df.censored].t_end.mean()),
            "never_released_mean_max_depth": float(df[df.censored].max_depth.mean()),
            "never_released_target_rate": float(df[df.censored].is_target.mean()),
            "capped_at_tax_cap": int(df.capped.sum()),
            "n_players": int(len(pt)), "n_players_min_n": int(pt.meets_min_n.sum()),
            "train_rows": int(m.is_train.sum()), "test_rows": int((~m.is_train).sum()),
        },
        "overall": {"mean_tax": float(m.tax.mean()), "median_tax": float(m.tax.median()),
                    "mean_release_t_released_only": float(m.release_t.mean()),
                    "mean_tax_uncapped": float(m.tax_uncapped.mean())},
        "expected_model": model,
        "split_half_reliability": rel,
        "sensitivity": {"default_reproduction_match_rate": repro, "settings": sens.to_dict("records")},
        "context": {k: v.to_dict("records") for k, v in ctx.items()},
        "top5_fastest_release_shrunk_min30": top5_raw.round(3).to_dict("records"),
        "top5_fastest_rtoe_shrunk_min30": top5_rtoe.round(3).to_dict("records"),
        "caveats": [
            "Delayed release is not automatically negative: chip_release plays release ~0.7 s later by design, "
            "and late releases are tied to shallower routes and specific target patterns (see context tables).",
            "Never-released plays are right-censored at t_end (a lower bound on true release time); they are "
            "kept in the mean and also reported separately (never_release_rate).",
            "Plays with no route-tracking row are excluded from tax and counted (n_no_tracking).",
            "max_depth is measured up to the throw frame, so it is partly a consequence of release time and "
            "time-to-throw; depth tables are descriptive, not causal.",
            "dropBackType (scramble/rollout) is post-snap information, used as context per spec.",
            "Expected model uses assignment (PFF chip label) as a feature, so RTOE compares chips to chips.",
            "Targets parsed from playDescription (no ball-flight tracking); catch = completion to that TE.",
            "Weeks 1-8 of 2021 only; per-player n is modest, use shrunk values and intervals.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json)

    print(f"plays {len(df)} | tracked {n_trk} | censored {int(df.censored.sum())} | "
          f"no tracking {int((~df.has_tracking).sum())}")
    print("test model", model["test_model"], "\nbaseline", model["test_baseline_train_mean"],
          "\nasg baseline", model["test_baseline_assignment_mean"])
    print("split-half", rel)
    print("EB", eb_info)
    print("default reproduction", repro)
    print(sens.to_string())
    print(ctx["by_assignment"].to_string())
    print(ctx["by_alignment"].to_string())
    print(ctx["by_play_action"].to_string())
    print(ctx["by_depth_bin"].to_string())
    print(ctx["by_tax_bin"].to_string())
    print(ctx["by_rtoe_quintile"].to_string())
    print(top5_raw.to_string())
    print(top5_rtoe.to_string())


if __name__ == "__main__":
    main()

"""Matchup Separation Over Expected (SOE)  --  slug: matchup_soe

SOE = observed separation (nearest defender distance) at a route landmark
      - E(separation | alignment, coverage, defender position at snap, route depth /
          lateral position at the landmark, time since snap, play context)

Landmarks: t in {1.0, 1.5, 2.0, 2.5, 3.0} s after the snap plus the throw frame, only for frames
at or before the end of the dropback (receiver_frames already ends there).
The expectation model is fit on ALL route runners (every position) in train weeks with
stats.cross_fit (GroupKFold by game OOF for train rows, full-train refit for test rows).
No receiver speed/acceleration, position, or identity is used as a feature.

Run from project root:  .venv/bin/python -m src.metrics.matchup_soe
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score  # noqa: E402

from src.common import config, io, stats  # noqa: E402

SLUG = "matchup_soe"
OUTDIR = config.OUT / SLUG
KEY = ["gameId", "playId", "nflId"]
PLAY = ["gameId", "playId"]

LANDMARK_T10 = [10, 15, 20, 25, 30]  # tenths of a second after snap
LANDMARKS = [f"t{t / 10:.1f}" for t in LANDMARK_T10] + ["throw"]
TE_ASSIGNMENTS = ["route", "chip_release"]
MIN_PLAYS_TOP = 40          # min distinct plays for top-5 / ranking-based sensitivity checks
MIN_PLAYS_DIST = 40         # min distinct plays for league-level player distributions
SEP_CAP_SENS = 10.0         # sensitivity: cap separation target at this many yards
EB_MIN_PLAYS_OWN_SD = 5     # below this, use pooled per-play SD for EB standard errors
Z = 1.96

NUM_FEATURES = [
    # alignment geometry at snap
    "x_rel_snap", "abs_lat", "receiver_num", "n_rec_side",
    # nearest defender at snap (offsets made side-relative: + = toward receiver's sideline)
    "sep_snap", "nd_dx_snap", "nd_dy_out_snap",
    # route depth / lateral position at landmark
    "x_rel", "lat_out", "dist_sideline",
    # time since snap
    "t",
    # play context
    "pff_playAction", "down", "yardsToGo", "red_zone",
]
CAT_FEATURES = ["alignment", "pff_passCoverage", "pff_passCoverageType", "landmark",
                "dropBackType", "offenseFormation"]
FEATURES = NUM_FEATURES + CAT_FEATURES
LANDMARK_LOC_FEATURES = ["x_rel", "lat_out", "dist_sideline"]

PARAM_GRID = [
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def build_landmark_rows() -> pd.DataFrame:
    p = io.plays()[PLAY + ["week", "is_train", "ball_y", "down", "yardsToGo", "red_zone",
                           "pff_playAction", "pff_passCoverage", "pff_passCoverageType",
                           "dropBackType", "offenseFormation", "possessionTeam", "targetNflId"]]
    r = io.routes()[KEY + ["x_rel_snap", "y_snap", "sep_snap", "nd_dx_snap", "nd_dy_snap", "t_end",
                           "end_event", "alignment", "officialPosition", "pff_blockType", "side",
                           "abs_lat", "receiver_num", "n_rec_side", "is_target", "is_catch"]]
    rf = io.receiver_frames(columns=KEY + ["frameId", "t", "x_rel", "y", "sep"])
    rf["t10"] = np.rint(rf["t"] * 10).astype(int)

    r = r.assign(t_end10=np.rint(r["t_end"] * 10).astype(int))
    rf = rf.merge(r[KEY + ["t_end10", "end_event"]], on=KEY, how="inner")
    fixed = rf[rf.t10.isin(LANDMARK_T10)].copy()
    fixed["landmark"] = "t" + (fixed.t10 / 10).map("{:.1f}".format)
    thr = rf[(rf.end_event == "throw") & (rf.t10 == rf.t_end10)].copy()
    thr["landmark"] = "throw"
    lm = pd.concat([fixed, thr], ignore_index=True).drop(columns=["t_end10", "end_event"])
    del rf, fixed, thr

    lm = lm.merge(r.drop(columns=["t_end10"]), on=KEY, how="left").merge(p, on=PLAY, how="left")
    side = lm["side"].replace(0, 1).astype(float)
    lm["nd_dy_out_snap"] = lm["nd_dy_snap"] * side
    lm["lat_out"] = (lm["y"] - lm["ball_y"]) * side
    lm["dist_sideline"] = np.minimum(lm["y"], config.FIELD_W - lm["y"])
    lm["n_rec_side"] = lm["n_rec_side"].astype(float)
    lm["red_zone"] = lm["red_zone"].astype(float)
    lm["is_train"] = lm["is_train"].astype(bool)
    for c in CAT_FEATURES:
        lm[c] = lm[c].astype("category")
    lm["landmark"] = lm["landmark"].cat.set_categories(LANDMARKS)
    return lm.reset_index(drop=True)


def model_frame(lm: pd.DataFrame) -> pd.DataFrame:
    """Integer-coded categoricals (NaN for missing) for HistGradientBoosting."""
    X = lm[NUM_FEATURES].astype(float).copy()
    for c in CAT_FEATURES:
        codes = lm[c].cat.codes.astype(float)
        X[c] = codes.where(codes >= 0)
    return X


def make_model(params, features):
    return HistGradientBoostingRegressor(
        loss="squared_error", early_stopping=False, random_state=config.SEED,
        categorical_features=[f in CAT_FEATURES for f in features], **params)


def reg_metrics(y, yhat):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, yhat))),
                mae=float(mean_absolute_error(y, yhat)), r2=float(r2_score(y, yhat)), n=int(len(y)))


def fit_expected(df, features, target, params):
    m = make_model(params, features)
    pred, full = stats.cross_fit(m, df, features, target)
    return pred, full


# --------------------------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------------------------
def play_level(rows: pd.DataFrame, by) -> pd.DataFrame:
    """Collapse landmark rows to one row per (by + play)."""
    g = rows.groupby(by + PLAY, observed=True)
    return g.agg(soe_sum=("soe", "sum"), n_lm=("soe", "size"), is_train=("is_train", "first")).reset_index()


def cluster_mean(pl: pd.DataFrame, by):
    """Landmark-row mean of SOE with play-clustered SE. pl from play_level()."""
    def f(d):
        n_rows, G = d.n_lm.sum(), len(d)
        m = d.soe_sum.sum() / n_rows
        if G >= 2:
            resid = d.soe_sum - m * d.n_lm
            se = np.sqrt(G / (G - 1) * np.sum(resid ** 2)) / n_rows
        else:
            se = np.nan
        return pd.Series({"value": m, "se": se, "n_rows": n_rows, "n_plays": G,
                          "value_playmean": (d.soe_sum / d.n_lm).mean()})
    if not by:
        return f(pl)
    out = pl.groupby(by, observed=True)[["soe_sum", "n_lm"]].apply(f).reset_index()
    return out


def player_table(rows: pd.DataFrame) -> pd.DataFrame:
    pl = play_level(rows, ["nflId"])
    t = cluster_mean(pl, ["nflId"])
    t["lo"], t["hi"] = t.value - Z * t.se, t.value + Z * t.se
    # EB: per-play SD (se * sqrt(G)); pooled fallback for tiny samples
    sd_play = t.se * np.sqrt(t.n_plays)
    pooled = np.nanmedian(sd_play[t.n_plays >= EB_MIN_PLAYS_OWN_SD])
    sd_play = sd_play.where(t.n_plays >= EB_MIN_PLAYS_OWN_SD, pooled)
    t["value_shrunk"] = stats.eb_shrink_mean(t.value, t.n_plays, sd_play)
    t["n_rows"] = t.n_rows.astype(int)
    t["n_plays"] = t.n_plays.astype(int)
    return t


def split_cols(rows, col, levels, prefix):
    """Per-player SOE (landmark-row mean) and n_plays within each level of a split column."""
    out = []
    for lv in levels:
        sub = rows[rows[col] == lv]
        if sub.empty:
            continue
        t = cluster_mean(play_level(sub, ["nflId"]), ["nflId"])[["nflId", "value", "n_plays"]]
        out.append(t.rename(columns={"value": f"soe_{prefix}{lv}", "n_plays": f"n_plays_{prefix}{lv}"})
                   .set_index("nflId"))
    return pd.concat(out, axis=1)


def league_rows(rows, by):
    t = cluster_mean(play_level(rows, by), by)
    if not by:
        t = t.to_frame().T
    t["lo"], t["hi"] = t.value - Z * t.se, t.value + Z * t.se
    if by:
        extra = rows.groupby(by, observed=True)[["sep", "exp_sep"]].mean().reset_index()
        t = t.merge(extra, on=by, how="left")
    else:
        t = t.reset_index(drop=True).assign(sep=rows.sep.mean(), exp_sep=rows.exp_sep.mean())
    return t


def recs(df, digits=4):
    return json.loads(df.round(digits).to_json(orient="records"))


def player_dist(pt, min_plays):
    d = pt[pt.n_plays >= min_plays]
    if d.empty:
        return {"n_players": 0}
    q = d.value.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n_players": int(len(d)), "mean_value": float(d.value.mean()), "sd_value": float(d.value.std()),
            "sd_value_shrunk": float(d.value_shrunk.std()),
            "p10": float(q[0.1]), "p25": float(q[0.25]), "p50": float(q[0.5]), "p75": float(q[0.75]),
            "p90": float(q[0.9]), "share_ci_above_0": float((d.lo > 0).mean()),
            "share_ci_below_0": float((d.hi < 0).mean())}


def target_check(thr: pd.DataFrame, label: str):
    """Target rate by SOE-at-throw quintile (and raw-separation quintile for comparison)."""
    d = thr.dropna(subset=["soe"]).copy()
    d["is_target"] = d.is_target.astype(int)
    out = {"sample": label, "n_rows": int(len(d)), "base_target_rate": float(d.is_target.mean())}
    for col in ["soe", "sep"]:
        d["q"] = pd.qcut(d[col], 5, labels=[1, 2, 3, 4, 5])
        g = d.groupby("q", observed=True).agg(n=("is_target", "size"), k=("is_target", "sum"),
                                              lo_val=(col, "min"), hi_val=(col, "max"),
                                              mean_val=(col, "mean")).reset_index()
        g["target_rate"] = g.k / g.n
        g["ci_lo"], g["ci_hi"] = stats.wilson_ci(g.k, g.n)
        out[f"by_{col}_quintile"] = recs(g.rename(columns={"q": "quintile"}))
    for col in ["soe", "sep", "exp_sep"]:
        out[f"auc_{col}"] = float(roc_auc_score(d.is_target, d[col]))
        te = d[~d.is_train]
        out[f"auc_{col}_test"] = float(roc_auc_score(te.is_target, te[col])) if te.is_target.nunique() > 1 else None
    return out


def within_play_check(thr_all: pd.DataFrame):
    """Among throw plays with a parsed target: how often is the target the receiver with the highest
    SOE vs the highest raw separation (all route runners on the play)."""
    d = thr_all.dropna(subset=["soe"])
    d = d[d.groupby(PLAY).is_target.transform("sum") == 1]
    top_soe = d.loc[d.groupby(PLAY).soe.idxmax()]
    top_sep = d.loc[d.groupby(PLAY).sep.idxmax()]
    n_rec = d.groupby(PLAY).size()
    return {"n_plays": int(len(top_soe)), "target_is_max_soe": float(top_soe.is_target.mean()),
            "target_is_max_sep": float(top_sep.is_target.mean()),
            "random_baseline": float((1 / n_rec).mean())}


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("building landmark rows")
    lm = build_landmark_rows()
    X = model_frame(lm)
    df = pd.concat([X, lm[["gameId", "is_train", "sep"]]], axis=1)
    df["sep_cap"] = df.sep.clip(upper=SEP_CAP_SENS)
    log(f"landmark rows: {len(lm):,} (train {lm.is_train.sum():,}); route-plays {lm[KEY].drop_duplicates().shape[0]:,}")

    tr, te = df.is_train.values, ~df.is_train.values

    # ---- tuning on train OOF only -----------------------------------------------------------
    tuning = []
    best = None
    for i, prm in enumerate(PARAM_GRID):
        pred, _ = fit_expected(df, FEATURES, "sep", prm)
        oof = reg_metrics(df.sep[tr], pred[tr])
        tuning.append({"params": prm, "train_oof": oof})
        log(f"grid {i}: train OOF RMSE {oof['rmse']:.4f} R2 {oof['r2']:.4f}")
        if best is None or oof["rmse"] < best[0]:
            best = (oof["rmse"], prm, pred)
    _, best_params, pred = best
    lm["exp_sep"] = pred.values
    lm["soe"] = lm.sep - lm.exp_sep

    # ---- model quality on TEST rows vs baseline (train mean by landmark) ---------------------
    base_map = lm[lm.is_train].groupby("landmark", observed=True).sep.mean()
    lm["base_sep"] = lm.landmark.map(base_map).astype(float)
    tl = lm[~lm.is_train]
    model_eval = {"test_overall": {"model": reg_metrics(tl.sep, tl.exp_sep),
                                   "baseline_train_mean_by_landmark": reg_metrics(tl.sep, tl.base_sep)},
                  "test_by_landmark": {}, "test_by_position": {}}
    for lmk, g in tl.groupby("landmark", observed=True):
        model_eval["test_by_landmark"][lmk] = {"model": reg_metrics(g.sep, g.exp_sep),
                                               "baseline": reg_metrics(g.sep, g.base_sep)}
    for pos in ["TE", "WR", "RB"]:
        g = tl[tl.officialPosition == pos]
        model_eval["test_by_position"][pos] = {"model": reg_metrics(g.sep, g.exp_sep),
                                               "baseline": reg_metrics(g.sep, g.base_sep),
                                               "mean_residual": float((g.sep - g.exp_sep).mean())}
    dec = pd.qcut(tl.exp_sep, 10, labels=False)
    model_eval["test_calibration_deciles"] = recs(
        tl.groupby(dec).agg(mean_pred=("exp_sep", "mean"), mean_obs=("sep", "mean"), n=("sep", "size"))
        .reset_index().rename(columns={"exp_sep": "decile"}))
    log(f"test model {model_eval['test_overall']['model']} | baseline "
        f"{model_eval['test_overall']['baseline_train_mean_by_landmark']}")

    # ---- sensitivity refits --------------------------------------------------------------------
    sens_models = {}
    pred_cap, _ = fit_expected(df, FEATURES, "sep_cap", best_params)
    lm["soe_cap"] = lm.sep.clip(upper=SEP_CAP_SENS) - pred_cap.values
    sens_models["sep_capped_10yd"] = {"test": reg_metrics(df.sep_cap[te], pred_cap[te])}
    f_presnap = [f for f in FEATURES if f not in LANDMARK_LOC_FEATURES]
    pred_ps, _ = fit_expected(df, f_presnap, "sep", best_params)
    lm["soe_presnap"] = lm.sep - pred_ps.values
    sens_models["no_landmark_location_features"] = {"features": f_presnap,
                                                    "test": reg_metrics(df.sep[te], pred_ps[te])}
    log("sensitivity refits done")

    # ---- TE rows -------------------------------------------------------------------------------
    tp = io.te_plays()[KEY + ["displayName", "assignment"]]
    lm = lm.merge(tp, on=KEY, how="left")
    te_rows = lm[(lm.officialPosition == "TE") & lm.assignment.isin(TE_ASSIGNMENTS)].copy()
    n_te_assign = int(tp.assignment.isin(TE_ASSIGNMENTS).sum())
    n_te_routes = int(te_rows[KEY].drop_duplicates().shape[0])

    pt = player_table(te_rows)
    names = tp.drop_duplicates("nflId").set_index("nflId").displayName
    team = (te_rows.groupby("nflId").possessionTeam
            .agg(lambda s: "/".join(s.value_counts().index.astype(str))))
    pt["displayName"] = pt.nflId.map(names)
    pt["team"] = pt.nflId.map(team)
    pt["n"] = pt.n_plays
    comp = pd.concat([
        te_rows.groupby("nflId")[["sep", "exp_sep"]].mean().rename(columns={"sep": "mean_sep", "exp_sep": "mean_exp_sep"}),
        split_cols(te_rows, "landmark", LANDMARKS, ""),
        split_cols(te_rows, "pff_passCoverageType", ["Man", "Zone"], ""),
        split_cols(te_rows, "alignment", ["inline", "detached"], ""),
        split_cols(te_rows, "assignment", TE_ASSIGNMENTS, ""),
    ], axis=1)
    comp.columns = [c.lower() if c.startswith(("soe_", "n_plays_")) else c for c in comp.columns]
    # route target rate for context
    tgt = te_rows.drop_duplicates(KEY).groupby("nflId").agg(targets=("is_target", "sum"))
    pt = pt.merge(comp, left_on="nflId", right_index=True, how="left").merge(tgt, left_on="nflId",
                                                                              right_index=True, how="left")
    pt["higher_is_better"] = True
    pt["meets_min_plays"] = pt.n_plays >= MIN_PLAYS_TOP
    lead = ["nflId", "displayName", "team", "n", "n_plays", "n_rows", "value", "value_shrunk", "lo", "hi",
            "se", "value_playmean", "mean_sep", "mean_exp_sep", "targets", "higher_is_better", "meets_min_plays"]
    pt = pt[lead + [c for c in pt.columns if c not in lead]].sort_values("value_shrunk", ascending=False)
    pt.to_csv(OUTDIR / "players.csv", index=False)

    # ---- TE plays.csv ------------------------------------------------------------------------------
    wide = te_rows.pivot_table(index=KEY, columns="landmark", values="soe", observed=True)
    wide.columns = [f"soe_{c}" for c in wide.columns]
    play_rows = (te_rows.groupby(KEY).agg(
        displayName=("displayName", "first"), team=("possessionTeam", "first"), week=("week", "first"),
        is_train=("is_train", "first"), assignment=("assignment", "first"), alignment=("alignment", "first"),
        pff_passCoverage=("pff_passCoverage", "first"), pff_passCoverageType=("pff_passCoverageType", "first"),
        n_landmarks=("soe", "size"), soe_play=("soe", "mean"), sep_mean=("sep", "mean"),
        exp_sep_mean=("exp_sep", "mean"), soe_cap_play=("soe_cap", "mean"), soe_presnap_play=("soe_presnap", "mean"),
        is_target=("is_target", "first"), is_catch=("is_catch", "first"))
        .join(wide).reset_index())
    play_rows.to_csv(OUTDIR / "plays.csv", index=False)
    te_rows[KEY + ["week", "is_train", "landmark", "t", "x_rel", "lat_out", "sep", "exp_sep", "soe",
                   "assignment", "alignment", "pff_passCoverageType"]].to_csv(OUTDIR / "te_landmarks.csv", index=False)

    # ---- league-level: TE splits ---------------------------------------------------------------------
    league_te = {
        "overall": recs(league_rows(te_rows, [])),
        "by_landmark": recs(league_rows(te_rows, ["landmark"])),
        "by_coverage_type": recs(league_rows(te_rows, ["pff_passCoverageType"])),
        "by_alignment": recs(league_rows(te_rows, ["alignment"])),
        "by_assignment": recs(league_rows(te_rows, ["assignment"])),
    }

    # ---- league-level: position comparison ---------------------------------------------------------------
    pos_rows = {"TE": te_rows, "WR": lm[lm.officialPosition == "WR"], "RB": lm[lm.officialPosition == "RB"],
                "FB": lm[lm.officialPosition == "FB"]}
    pos_tables = {k: player_table(v) for k, v in pos_rows.items()}
    position_compare = {}
    for k, v in pos_rows.items():
        position_compare[k] = {
            "row_level": recs(league_rows(v, []))[0],
            "by_landmark": recs(league_rows(v, ["landmark"])),
            "by_coverage_type": recs(league_rows(v, ["pff_passCoverageType"])),
            "player_distribution_min_plays": MIN_PLAYS_DIST,
            "player_distribution": player_dist(pos_tables[k], MIN_PLAYS_DIST),
        }
    pos_dist_test = {}
    # Kruskal-Wallis on player-level values (min plays) TE vs WR vs RB
    vals = [pos_tables[k].query("n_plays >= @MIN_PLAYS_DIST").value for k in ["TE", "WR", "RB"]]
    if all(len(v) > 2 for v in vals):
        kw = sps.kruskal(*vals)
        pos_dist_test["kruskal_TE_WR_RB_player_values"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}
        mw = sps.mannwhitneyu(vals[0], vals[1])
        pos_dist_test["mannwhitney_TE_vs_WR"] = {"U": float(mw.statistic), "p": float(mw.pvalue)}
    league_pos = league_rows(pd.concat([v.assign(pos=k) for k, v in pos_rows.items()]), ["pos"])

    # ---- reliability ----------------------------------------------------------------------------------------
    def reliab(rows, min_n):
        pl = play_level(rows, ["nflId"])
        pl["soe_play"] = pl.soe_sum / pl.n_lm
        rho, n = stats.split_half_reliability(pl, "nflId", "soe_play", min_n=min_n)
        return {"spearman": None if pd.isna(rho) else float(rho), "n_players": int(n), "min_plays_each_half": min_n}
    reliability = {"TE": [reliab(te_rows, 10), reliab(te_rows, 20)],
                   "WR": [reliab(pos_rows["WR"], 10), reliab(pos_rows["WR"], 20)],
                   "RB": [reliab(pos_rows["RB"], 10)],
                   "TE_raw_separation_for_reference": None}
    raw = te_rows.assign(soe=te_rows.sep)
    reliability["TE_raw_separation_for_reference"] = reliab(raw, 10)
    log(f"TE split-half reliability: {reliability['TE']}")

    # ---- target check ---------------------------------------------------------------------------------------
    thr_all = lm[(lm.landmark == "throw") & lm.targetNflId.notna()]
    thr_te = thr_all[thr_all.index.isin(te_rows.index)]
    target_checks = {"TE": target_check(thr_te, "TE route/chip_release, throw landmark"),
                     "all_route_runners": target_check(thr_all, "all route runners, throw landmark"),
                     "within_play_all_route_runners": within_play_check(thr_all)}

    # ---- sensitivity: ranking stability among TEs >= MIN_PLAYS_TOP -----------------------------------------
    main_rank = pt.set_index("nflId")
    elig = main_rank.index[main_rank.n_plays >= MIN_PLAYS_TOP]

    def variant(rows, col="soe"):
        v = player_table(rows.assign(soe=rows[col])).set_index("nflId")
        common = elig.intersection(v.index)
        rho = sps.spearmanr(main_rank.loc[common, "value"], v.loc[common, "value"]).correlation
        return {"spearman_vs_main_value": float(rho), "n_players": int(len(common)),
                "league_mean_soe": float(rows[col].mean()),
                "split_half_reliability": reliab(rows.assign(soe=rows[col]), 10),
                "top5": v.loc[common].sort_values("value_shrunk", ascending=False).head(5).index.map(names).tolist()}
    sensitivity = {
        "route_only_no_chip": variant(te_rows[te_rows.assignment == "route"]),
        "fixed_time_landmarks_only": variant(te_rows[te_rows.landmark != "throw"]),
        "throw_landmark_only": variant(te_rows[te_rows.landmark == "throw"]),
        "sep_capped_10yd_refit": {**variant(te_rows, "soe_cap"), **sens_models["sep_capped_10yd"]},
        "presnap_context_only_refit": {**variant(te_rows, "soe_presnap"),
                                       "test": sens_models["no_landmark_location_features"]["test"]},
        "play_weighted_mean": {"spearman_vs_main_value": float(sps.spearmanr(
            main_rank.loc[elig, "value"], main_rank.loc[elig, "value_playmean"]).correlation),
            "n_players": int(len(elig))},
        "shrunk_vs_raw_value": {"spearman": float(sps.spearmanr(
            main_rank.loc[elig, "value"], main_rank.loc[elig, "value_shrunk"]).correlation)},
    }

    top5 = pt[pt.meets_min_plays].head(5)[["nflId", "displayName", "team", "n_plays", "n_rows", "value",
                                            "value_shrunk", "lo", "hi"]]
    log("top-5:\n" + top5.to_string(index=False))

    summary = {
        "metric": "Matchup Separation Over Expected (SOE)",
        "slug": SLUG,
        "higher_is_better": True,
        "definition": (
            "For each route runner and landmark (t = 1.0/1.5/2.0/2.5/3.0 s after snap, only if at or before "
            "the end of the dropback, plus the throw frame on plays ending in a throw), SOE = observed "
            "nearest-defender separation (receiver_frames.sep) - expected separation from a "
            "HistGradientBoostingRegressor fit on all route runners (all positions) in train weeks, cross-fit "
            "by game (OOF for weeks 1-6, full-train refit for weeks 7-8). Player value = mean SOE over all "
            "landmark rows on the TE's route + chip_release plays; CI is play-clustered (cluster-robust SE "
            "of the row mean); value_shrunk = normal-normal EB toward the TE grand mean with n = plays and "
            "per-play SD = cluster SE * sqrt(n_plays)."),
        "parameters": {
            "landmarks": LANDMARKS, "landmark_rule": "t rounded to 0.1 s; throw = last receiver_frames frame "
            "on plays with end_event == 'throw' (coincides with a fixed landmark when the throw is at that t; both kept)",
            "te_assignments": TE_ASSIGNMENTS, "min_plays_top5": MIN_PLAYS_TOP,
            "min_plays_player_distribution": MIN_PLAYS_DIST, "sep_cap_sensitivity": SEP_CAP_SENS,
            "eb_min_plays_own_sd": EB_MIN_PLAYS_OWN_SD,
            "features_numeric": NUM_FEATURES, "features_categorical": CAT_FEATURES,
            "feature_notes": {
                "nd_dy_out_snap": "nd_dy_snap * side (+ = nearest defender toward receiver's sideline)",
                "lat_out": "(y - ball_y) * side at landmark (+ = outside of ball on alignment side)",
                "dist_sideline": "min(y, 53.3 - y) at landmark",
                "x_rel": "route depth past LOS at landmark",
                "excluded": "receiver speed/accel/velocity, officialPosition, player identity, nearest-defender "
                            "offset at the landmark (outcome)"},
            "param_grid": PARAM_GRID, "chosen_params": best_params, "tuning_metric": "train OOF RMSE",
            "train_weeks": config.TRAIN_WEEKS, "test_weeks": config.TEST_WEEKS, "n_folds": config.N_FOLDS,
        },
        "sample_sizes": {
            "landmark_rows_all_positions": int(len(lm)), "landmark_rows_train": int(lm.is_train.sum()),
            "landmark_rows_test": int((~lm.is_train).sum()),
            "route_plays_all_positions": int(lm[KEY].drop_duplicates().shape[0]),
            "landmark_rows_by_landmark": {k: int(v) for k, v in lm.landmark.value_counts().items()},
            "te_route_chip_assignments_in_te_plays": n_te_assign,
            "te_route_chip_plays_with_landmarks": n_te_routes,
            "te_landmark_rows": int(len(te_rows)), "n_te_players": int(len(pt)),
            "n_te_players_min_plays": int(pt.meets_min_plays.sum()),
        },
        "model_tuning": tuning,
        "model_eval": model_eval,
        "league_te": league_te,
        "position_comparison": {"by_position_row_level": recs(league_pos), "per_position": position_compare,
                                "tests": pos_dist_test},
        "split_half_reliability": reliability,
        "target_check": target_checks,
        "sensitivity": sensitivity,
        "top5_min_plays": recs(top5),
        "caveats": [
            "Landmark-location features (route depth, lateral position, distance to sideline at the landmark) "
            "are partly receiver-controlled; they are included per spec so SOE compares receivers at the same "
            "spot. The pre-snap-only refit in sensitivity shows how much this matters.",
            "Separation = distance to nearest defender only; it ignores leverage, defender closing speed and "
            "the ball; tracking stops at the throw so no catch-point separation exists.",
            "Fixed landmarks after the end of the dropback are dropped, so long-developing routes are only "
            "observed on long dropbacks (selection toward quick-game at late landmarks).",
            "Chip-release plays include pre-release frames where the TE is engaged with a rusher; the model "
            "has no assignment feature, so chip plays can carry negative early SOE (see by_assignment).",
            "Expected model is position-blind; position-level mean SOE differences reflect what context "
            "features do not explain (e.g., TEs drawing linebackers, WRs on the boundary).",
            "Coverage labels (PFF) are play-level, not receiver-level; man vs zone splits are play-level.",
            "Throw landmark target check is descriptive: QBs target open receivers, and separation at the "
            "throw frame partly reflects the QB's read; no causal claim.",
            f"{n_te_assign - n_te_routes} TE route/chip assignments in te_plays have no routes/receiver_frames row "
            "and are excluded.",
            "8 weeks only; split-half reliability compares weeks 1-6 vs 7-8 (unequal halves). TE SOE "
            f"split-half Spearman = {reliability['TE'][0]['spearman']:.3f} (n={reliability['TE'][0]['n_players']}); "
            "if near zero, player SOE over this window is mostly noise: treat rankings as descriptive and "
            "rely on value_shrunk + CIs.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUTDIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"done in {summary['runtime_sec']} s -> {OUTDIR}")


if __name__ == "__main__":
    main()

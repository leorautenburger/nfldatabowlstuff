"""Coverage Gravity: extra defensive attention drawn by a TE vs comparable aligned receivers.

Pipeline
  1. For every route runner (ALL positions) compute attention measures from tracking at
     three landmarks: snap (t=0), t* = min(2.0 s, end of dropback), and the throw /
     end-of-dropback frame.
       - n_def_3 / n_def_5   defenders within 3 / 5 yd
       - att                 sum_defenders exp(-d / 3)     (smooth attention index)
       - negsep              -(nearest-defender distance)  (higher = more attention)
       - bracket             >=2 defenders within 4 yd on opposite sides
                             (inside/outside OR under/over, each offset >= 0.5 yd)
       - safety_pull         d(receiver, nearest safety) at snap - same at t*  (safeties
                             = pff_positionLinedUp in FS/FSL/FSR/SS/SSL/SSR)
       - saf_depth_chg       depth change (x_rel) snap -> t* of the safety nearest at t*
  2. Expected attention: HistGradientBoosting on context only (alignment, formation,
     coverage, situation, receiver location at landmark), cross-fit on train weeks,
     no player identity / position.
  3. Coverage Gravity per TE = mean(att_obs - att_exp at t*) on NON-targeted TE routes.

Run: .venv/bin/python -m src.metrics.coverage_gravity
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import time

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from src.common import io, stats
from src.common.config import FIELD_W, FPS, OUT, RED_ZONE_YARDS, SEED, TEST_WEEKS, TRAIN_WEEKS

SLUG = "coverage_gravity"
OUT_DIR = OUT / SLUG

# ---------------------------------------------------------------- parameters
T_STAR = 2.0                 # landmark seconds after snap (capped at end of dropback)
DECAY = 3.0                  # yards, attention index scale exp(-d/DECAY)
DECAY_SENS = [2.0, 5.0]      # sensitivity scales
BRACKET_R = 4.0              # yards radius for bracket defenders
BRACKET_MIN_OFF = 0.5        # yards min offset for "inside/outside" or "under/over"
SAFETY_POS = ["FS", "FSL", "FSR", "SS", "SSL", "SSR"]
MIN_TOP = 40                 # min non-target routes for leaderboard
MIN_SPLIT_HALF = 10

CAT = ["offenseFormation", "personnelO", "pff_passCoverage", "pff_passCoverageType"]
CTX = ["x_rel_snap", "abs_lat", "receiver_num", "n_rec_side", "side", "offenseFormation",
       "personnelO", "defendersInBox", "pff_passCoverage", "pff_passCoverageType",
       "red_zone", "yards_to_goal", "down", "yardsToGo"]
LOC = ["x_rel", "lat", "sideline", "t"]  # suffixed with _star / _end

HGB_KW = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, random_state=SEED)


# ---------------------------------------------------------------- tracking features
def _landmark_frames():
    """Receiver positions at snap / t* / end for every route runner."""
    rf = io.receiver_frames(columns=["gameId", "playId", "nflId", "t", "x", "y", "x_rel"])
    rf["tick"] = np.rint(rf["t"] * FPS).astype(int)
    end = rf.groupby(["gameId", "playId", "nflId"])["tick"].transform("max")
    star = np.minimum(int(round(T_STAR * FPS)), end)
    parts = []
    for name, mask in [("snap", rf["tick"] == 0), ("star", rf["tick"] == star),
                       ("end", rf["tick"] == end)]:
        parts.append(rf[mask].assign(lm=name))
    lm = pd.concat(parts, ignore_index=True).drop(columns="t")
    lm = lm.rename(columns={"x": "xr", "y": "yr", "x_rel": "xrel_r"})
    return lm


def _pair_features(lm_g, dfn, ball_y):
    """Merge receivers with defenders at the same tick; aggregate attention measures."""
    pr = lm_g.merge(dfn, on=["playId", "tick"], how="inner")
    dx = pr["xd"].to_numpy() - pr["xr"].to_numpy()
    dy = pr["yd"].to_numpy() - pr["yr"].to_numpy()
    d = np.hypot(dx, dy)
    by = pr["playId"].map(ball_y).to_numpy()
    insign = np.sign(by - pr["yr"].to_numpy())
    insign[insign == 0] = 1.0
    lat_in = dy * insign  # >0 = defender inside (toward ball), <0 = outside
    w4 = d <= BRACKET_R
    pr = pr.assign(
        d=d,
        c3=(d <= 3).astype(np.int16), c5=(d <= 5).astype(np.int16), c4=w4.astype(np.int16),
        att=np.exp(-d / DECAY), **{f"att_{int(k)}": np.exp(-d / k) for k in DECAY_SENS},
        has_in=(w4 & (lat_in >= BRACKET_MIN_OFF)), has_out=(w4 & (lat_in <= -BRACKET_MIN_OFF)),
        has_over=(w4 & (dx >= BRACKET_MIN_OFF)), has_under=(w4 & (dx <= -BRACKET_MIN_OFF)),
        d_saf=np.where(pr["is_saf"].to_numpy(), d, np.nan),
    )
    keys = ["playId", "nflId", "lm"]
    agg = pr.groupby(keys).agg(
        n_def_3=("c3", "sum"), n_def_5=("c5", "sum"), n_def_4=("c4", "sum"),
        att=("att", "sum"), **{f"att_{int(k)}": (f"att_{int(k)}", "sum") for k in DECAY_SENS},
        sep=("d", "min"), has_in=("has_in", "any"), has_out=("has_out", "any"),
        has_over=("has_over", "any"), has_under=("has_under", "any"), d_saf=("d_saf", "min"),
        n_def_tracked=("d", "size"))
    agg["bracket"] = ((agg["n_def_4"] >= 2) & ((agg["has_in"] & agg["has_out"]) |
                                               (agg["has_over"] & agg["has_under"]))).astype(int)
    agg["negsep"] = -agg["sep"]
    # nearest safety at t*: its depth change from snap
    sp = pr[pr["is_saf"] & (pr["lm"] == "star")]
    if len(sp):
        idx = sp.groupby(["playId", "nflId"])["d"].idxmin()
        ns = sp.loc[idx, ["playId", "nflId", "def_id", "xreld"]].rename(
            columns={"xreld": "saf_xrel_star"})
        snap_pos = dfn[dfn["tick"] == 0][["playId", "def_id", "xreld"]].rename(
            columns={"xreld": "saf_xrel_snap"})
        ns = ns.merge(snap_pos, on=["playId", "def_id"], how="left")
        ns["saf_depth_chg"] = ns["saf_xrel_star"] - ns["saf_xrel_snap"]
        ns = ns[["playId", "nflId", "saf_depth_chg"]]
    else:
        ns = pd.DataFrame(columns=["playId", "nflId", "saf_depth_chg"])
    return agg.reset_index(), ns


def compute_attention(plays):
    t0 = time.time()
    lm = _landmark_frames()
    pp = io.player_plays()[["gameId", "playId", "nflId", "pff_positionLinedUp"]]
    saf = pp[pp["pff_positionLinedUp"].isin(SAFETY_POS)][["gameId", "playId", "nflId"]]
    saf = saf.rename(columns={"nflId": "def_id"}).assign(is_saf=True)
    ball_y_all = plays.set_index(["gameId", "playId"])["ball_y"]
    out, out_saf = [], []
    for i, g in enumerate(sorted(lm["gameId"].unique())):
        lm_g = lm[lm["gameId"] == g]
        need = lm_g[["playId", "tick"]].drop_duplicates()
        tr = io.tracking(columns=["gameId", "playId", "nflId", "t", "is_def", "x", "y", "x_rel"],
                         filters=[("gameId", "==", int(g)), ("is_def", "==", True)])
        tr["tick"] = np.rint(tr["t"] * FPS).astype(int)
        tr = tr.merge(need, on=["playId", "tick"], how="inner")
        tr = tr.rename(columns={"nflId": "def_id", "x": "xd", "y": "yd", "x_rel": "xreld"})
        tr = tr.merge(saf[saf["gameId"] == g].drop(columns="gameId"), on=["playId", "def_id"],
                      how="left")
        tr["is_saf"] = tr["is_saf"].notna()
        tr = tr[["playId", "tick", "def_id", "xd", "yd", "xreld", "is_saf"]]
        ball_y = ball_y_all.loc[g]
        agg, ns = _pair_features(lm_g, tr, ball_y)
        out.append(agg.assign(gameId=g))
        out_saf.append(ns.assign(gameId=g))
        if i % 20 == 0:
            print(f"  game {i + 1}: {time.time() - t0:.0f}s", flush=True)
    agg = pd.concat(out, ignore_index=True)
    ns = pd.concat(out_saf, ignore_index=True)
    # wide: one row per route runner-play
    cols = ["n_def_3", "n_def_5", "att", "att_2", "att_5", "sep", "negsep", "bracket", "d_saf",
            "n_def_tracked"]
    key3 = ["gameId", "playId", "nflId"]
    wide = agg.set_index(key3 + ["lm"])[cols].unstack("lm")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    loc = lm.set_index(key3 + ["lm"])[["xrel_r", "yr", "tick"]].unstack("lm")
    loc.columns = [f"{a}_{b}" for a, b in loc.columns]
    wide = wide.merge(loc.reset_index(), on=["gameId", "playId", "nflId"], how="left")
    wide = wide.merge(ns, on=["gameId", "playId", "nflId"], how="left")
    wide["safety_pull_star"] = wide["d_saf_snap"] - wide["d_saf_star"]
    print(f"  attention features done in {time.time() - t0:.0f}s", flush=True)
    return wide


# ---------------------------------------------------------------- modelling helpers
def _reg_metrics(y, p, base):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.full_like(y, base)
    return {"n": int(len(y)),
            "model": {"rmse": float(np.sqrt(mean_squared_error(y, p))),
                      "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))},
            "baseline_train_mean": {"rmse": float(np.sqrt(mean_squared_error(y, b))),
                                    "mae": float(mean_absolute_error(y, b)),
                                    "r2": float(r2_score(y, b))}}


def _clf_metrics(y, p, base):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    b = np.full(len(y), base)
    cal = pd.DataFrame({"p": p, "y": y})
    cal["bin"] = pd.qcut(cal["p"], 5, labels=False, duplicates="drop")
    cal = cal.groupby("bin").agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
    return {"n": int(len(y)), "base_rate_test": float(y.mean()),
            "model": {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
                      "brier": float(brier_score_loss(y, p))},
            "baseline_train_rate": {"auc": 0.5, "log_loss": float(log_loss(y, b)),
                                    "brier": float(brier_score_loss(y, b))},
            "calibration_5bin": cal.round(4).reset_index().to_dict(orient="records")}


def fit_expected(df, target, features, kind="reg"):
    sub = df[df[target].notna()]
    if kind == "reg":
        m = HistGradientBoostingRegressor(categorical_features=[f for f in features if f in CAT],
                                          **HGB_KW)
        pred, _ = stats.cross_fit(m, sub, features, target)
    else:
        m = HistGradientBoostingClassifier(categorical_features=[f for f in features if f in CAT],
                                           **HGB_KW)
        pred, _ = stats.cross_fit(m, sub, features, target, proba=True)
    tr, te = sub["is_train"], ~sub["is_train"]
    base = sub.loc[tr, target].mean()
    met = (_reg_metrics if kind == "reg" else _clf_metrics)(sub.loc[te, target], pred[te], base)
    te_te = te & (sub["officialPosition"] == "TE")
    met["te_rows_only"] = (_reg_metrics if kind == "reg" else _clf_metrics)(
        sub.loc[te_te, target], pred[te_te], base) if te_te.sum() > 50 else None
    out = pd.Series(np.nan, index=df.index)
    out.loc[sub.index] = pred
    return out, met


# ---------------------------------------------------------------- player aggregation
def _player_summary(d, col, prefix, shrink=True, pooled_sd=None):
    g = d.groupby("nflId")[col].agg(["mean", "count", "std"])
    sd = g["std"].fillna(pooled_sd if pooled_sd is not None else d[col].std())
    res = pd.DataFrame({f"{prefix}": g["mean"], f"n_{prefix}": g["count"]})
    se = sd / np.sqrt(g["count"])
    res[f"{prefix}_lo"] = g["mean"] - 1.96 * se
    res[f"{prefix}_hi"] = g["mean"] + 1.96 * se
    res.loc[g["count"] < 2, [f"{prefix}_lo", f"{prefix}_hi"]] = np.nan
    if shrink:
        # pooled within-player SD for every player: per-player sample SDs from 2-5 routes are
        # unreliable and let tiny-n players escape shrinkage in stats.eb_shrink_mean
        sd_pool = pd.Series(pooled_sd if pooled_sd is not None else d[col].std(), index=g.index)
        res[f"{prefix}_shrunk"] = stats.eb_shrink_mean(g["mean"], g["count"], sd_pool)
    return res


def _corr(x, y):
    m = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = len(x)
    if n < 5:
        return {"n": int(n)}
    r = sps.pearsonr(x, y)
    z, se = np.arctanh(r.statistic), 1 / np.sqrt(n - 3)
    return {"n": int(n), "pearson": float(r.statistic), "pearson_p": float(r.pvalue),
            "pearson_ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
            "spearman": float(sps.spearmanr(x, y).correlation)}


# ---------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plays = io.plays()
    routes = io.routes()
    tep = io.te_plays()[["gameId", "playId", "nflId", "assignment"]]
    names = io.player_plays()[["nflId", "displayName"]].drop_duplicates("nflId")

    print("computing attention features from tracking ...", flush=True)
    att = compute_attention(plays)

    pc = plays[["gameId", "playId", "week", "possessionTeam", "offenseFormation", "personnelO",
                "defendersInBox", "pff_passCoverage", "pff_passCoverageType", "red_zone",
                "yards_to_goal", "down", "yardsToGo", "ball_y", "end_event"]]
    # routes' *_end attention columns are recomputed here from tracking at the same frame;
    # keep the shared versions under a suffix for a consistency check
    dup = [c for c in att.columns if c in routes.columns and c not in ("gameId", "playId", "nflId")]
    df = routes.drop(columns=["end_event"]).rename(columns={c: f"{c}_routes" for c in dup})
    df = df.merge(pc, on=["gameId", "playId"], how="left")
    df = df.merge(att, on=["gameId", "playId", "nflId"], how="inner")
    consistency = {f"{c}_max_abs_diff_vs_routes": float((df[c] - df[f"{c}_routes"]).abs().max())
                   for c in dup}
    print("consistency vs routes table:", consistency, flush=True)
    df = df.merge(tep, on=["gameId", "playId", "nflId"], how="left")
    df = df.reset_index(drop=True)
    for c in CAT:  # integer codes; NaN stays missing for HGB
        df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    df["red_zone"] = df["red_zone"].astype(float)
    df["is_train"] = df["is_train"].astype(bool)
    for lmk in ["star", "end"]:
        df[f"x_rel_{lmk}"] = df[f"xrel_r_{lmk}"]
        df[f"lat_{lmk}"] = (df[f"yr_{lmk}"] - df["ball_y"]).abs()
        df[f"sideline_{lmk}"] = np.minimum(df[f"yr_{lmk}"], FIELD_W - df[f"yr_{lmk}"])
        df[f"t_{lmk}"] = df[f"tick_{lmk}"] / FPS
    feat_star = CTX + [f"{c}_star" for c in LOC]
    feat_end = CTX + [f"{c}_end" for c in LOC]
    df["is_te"] = df["officialPosition"] == "TE"
    print(f"route runners: {len(df)} ({df['is_te'].sum()} TE)", flush=True)

    # ---- expected models
    specs = [  # (target, features, kind, tag)
        ("att_star", feat_star, "reg", "primary"),
        ("negsep_star", feat_star, "reg", "component"),
        ("bracket_star", feat_star, "clf", "component"),
        ("n_def_5_star", feat_star, "reg", "component"),
        ("n_def_3_star", feat_star, "reg", "component"),
        ("safety_pull_star", feat_star, "reg", "component"),
        ("saf_depth_chg", feat_star, "reg", "component"),
        ("att_end", feat_end, "reg", "throw_frame"),
        ("negsep_end", feat_end, "reg", "throw_frame"),
        ("bracket_end", feat_end, "clf", "throw_frame"),
        ("n_def_5_end", feat_end, "reg", "throw_frame"),
        ("att_2_star", feat_star, "reg", "sensitivity_decay2"),
        ("att_5_star", feat_star, "reg", "sensitivity_decay5"),
    ]
    model_metrics = {}
    for tgt, feats, kind, tag in specs:
        t0 = time.time()
        df[f"{tgt}_exp"], met = fit_expected(df, tgt, feats, kind)
        df[f"{tgt}_oe"] = df[tgt] - df[f"{tgt}_exp"]
        met["tag"] = tag
        model_metrics[tgt] = met
        key = "r2" if kind == "reg" else "auc"
        print(f"  {tgt:18s} test {key}={met['model'][key]:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    # sensitivity: context-only expectation (no receiver location at t*)
    df["att_star_ctxonly_exp"], met = fit_expected(df, "att_star", CTX + ["t_star"], "reg")
    df["att_star_ctxonly_oe"] = df["att_star"] - df["att_star_ctxonly_exp"]
    met["tag"] = "sensitivity_no_location"
    model_metrics["att_star_ctxonly"] = met

    # sensitivity: play-relative gravity (TE residual minus mean residual of the other route
    # runners on the same play) to net out play-wide coverage tightness
    grp = df.groupby(["gameId", "playId"])["att_star_oe"]
    s_, n_ = grp.transform("sum"), grp.transform("count")
    df["others_att_oe"] = np.where(n_ > 1, (s_ - df["att_star_oe"]) / (n_ - 1), np.nan)
    df["gravity_rel"] = df["att_star_oe"] - df["others_att_oe"]

    # ---- TE play-level table
    te = df[df["is_te"]].copy()
    te["inline_detached"] = np.where(te["alignment"].isin(["inline", "wing"]), "inline",
                                     np.where(te["alignment"] == "detached", "detached",
                                              te["alignment"]))
    te["gravity"] = te["att_star_oe"]
    nt = te[~te["is_target"]]
    dev = nt["gravity"] - nt.groupby("nflId")["gravity"].transform("mean")
    pooled = float(np.sqrt((dev ** 2).sum() / (len(nt) - nt["nflId"].nunique())))

    # ---- player table
    P = _player_summary(nt, "gravity", "value", pooled_sd=pooled)
    P = P.rename(columns={"n_value": "n", "value_lo": "lo", "value_hi": "hi"})
    A = _player_summary(te, "gravity", "gravity_all", pooled_sd=pooled)
    RZ = _player_summary(nt[nt["red_zone"] == 1], "gravity", "gravity_rz", pooled_sd=pooled)
    IN = _player_summary(nt[nt["inline_detached"] == "inline"], "gravity", "gravity_inline",
                         pooled_sd=pooled)
    DE = _player_summary(nt[nt["inline_detached"] == "detached"], "gravity", "gravity_detached",
                         pooled_sd=pooled)
    comp = nt.groupby("nflId").agg(
        att_obs=("att_star", "mean"), att_exp=("att_star_exp", "mean"),
        bracket_rate=("bracket_star", "mean"), bracket_oe=("bracket_star_oe", "mean"),
        safety_pull_oe=("safety_pull_star_oe", "mean"), n_safety=("safety_pull_star_oe", "count"),
        saf_depth_chg_oe=("saf_depth_chg_oe", "mean"), negsep_oe=("negsep_star_oe", "mean"),
        n_def_5_oe=("n_def_5_star_oe", "mean"), n_def_3_oe=("n_def_3_star_oe", "mean"),
        gravity_throw=("att_end_oe", "mean"), bracket_throw_oe=("bracket_end_oe", "mean"),
        negsep_throw_oe=("negsep_end_oe", "mean"),
        gravity_decay2=("att_2_star_oe", "mean"), gravity_decay5=("att_5_star_oe", "mean"),
        gravity_ctxonly=("att_star_ctxonly_oe", "mean"),
        gravity_rel=("gravity_rel", "mean"),
        share_inline=("inline_detached", lambda s: (s == "inline").mean()))
    tgt_rate = te.groupby("nflId")["is_target"].mean().rename("target_rate")
    team = te.groupby("nflId")["possessionTeam"].agg(lambda s: s.mode().iat[0]).rename("team")
    P = (P.join([A, RZ, IN, DE, comp, tgt_rate, team], how="left").reset_index()
         .merge(names, on="nflId", how="left"))
    P["higher_is_better"] = True
    P["meets_min_n"] = P["n"] >= MIN_TOP
    lead = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi",
            "higher_is_better", "meets_min_n"]
    P = P[lead + [c for c in P.columns if c not in lead]]
    P = P.sort_values("value_shrunk", ascending=False)
    P.to_csv(OUT_DIR / "players.csv", index=False)

    # ---- plays.csv (TE route-level)
    keep = ["gameId", "playId", "nflId", "week", "is_train", "possessionTeam", "alignment",
            "inline_detached", "assignment", "receiver_num", "n_rec_side", "red_zone",
            "is_target", "is_catch", "end_event", "t_star", "t_end", "x_rel_star", "lat_star",
            "att_snap", "att_star", "att_star_exp", "gravity", "att_end", "att_end_exp",
            "att_end_oe", "n_def_3_star", "n_def_5_star", "n_def_5_star_exp", "sep_star",
            "negsep_star_exp", "bracket_star", "bracket_star_exp", "bracket_end",
            "bracket_end_exp", "d_saf_snap", "d_saf_star", "safety_pull_star",
            "safety_pull_star_exp", "saf_depth_chg", "saf_depth_chg_exp", "att_star_ctxonly_oe",
            "others_att_oe", "gravity_rel"]
    te[keep].merge(names, on="nflId", how="left").to_csv(OUT_DIR / "plays.csv", index=False)

    # ---- reliability
    rel = {}
    for lab, d_, c in [("gravity_nontarget", nt, "gravity"), ("gravity_all_routes", te, "gravity"),
                       ("gravity_throw_nontarget", nt, "att_end_oe"),
                       ("bracket_oe_nontarget", nt, "bracket_star_oe"),
                       ("safety_pull_oe_nontarget", nt, "safety_pull_star_oe"),
                       ("raw_att_star_nontarget", nt, "att_star"),
                       ("gravity_ctxonly_nontarget", nt, "att_star_ctxonly_oe"),
                       ("gravity_play_relative_nontarget", nt, "gravity_rel"),
                       ("negsep_oe_nontarget", nt, "negsep_star_oe"),
                       ("n_def_5_oe_nontarget", nt, "n_def_5_star_oe")]:
        r, n = stats.split_half_reliability(d_, "nflId", c, min_n=MIN_SPLIT_HALF)
        rel[lab] = {"spearman": None if pd.isna(r) else float(r), "n_players": int(n)}
    # odd/even game split (within-season, larger halves) as a supplementary check
    nt2 = nt.assign(half=(nt["gameId"].rank(method="dense") % 2 == 0))
    r, n = stats.split_half_reliability(nt2, "nflId", "gravity", half_col="half",
                                        min_n=MIN_SPLIT_HALF)
    rel["gravity_nontarget_odd_even_games"] = {"spearman": float(r), "n_players": int(n)}

    # ---- sensitivity: rank agreement of variants among qualified TEs
    q = P[P["meets_min_n"]]
    sens = {}
    for c in ["gravity_all", "gravity_throw", "gravity_decay2", "gravity_decay5",
              "gravity_ctxonly", "gravity_rel", "bracket_oe", "safety_pull_oe", "negsep_oe", "n_def_5_oe",
              "gravity_inline", "gravity_detached", "gravity_rz"]:
        m = q[c].notna()
        sens[c] = {"spearman_vs_value": float(sps.spearmanr(q.loc[m, "value"],
                                                             q.loc[m, c]).correlation),
                   "n_players": int(m.sum())}
    # TE vs other positions, overall residual check (model is position-blind)
    pos_resid = df.groupby("officialPosition")["att_star_oe"].agg(["mean", "count"]).round(4)
    pos_resid = pos_resid[pos_resid["count"] >= 100].to_dict(orient="index")
    align_resid = te.groupby(["inline_detached", "is_target"])["gravity"].agg(
        ["mean", "count"]).round(4).reset_index().to_dict(orient="records")

    # ---- sanity check: TE gravity vs other receivers' separation on same play
    df["sep_end_oe"] = -df["negsep_end_oe"]
    oth = df[~df["is_te"]].groupby(["gameId", "playId"]).agg(
        max_sep_end_other=("sep_end", "max"), mean_sep_oe_other=("sep_end_oe", "mean"),
        max_sep_oe_other=("sep_end_oe", "max"), n_other=("sep_end", "size"))
    tp = te.groupby(["gameId", "playId"]).agg(
        te_gravity=("gravity", "mean"), te_targeted=("is_target", "any"),
        te_gravity_throw=("att_end_oe", "mean"), team=("possessionTeam", "first"))
    sp = tp.join(oth, how="inner").reset_index()
    sp = sp.merge(plays[["gameId", "playId", "end_event"]], on=["gameId", "playId"])
    sp = sp[sp["end_event"] == "throw"]
    # remove the mechanical "more receivers -> higher max" effect
    sp["max_sep_other_adj"] = sp["max_sep_end_other"] - sp.groupby("n_other")[
        "max_sep_end_other"].transform("mean")
    spn = sp[~sp["te_targeted"]]
    sanity = {
        "play_level_te_not_targeted": {
            "te_gravity_vs_max_sep_end_other_raw": _corr(spn["te_gravity"], spn["max_sep_end_other"]),
            "te_gravity_vs_max_sep_end_other_adj_n_receivers": _corr(spn["te_gravity"],
                                                                     spn["max_sep_other_adj"]),
            "te_gravity_vs_mean_sep_over_expected_other": _corr(spn["te_gravity"],
                                                                spn["mean_sep_oe_other"]),
            "te_gravity_vs_max_sep_over_expected_other": _corr(spn["te_gravity"],
                                                               spn["max_sep_oe_other"]),
            "te_gravity_throw_vs_mean_sep_over_expected_other": _corr(spn["te_gravity_throw"],
                                                                      spn["mean_sep_oe_other"]),
        },
        "play_level_all_throws": {
            "te_gravity_vs_max_sep_end_other_raw": _corr(sp["te_gravity"], sp["max_sep_end_other"]),
            "te_gravity_vs_mean_sep_over_expected_other": _corr(sp["te_gravity"],
                                                                sp["mean_sep_oe_other"]),
        },
    }
    rel_tp = te[~te["is_target"]].groupby(["gameId", "playId"])["gravity_rel"].mean()
    spn = spn.join(rel_tp.rename("te_gravity_rel"), on=["gameId", "playId"])
    sanity["play_level_te_not_targeted"]["te_gravity_play_relative_vs_mean_sep_over_expected_"
                                         "other_MECHANICALLY_COUPLED"] = _corr(
        spn["te_gravity_rel"], spn["mean_sep_oe_other"])
    tm = spn.groupby("team").agg(g=("te_gravity", "mean"), s=("max_sep_end_other", "mean"),
                                 so=("mean_sep_oe_other", "mean"))
    sanity["team_level"] = {"te_gravity_vs_mean_max_sep_end_other": _corr(tm["g"], tm["s"]),
                            "te_gravity_vs_mean_sep_over_expected_other": _corr(tm["g"], tm["so"])}
    pl = spn.merge(nt[["gameId", "playId", "nflId"]], on=["gameId", "playId"]).groupby("nflId")[
        "mean_sep_oe_other"].agg(["mean", "count"])
    pl = pl[pl["count"] >= MIN_TOP].join(P.set_index("nflId")["value_shrunk"])
    sanity["te_player_level_min40"] = {
        "te_gravity_shrunk_vs_teammates_sep_over_expected": _corr(pl["value_shrunk"], pl["mean"])}
    # quintiles of TE gravity -> other receivers' sep over expected
    spn = spn.assign(q=pd.qcut(spn["te_gravity"], 5, labels=False))
    sanity["quintiles_te_gravity"] = spn.groupby("q").agg(
        te_gravity=("te_gravity", "mean"), max_sep_end_other=("max_sep_end_other", "mean"),
        mean_sep_oe_other=("mean_sep_oe_other", "mean"), n=("te_gravity", "size")
    ).round(4).reset_index().to_dict(orient="records")

    # ---- summary
    top = P[P["meets_min_n"]].head(5)[["displayName", "team", "n", "value", "value_shrunk", "lo",
                                       "hi", "bracket_oe", "safety_pull_oe"]]
    summary = {
        "metric": "Coverage Gravity", "slug": SLUG, "higher_is_better": True,
        "definition": ("Per TE: mean over NON-targeted route snaps of (observed - expected) "
                       "attention index at landmark t* = min(2.0 s, end of dropback); attention "
                       "index = sum over the 11 defenders of exp(-d/3), d = yards from receiver. "
                       "Expectation from a position-blind HistGradientBoosting model fit on all "
                       "route runners (all positions) in train weeks using alignment/formation/"
                       "coverage/situation context and receiver location at t*; cross-fit "
                       "(GroupKFold by game) for train weeks, train-fit predictions for test weeks."),
        "value_column": "value = raw mean residual (non-target routes); value_shrunk = normal-"
                        "normal EB; lo/hi = 95% normal CI on raw mean",
        "parameters": {"t_star_s": T_STAR, "decay_yd": DECAY, "decay_sensitivity": DECAY_SENS,
                       "n_def_radii_yd": [3, 5], "bracket_radius_yd": BRACKET_R,
                       "bracket_min_offset_yd": BRACKET_MIN_OFF,
                       "bracket_rule": ">=2 defenders within 4 yd with one inside & one outside "
                                       "(toward/away from ball_y) OR one over & one under "
                                       "(dx>=+0.5 / dx<=-0.5)",
                       "safety_positions": SAFETY_POS,
                       "safety_pull": "d(receiver, nearest safety) at snap - at t* (yd; + = "
                                      "safety closer)",
                       "saf_depth_chg": "x_rel(t*) - x_rel(snap) of safety nearest at t* (+ = "
                                        "deeper; two-sided, descriptive)",
                       "throw_frame": "last receiver_frames frame (throw, or end of dropback on "
                                      "sacks/scrambles)",
                       "eb_shrinkage": "stats.eb_shrink_mean with pooled within-TE SD "
                                       f"({pooled:.4f}) for every player",
                       "min_routes_leaderboard": MIN_TOP, "split_half_min_n": MIN_SPLIT_HALF,
                       "red_zone_yards": RED_ZONE_YARDS, "train_weeks": TRAIN_WEEKS,
                       "test_weeks": TEST_WEEKS, "hgb": HGB_KW,
                       "features_t_star": feat_star, "features_throw": feat_end,
                       "inline_split": "inline = alignment inline or wing; detached = detached; "
                                       "backfield reported only in all-routes"},
        "sample_sizes": {"route_runners": int(len(df)), "te_routes": int(len(te)),
                         "te_nontarget_routes": int(len(nt)), "tes": int(te["nflId"].nunique()),
                         "tes_min_n": int(P["meets_min_n"].sum()),
                         "te_nontarget_red_zone": int((nt["red_zone"] == 1).sum()),
                         "te_nontarget_inline": int((nt["inline_detached"] == "inline").sum()),
                         "te_nontarget_detached": int((nt["inline_detached"] == "detached").sum()),
                         "te_routes_with_safety": int(te["safety_pull_star"].notna().sum())},
        "consistency_with_shared_routes_table": consistency,
        "test_set_model_metrics": model_metrics,
        "split_half_reliability": rel,
        "sensitivity_rank_corr_vs_value_min_n": sens,
        "mean_att_residual_by_position_all_runners": pos_resid,
        "te_gravity_by_alignment_and_target": align_resid,
        "sanity_check_other_receiver_separation": sanity,
        "top5_min_n": top.round(4).to_dict(orient="records"),
        "caveats": [
            "Attention is proximity-based; it cannot distinguish a defender assigned to the TE "
            "from one passing through the same zone.",
            "Receiver location at t* is a model feature, so gravity measures attention beyond "
            "what a comparable receiver at the same spot gets; route choice that takes the TE "
            "into traffic is absorbed (see gravity_ctxonly sensitivity).",
            "Inline TEs at t* are often near the box, where expected attention is high; "
            "inline and detached splits should be compared within split.",
            "Safety measures exist only on plays with PFF-labelled safeties; depth change is "
            "two-sided and not signed as good/bad.",
            "Throw frame = last tracked frame of dropback; no ball-flight tracking exists.",
            "Non-target restriction relies on targetNflId parsed from playDescription "
            "(unmatched attempts are treated as non-targets).",
            "8 weeks of data: per-TE samples are small; use shrunk values and CIs.",
            "Targeted routes have lower residual attention than non-targeted ones (QBs throw to "
            "open receivers), so the non-target restriction mechanically raises TE values; "
            "compare TEs to each other, not to zero.",
            "Play-level residuals share a play-wide 'coverage tightness' component (tight man "
            "coverage puts defenders near everyone), so TE gravity correlates NEGATIVELY with "
            "teammates' separation-over-expected; the sanity check does not support a "
            "'TE attention frees teammates' interpretation in this data. gravity_rel nets out "
            "the play-wide component but is mechanically coupled to teammates' attention.",
            "Split-half reliability is near zero: at this sample size Coverage Gravity is not "
            "demonstrably a stable player trait; weight it lightly in any composite.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"reliability": rel, "sanity": sanity["play_level_te_not_targeted"],
                      "team": sanity["team_level"]}, indent=1))
    print(top.to_string())


if __name__ == "__main__":
    main()

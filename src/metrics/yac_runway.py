"""YAC Runway.

Space ahead of / around the receiver at the (projected) catch point, and how much of the
yards-after-catch opportunity comes from that space versus the receiver's own speed into the catch.

Tracking ends ~0.5 s after the throw and there is no catch-point / air-yards / YAC data, so:
  1. flight time T_hat = |QB - target| at the throw frame / BALL_SPEED, where BALL_SPEED is
     calibrated on plays whose football track reaches pass_arrived / pass_outcome_caught;
  2. target and every defender are projected from the LAST tracked frame at constant velocity for
     tau = max(T_hat - (last_frame - throw_frame)/10, 0) seconds;
  3. catch_depth = projected target x_rel; yac_est = prePenaltyPlayResult - catch_depth
     (implausible values are flagged, never dropped; the model target is clipped to YAC_CLIP);
  4. Runway    = E[YAC | space at catch]               (HGB, cross-fit, all receivers' completions)
     Runway+   = E[YAC | space + receiver speed/vx/accel]
     momentum  = Runway+ - Runway     yacoe = yac (clipped) - Runway+
  5. TE value = mean Runway per TE (higher = better), EB-shrunk, with 95% CI.

Run from project root:  .venv/bin/python -m src.metrics.yac_runway
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from src.common import io, stats  # noqa: E402
from src.common.config import FIELD_W, FPS, OUT, SEED, TEST_WEEKS, TRAIN_WEEKS  # noqa: E402

SLUG = "yac_runway"
OUT_DIR = OUT / SLUG
PKEY = ["gameId", "playId"]

ARRIVAL_EVENTS = ["pass_arrived", "pass_outcome_caught"]
YAC_IMPLAUSIBLE = -3.0      # yac_est below this is flagged (projection / target-parse suspect)
YAC_CLIP = (-5.0, 30.0)     # model target clipping (squared loss on clipped target => robust mean)
CONE_HALF_ANGLE = 45.0      # degrees either side of straight downfield (+x)
CONE_RADIUS = 10.0          # yards
RADII = (5.0, 10.0)
MIN_TOP = 10                # min catches for top-5 eligibility
REL_MIN_N = [10, 5]         # per-half minimum catches for reliability
TRACK_COLS = ["gameId", "playId", "nflId", "frameId", "team", "is_def", "pff_role",
              "x", "y", "x_rel", "s", "a", "vx", "vy", "event"]

SPACE_FEATS = ["catch_depth", "nd_dist", "nd_dx", "n_def_5", "n_def_10", "n_cone",
               "sideline_dist", "ytg_catch", "nd_closing"]
SPEED_FEATS = ["rec_s", "rec_vx", "rec_a"]
GBM = dict(loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
           min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


# --------------------------------------------------------------------------------------------- data
def load_states(pl: pd.DataFrame):
    """Per play: QB/target at throw, target + defenders at last frame, football arrival (if any)."""
    rows_thr, rows_last, rows_def, rows_arr = [], [], [], []
    meta = pl[PKEY + ["throw_frame", "last_frame", "targetNflId"]]
    for g, mg in meta.groupby("gameId"):
        tr = io.tracking(columns=TRACK_COLS, filters=[("gameId", "==", int(g))])
        tr = tr.merge(mg, on=PKEY, how="inner")
        tr = tr[tr.frameId >= tr.throw_frame]
        at_thr = tr[tr.frameId == tr.throw_frame]
        rows_thr.append(at_thr[at_thr.pff_role == "Pass"].drop_duplicates(PKEY)[PKEY + ["x", "y"]]
                        .rename(columns={"x": "qb_x", "y": "qb_y"})
                        .merge(at_thr[at_thr.nflId == at_thr.targetNflId][PKEY + ["x", "y", "vx", "vy"]]
                               .rename(columns={"x": "rt_x", "y": "rt_y", "vx": "rt_vx", "vy": "rt_vy"}),
                               on=PKEY))
        last = tr[tr.frameId == tr.last_frame]
        rows_last.append(last[last.nflId == last.targetNflId][PKEY + ["x", "y", "x_rel", "s", "a", "vx", "vy"]])
        rows_def.append(last[last.is_def][PKEY + ["nflId", "x", "y", "vx", "vy"]])
        ball = tr[(tr.team == "football") & tr.event.isin(ARRIVAL_EVENTS)]
        arr = ball.groupby(PKEY).agg(arr_frame=("frameId", "min")).reset_index()
        arr = arr.merge(ball[PKEY + ["frameId", "x", "y"]].rename(
            columns={"frameId": "arr_frame", "x": "ball_x_arr", "y": "ball_y_arr"}), on=PKEY + ["arr_frame"])
        rec_arr = tr[tr.nflId == tr.targetNflId][PKEY + ["frameId", "x", "y"]].rename(
            columns={"frameId": "arr_frame", "x": "rec_x_arr", "y": "rec_y_arr"})
        rows_arr.append(arr.merge(rec_arr, on=PKEY + ["arr_frame"], how="left"))
    thr = pd.concat(rows_thr, ignore_index=True)
    lst = pd.concat(rows_last, ignore_index=True)
    dfn = pd.concat(rows_def, ignore_index=True)
    arr = pd.concat(rows_arr, ignore_index=True).drop_duplicates(PKEY)
    return thr, lst, dfn, arr


# -------------------------------------------------------------------------------------- calibration
def calibrate(thr: pd.DataFrame, arr: pd.DataFrame, pl: pd.DataFrame):
    c = arr.merge(thr, on=PKEY).merge(pl[PKEY + ["throw_frame", "passResult"]], on=PKEY)
    c["flight_t"] = (c.arr_frame - c.throw_frame) / FPS
    c = c[c.flight_t > 0].copy()
    c["d_throw"] = np.hypot(c.rt_x - c.qb_x, c.rt_y - c.qb_y)
    v = float(np.median(c.d_throw / c.flight_t))
    c["t_hat"] = c.d_throw / v
    # project receiver from the THROW frame over t_hat (the honest test: real horizon, no peeking)
    px, py = c.rt_x + c.rt_vx * c.t_hat, c.rt_y + c.rt_vy * c.t_hat
    err_proj = np.hypot(px - c.rec_x_arr, py - c.rec_y_arr)
    err_static = np.hypot(c.rt_x - c.rec_x_arr, c.rt_y - c.rec_y_arr)
    err_dx = (px - c.rec_x_arr)
    rec_ball = np.hypot(c.rec_x_arr - c.ball_x_arr, c.rec_y_arr - c.ball_y_arr)
    q = lambda s: {k: round(float(s.quantile(p)), 3) for k, p in [("p50", .5), ("p90", .9)]}  # noqa: E731
    out = {
        "n_calibration_plays": int(len(c)),
        "n_caught_event_plays": int((c.passResult == "C").sum()),
        "ball_speed_yd_per_s": round(v, 3),
        "calibration_d_throw_yd": {"mean": round(float(c.d_throw.mean()), 2), "max": round(float(c.d_throw.max()), 2)},
        "calibration_flight_t_s": {"mean": round(float(c.flight_t.mean()), 3), "max": float(c.flight_t.max())},
        "flight_time_mae_s": round(float((c.t_hat - c.flight_t).abs().mean()), 3),
        "catch_point_error_yd_const_velocity": {"mean": round(float(err_proj.mean()), 3), **q(err_proj)},
        "catch_point_error_yd_no_projection": {"mean": round(float(err_static.mean()), 3), **q(err_static)},
        "catch_depth_bias_yd (proj - actual x)": round(float(err_dx.mean()), 3),
        "catch_depth_mae_yd": round(float(err_dx.abs().mean()), 3),
        "receiver_to_ball_at_arrival_yd_p50": round(float(rec_ball.median()), 3),
        "note": ("Calibration plays are ONLY quick throws whose arrival falls inside the ~0.5 s of "
                 "post-throw tracking (flight <= 0.5 s, short distance). Errors on longer throws "
                 "(longer projection horizon) will be larger."),
    }
    return v, out


def ball_speed_crosscheck(pl):
    """Tracked football ground speed in flight (all throws) as an independent sanity check."""
    b = io.tracking(columns=["gameId", "playId", "frameId", "s"], filters=[("team", "==", "football")])
    b = b.merge(pl[PKEY + ["throw_frame"]], on=PKEY)
    b = b[b.frameId >= b.throw_frame + 2]
    return round(float(b.groupby(PKEY).s.median().median()), 3)


# --------------------------------------------------------------------------------------- features
def build_features(comp, thr, lst, dfn, v_ball):
    df = comp.merge(thr, on=PKEY).merge(lst, on=PKEY)
    df["d_throw"] = np.hypot(df.rt_x - df.qb_x, df.rt_y - df.qb_y)
    df["t_hat"] = df.d_throw / v_ball
    df["elapsed"] = (df.last_frame - df.throw_frame) / FPS
    df["tau"] = (df.t_hat - df.elapsed).clip(lower=0)
    df["cx"] = df.x + df.vx * df.tau
    df["cy"] = (df.y + df.vy * df.tau).clip(0, FIELD_W)
    df["catch_depth"] = df.x_rel + df.vx * df.tau
    df = df.rename(columns={"s": "rec_s", "a": "rec_a", "vx": "rec_vx", "vy": "rec_vy"})
    d = dfn.merge(df[PKEY + ["tau", "cx", "cy", "rec_vx", "rec_vy"]], on=PKEY)
    d["dx_"] = (d.x + d.vx * d.tau) - d.cx          # defender minus receiver (projected)
    d["dy_"] = (d.y + d.vy * d.tau) - d.cy
    d["dist"] = np.hypot(d.dx_, d.dy_)
    ang = np.degrees(np.abs(np.arctan2(d.dy_, d.dx_)))
    d["in_cone"] = (d.dist <= CONE_RADIUS) & (ang <= CONE_HALF_ANGLE)
    # closing speed: rate at which defender-receiver distance shrinks (positive = closing)
    d["closing"] = -((d.vx - d.rec_vx) * d.dx_ + (d.vy - d.rec_vy) * d.dy_) / d.dist.clip(lower=0.1)
    nd = d.sort_values("dist").drop_duplicates(PKEY)[PKEY + ["dist", "dx_", "closing"]].rename(
        columns={"dist": "nd_dist", "dx_": "nd_dx", "closing": "nd_closing"})
    agg = d.groupby(PKEY).agg(n_def_5=("dist", lambda s: int((s <= RADII[0]).sum())),
                              n_def_10=("dist", lambda s: int((s <= RADII[1]).sum())),
                              n_cone=("in_cone", "sum")).reset_index()
    df = df.merge(nd, on=PKEY, how="left").merge(agg, on=PKEY, how="left")
    df["sideline_dist"] = np.minimum(df.cy, FIELD_W - df.cy)
    df["ytg_catch"] = (df.yards_to_goal - df.catch_depth).clip(lower=0)
    df["yac_est"] = df.prePenaltyPlayResult - df.catch_depth
    df["flag_implausible"] = df.yac_est < YAC_IMPLAUSIBLE
    df["yac_clip"] = df.yac_est.clip(*YAC_CLIP)
    return df


# ------------------------------------------------------------------------------------------ models
def regression_metrics(y, p, base):
    return {"rmse": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "mae": round(float(mean_absolute_error(y, p)), 3), "r2": round(float(r2_score(y, p)), 4),
            "baseline_rmse": round(float(np.sqrt(mean_squared_error(y, np.full(len(y), base)))), 3),
            "baseline_mae": round(float(mean_absolute_error(y, np.full(len(y), base))), 3),
            "baseline_r2": round(float(r2_score(y, np.full(len(y), base))), 4)}


def fit_models(df):
    res = {}
    model = HistGradientBoostingRegressor(**GBM)
    for name, feats in [("runway", SPACE_FEATS), ("runway_plus", SPACE_FEATS + SPEED_FEATS)]:
        df[name], _ = stats.cross_fit(model, df, feats, "yac_clip")
    te = df[~df.is_train]
    base = float(df.loc[df.is_train, "yac_clip"].mean())
    for name in ["runway", "runway_plus"]:
        res[name] = {"test_vs_clipped_target": regression_metrics(te.yac_clip, te[name], base),
                     "test_vs_raw_yac_est": regression_metrics(te.yac_est, te[name], base)}
        tte = te[te.is_te]
        res[name]["test_TE_only_vs_clipped"] = regression_metrics(tte.yac_clip, tte[name], base)
    res["train_mean_baseline"] = round(base, 3)
    df["momentum"] = df.runway_plus - df.runway
    df["yacoe"] = df.yac_clip - df.runway_plus
    return res


def reliability(d, col):
    out = {}
    for mn in REL_MIN_N:
        r1, n1 = stats.split_half_reliability(d, "nflId", col, min_n=mn)
        r2, n2 = stats.odd_even_reliability(d, "nflId", col, min_n=mn)
        out[f"min_n_{mn}"] = {"split_half_weeks": [None if np.isnan(r1) else round(float(r1), 3), int(n1)],
                              "odd_even_games": [None if np.isnan(r2) else round(float(r2), 3), int(n2)]}
    return out


# -------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl = io.plays()
    pl = pl[(pl.end_event == "throw") & pl.targetNflId.notna()].copy()
    pl["targetNflId"] = pl.targetNflId.astype("int64")
    thr, lst, dfn, arr = load_states(pl)
    v_ball, calib = calibrate(thr, arr, pl)
    calib["tracked_ball_ground_speed_median_all_throws"] = ball_speed_crosscheck(pl)
    print(f"ball speed {v_ball:.2f} yd/s  ({time.time() - t0:.0f}s)")

    pp = io.player_plays()[PKEY + ["nflId", "displayName", "officialPosition"]].rename(
        columns={"nflId": "targetNflId"})
    comp = pl[pl.passResult == "C"][PKEY + ["week", "is_train", "throw_frame", "last_frame", "targetNflId",
                                            "possessionTeam", "prePenaltyPlayResult", "yards_to_goal"]]
    comp = comp.merge(pp, on=PKEY + ["targetNflId"], how="left")
    comp["is_te"] = comp.officialPosition == "TE"
    df = build_features(comp, thr, lst, dfn, v_ball)
    n_missing_def = int(df.nd_dist.isna().sum())
    df = df[df.nd_dist.notna()].copy()
    model_res = fit_models(df)
    print(json.dumps(model_res, indent=1))

    df = df.rename(columns={"targetNflId": "nflId"})
    keep = PKEY + ["nflId", "displayName", "officialPosition", "is_te", "possessionTeam", "week", "is_train",
                   "prePenaltyPlayResult", "d_throw", "t_hat", "tau"] + SPACE_FEATS + SPEED_FEATS + [
        "yac_est", "yac_clip", "flag_implausible", "runway", "runway_plus", "momentum", "yacoe"]
    df[keep].round(4).to_csv(OUT_DIR / "plays.csv", index=False)

    te = df[df.is_te].copy()
    sd = stats.pooled_within_sd(te, "nflId", "runway")
    sd_y = stats.pooled_within_sd(te, "nflId", "yacoe")
    g = te.groupby("nflId")
    tab = g.agg(displayName=("displayName", "first"), team=("possessionTeam", lambda s: s.mode().iat[0]),
                n=("runway", "size"), value=("runway", "mean"), momentum=("momentum", "mean"),
                yacoe=("yacoe", "mean"), yac_est_mean=("yac_est", "mean"),
                catch_depth_mean=("catch_depth", "mean"), nd_dist_mean=("nd_dist", "mean"),
                n_implausible=("flag_implausible", "sum")).reset_index()
    ci = g.runway.apply(stats.mean_ci).unstack()[["lo", "hi"]].reset_index()
    tab = tab.merge(ci, on="nflId")
    tab["value_shrunk"] = stats.eb_shrink_mean(tab.value, tab.n, sd)
    tab["yacoe_shrunk"] = stats.eb_shrink_mean(tab.yacoe, tab.n, sd_y)
    tab["top5_eligible"] = tab.n >= MIN_TOP
    tab["higher_is_better"] = True
    # method-of-moments tau^2 on n >= 10 players (same rule eb_shrink_mean uses), for reporting
    h = tab[tab.n >= MIN_TOP]
    tau2_runway = float(h.value.var(ddof=0) - (sd**2 / h.n).mean())
    # when tau^2 <= 0 every value_shrunk collapses to the grand mean; break ties on the raw mean
    tab["_vs"] = tab.value_shrunk.round(3)
    tab = tab.sort_values(["_vs", "value"], ascending=False).drop(columns="_vs")
    cols = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "momentum", "yacoe",
            "yacoe_shrunk", "yac_est_mean", "catch_depth_mean", "nd_dist_mean", "n_implausible",
            "top5_eligible", "higher_is_better"]
    tab[cols].round(4).to_csv(OUT_DIR / "players.csv", index=False)
    top5 = tab[tab.top5_eligible].head(5)[["displayName", "n", "value", "value_shrunk", "momentum", "yacoe"]]
    print(top5.round(2).to_string(index=False))

    rel = {"TE": {c: reliability(te, c) for c in ["runway", "momentum", "yacoe"]},
           "all_receivers_context": {c: reliability(df, c) for c in ["runway", "yacoe"]}}
    print(json.dumps(rel["TE"], indent=1))

    summary = {
        "metric": "YAC Runway", "slug": SLUG,
        "definition": ("Completions with a parsed target. Flight time = QB-target distance at throw / "
                       "calibrated ball speed; target and defenders projected at constant velocity from "
                       "the last tracked frame (throw+~0.5 s) for the remaining flight time. catch_depth = "
                       "projected target x_rel; yac_est = prePenaltyPlayResult - catch_depth. "
                       "Runway = E[YAC | space at catch]; Runway+ = E[YAC | space + receiver s/vx/a]. "
                       "TE value = mean Runway (expected YAC from space); momentum = Runway+ - Runway; "
                       "yacoe = clipped yac_est - Runway+."),
        "parameters": {"ball_speed_yd_per_s": round(v_ball, 3), "arrival_events": ARRIVAL_EVENTS,
                       "yac_implausible_below": YAC_IMPLAUSIBLE, "yac_target_clip": YAC_CLIP,
                       "cone_half_angle_deg": CONE_HALF_ANGLE, "cone_radius_yd": CONE_RADIUS,
                       "def_radii_yd": RADII, "min_catches_top5": MIN_TOP,
                       "space_features": SPACE_FEATS, "speed_features": SPEED_FEATS,
                       "model": "HistGradientBoostingRegressor", "model_params": GBM,
                       "loss_choice": ("squared_error on yac_est clipped to [-5, 30] (keeps E[.] a mean "
                                       "while limiting long-run outliers); no tuning grid"),
                       "position_used_as_feature": False,
                       "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS},
        "projection_calibration": calib,
        "sample_sizes": {"completions_with_target_and_tracking": int(len(df)),
                         "dropped_no_defender_rows": n_missing_def,
                         "train": int(df.is_train.sum()), "test": int((~df.is_train).sum()),
                         "te_catches": int(len(te)), "te_players": int(te.nflId.nunique()),
                         "te_players_n_ge_10": int((tab.n >= MIN_TOP).sum()),
                         "implausible_flagged_all": int(df.flag_implausible.sum()),
                         "implausible_flagged_te": int(te.flag_implausible.sum()),
                         "tau_s (projection beyond tracking)": {
                             "median": round(float(df.tau.median()), 3),
                             "p90": round(float(df.tau.quantile(.9)), 3),
                             "share_zero": round(float((df.tau == 0).mean()), 3)}},
        "model_test_metrics": model_res,
        "eb_pooled_within_sd": {"runway": round(sd, 4), "yacoe": round(sd_y, 4)},
        "eb_tau2_runway_raw_moment": round(tau2_runway, 4),
        "eb_note": ("tau^2 <= 0 means between-TE spread of mean Runway (n>=10) is no larger than "
                    "sampling noise, so value_shrunk collapses to the grand mean for everyone; the "
                    "players.csv order then falls back to the raw mean."),
        "reliability": rel,
        "top5_min10": top5.round(3).to_dict(orient="records"),
        "caveats": [
            "No catch point, air yards or YAC exist in the data: catch depth and YAC are projections, "
            "so every number inherits projection error (see projection_calibration).",
            "Ball speed is calibrated only on quick throws (<=0.5 s flight) whose arrival is tracked; "
            "longer throws extrapolate both the speed and constant-velocity paths.",
            "Constant-velocity projection ignores route breaks and defender reactions after throw+0.5 s.",
            "Small samples: most TEs have well under 30 catches in 8 weeks; per-TE CIs are wide and "
            "split-half reliability uses few players (weeks 7-8 hold ~1/4 of the data). Treat ranks "
            "as descriptive, not as a stable trait estimate, unless reliability says otherwise.",
            "Runway mixes scheme and player: space at the catch reflects play design, coverage AND the "
            "receiver's route; momentum/yacoe isolate the player's speed and finishing only partially.",
            "yac_est includes fumbles/laterals and any penalty-free play result quirks; implausible "
            "values (< -3) are flagged and kept (bounded by the target clip).",
        ],
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

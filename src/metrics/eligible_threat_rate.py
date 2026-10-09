"""Eligible Threat Rate (ETR) for tight ends.

A TE dropback snap is an *eligible threat* if ALL of:
  1. assignment in {route, chip_release}
  2. release_t <= T_rel   (T_rel = train-week 75th pct of TE release_t on pure route snaps)
  3. at some frame f with release_t <= t(f) and
         throw plays:     t(f) <  time_to_throw            (strictly before the throw frame)
         non-throw plays: t(f) <= min(time_to_end, T_cap) (T_cap = train p75 of time_to_throw)
     the TE is in a viable receiving position:
         sep >= VIABLE_MIN_SEP, lane_clear >= VIABLE_MIN_LANE, x_rel in VIABLE_DEPTH,
         VIABLE_SIDELINE_BUFFER <= y <= FIELD_W - VIABLE_SIDELINE_BUFFER,
         v_away <= MAX_AWAY_SPEED, where v_away = (vx, vy) . unit(QB -> TE)
         (receiver's own velocity along the QB->receiver axis; QB motion is not subtracted).

Denominator = every TE dropback snap with tracking (te_plays rows on plays with a snap frame).

Run: .venv/bin/python -m src.metrics.eligible_threat_rate
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
from src.common.config import (FIELD_W, OUT, SEED, TEST_WEEKS, TRAIN_WEEKS,  # noqa: E402
                               VIABLE_DEPTH, VIABLE_MIN_LANE, VIABLE_MIN_SEP,
                               VIABLE_SIDELINE_BUFFER, RELEASE_MIN_DISP, RELEASE_MIN_SPEED,
                               RELEASE_SUSTAIN)

warnings.filterwarnings("ignore", category=FutureWarning)

SLUG = "eligible_threat_rate"
OUT_DIR = OUT / SLUG
KEY = ["gameId", "playId", "nflId"]
EPS = 1e-6

T_REL_PCT = 0.75          # percentile of train-week TE release_t (pure route snaps) -> T_rel
T_CAP_PCT = 0.75          # percentile of train-week plays.time_to_throw -> T_cap
MAX_AWAY_SPEED = 1.0      # yd/s, max receiver velocity away from QB along QB->receiver axis
MIN_N_RANK = 50           # min snaps for rankings / top-5 / sensitivity Spearman
ROUTE_ASSIGN = ["route", "chip_release"]

GRID_T_REL = [0.8, 1.0, None, 1.2, 1.5]   # None -> derived default T_rel
GRID_SEP = [2.0, 2.5, 3.0]
GRID_LANE = [0.5, 1.0, 1.5]
GRID_AWAY = [0.5, 1.0, 2.0, 3.0, np.inf]

CAT_FEATS = ["alignment", "offenseFormation", "personnelO", "pff_passCoverage", "dropBackType"]
NUM_FEATS = ["pff_playAction", "down", "yardsToGo", "red_zone"]
FEATURES = CAT_FEATS + NUM_FEATS
HGB_GRID = [
    dict(learning_rate=0.05, max_iter=200, max_depth=3, min_samples_leaf=40, l2_regularization=1.0),
    dict(learning_rate=0.05, max_iter=400, max_depth=2, min_samples_leaf=80, l2_regularization=1.0),
    dict(learning_rate=0.03, max_iter=300, max_depth=4, min_samples_leaf=60, l2_regularization=3.0),
]


# --------------------------------------------------------------------------- data
def load():
    p = io.plays()
    te = io.te_plays()
    ctx = p[["gameId", "playId", "possessionTeam", "offenseFormation", "personnelO", "pff_passCoverage",
             "pff_playAction", "dropBackType", "down", "yardsToGo", "red_zone", "snap_frame",
             "end_event", "time_to_throw", "time_to_end"]]
    te = te.merge(ctx, on=["gameId", "playId"], how="left")
    n_all = len(te)
    te = te[te.snap_frame.notna()].reset_index(drop=True)   # no snap frame -> no tracking
    te["is_throw"] = te.end_event.eq("throw") & te.time_to_throw.notna()
    te["is_route"] = te.assignment.isin(ROUTE_ASSIGN)

    rf = io.receiver_frames(
        columns=["gameId", "playId", "nflId", "t", "x", "y", "x_rel", "vx", "vy", "qb_x", "qb_y",
                 "sep", "lane_clear", "dist_qb"],
        filters=[("nflId", "in", te.loc[te.is_route, "nflId"].unique().tolist())])
    rf = rf.merge(te.loc[te.is_route, KEY + ["release_t", "is_throw", "time_to_throw", "time_to_end"]],
                  on=KEY, how="inner")
    d = rf.dist_qb.where(rf.dist_qb > 0.1)
    rf["v_away"] = (rf.vx * (rf.x - rf.qb_x) + rf.vy * (rf.y - rf.qb_y)) / d
    return p, te, rf, n_all


def derive_thresholds(p, te):
    tr_route = te[te.is_train & (te.assignment == "route")].release_t.dropna()
    t_rel = float(np.round(tr_route.quantile(T_REL_PCT), 2))
    tr_tt = p[p.is_train & p.end_event.eq("throw")].time_to_throw.dropna()
    t_cap = float(np.round(tr_tt.quantile(T_CAP_PCT), 2))
    return t_rel, t_cap, len(tr_route), len(tr_tt)


def window_mask(rf, t_cap, cap_throws=False):
    """Frames from release through the end of the decision window."""
    end_throw = rf.time_to_throw - 0.1 + EPS                       # strictly before throw frame
    if cap_throws:
        end_throw = np.minimum(end_throw, t_cap + EPS)
    end_other = np.minimum(rf.time_to_end, t_cap) + EPS
    end = np.where(rf.is_throw, end_throw, end_other)
    return rf.release_t.notna().values & (rf.t.values >= rf.release_t.values - EPS) & (rf.t.values <= end)


def viable_mask(rf, sep=VIABLE_MIN_SEP, lane=VIABLE_MIN_LANE, away=MAX_AWAY_SPEED):
    m = ((rf.sep >= sep) & (rf.lane_clear >= lane)
         & rf.x_rel.between(VIABLE_DEPTH[0], VIABLE_DEPTH[1])
         & rf.y.between(VIABLE_SIDELINE_BUFFER, FIELD_W - VIABLE_SIDELINE_BUFFER))
    if np.isfinite(away):
        m &= rf.v_away <= away   # NaN v_away (QB on top of TE) -> not viable
    return m.values


def any_by_snap(rf, mask, te):
    """Per TE-snap: did any frame satisfy mask? aligned to te index."""
    hit = rf.loc[mask, KEY].drop_duplicates()
    hit["_hit"] = True
    return te[KEY].merge(hit, on=KEY, how="left")["_hit"].fillna(False).astype(bool).values


def eligible_flags(te, rf, win, t_rel, sep, lane, away):
    viable = any_by_snap(rf, win & viable_mask(rf, sep, lane, away), te)
    rel_ok = te.is_route.values & (te.release_t.values <= t_rel + EPS)
    return rel_ok & viable


# --------------------------------------------------------------------------- per-player
def beta_prior_strength(k, n, min_n=MIN_N_RANK):
    """Method-of-moments beta-binomial prior strength from players with n >= min_n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    sel = n >= min_n
    k, n = k[sel], n[sel]
    m = k.sum() / n.sum()
    r = k / n
    tau2 = np.var(r, ddof=1) - np.mean(m * (1 - m) / n)
    if tau2 <= 1e-6:
        return 1000.0, m, float(max(tau2, 0))
    return float(min(max(m * (1 - m) / tau2 - 1, 5.0), 1000.0)), m, float(tau2)


def rate_table(df, flag, by="nflId"):
    g = df.groupby(by)[flag].agg(["sum", "count"])
    g.columns = ["k", "n"]
    return g


def player_table(te, prior):
    g = te.groupby("nflId")
    out = pd.DataFrame({
        "displayName": g.displayName.agg(lambda s: s.mode().iat[0]),
        "team": g.possessionTeam.agg(lambda s: s.mode().iat[0]),
        "n": g.size(),
        "k_eligible": g.eligible.sum(),
    })
    out["value"] = out.k_eligible / out.n
    out["value_shrunk"] = stats.beta_shrink(out.k_eligible, out.n, prior["all"]["mean"], prior["all"]["strength"])
    out["lo"], out["hi"] = stats.wilson_ci(out.k_eligible, out.n)

    r = te[te.is_route].groupby("nflId")
    out["n_route"] = r.size().reindex(out.index).fillna(0).astype(int)
    out["k_route_eligible"] = r.eligible.sum().reindex(out.index).fillna(0).astype(int)
    out["route_share"] = out.n_route / out.n
    out["etr_route"] = out.k_route_eligible / out.n_route.replace(0, np.nan)
    out["etr_route_lo"], out["etr_route_hi"] = stats.wilson_ci(out.k_route_eligible, out.n_route)
    out["etr_route_shrunk"] = stats.beta_shrink(out.k_route_eligible, out.n_route,
                                                prior["route"]["mean"], prior["route"]["strength"])
    out["release_ok_rate_route"] = r.release_ok.mean().reindex(out.index)
    out["viable_rate_route"] = r.viable_any.mean().reindex(out.index)
    out["median_first_viable_t"] = te[te.eligible].groupby("nflId").first_viable_t.median().reindex(out.index)

    # context-adjusted
    out["expected"] = g.expected.mean()
    res = te.groupby("nflId").resid.apply(stats.mean_ci).unstack()
    out["etr_oe"] = res["mean"]
    out["etr_oe_lo"], out["etr_oe_hi"] = res["lo"], res["hi"]
    out["etr_oe_shrunk"] = stats.eb_shrink_mean(out.etr_oe, out.n, g.resid.std(ddof=1).fillna(0))

    # alignment splits (reporting principle: separate inline / detached / backfield)
    for a in ["inline", "wing", "detached", "backfield"]:
        s = te[te.alignment == a].groupby("nflId").eligible.agg(["mean", "count"]).reindex(out.index)
        out[f"n_{a}"] = s["count"].fillna(0).astype(int)
        out[f"etr_{a}"] = s["mean"]
    out["higher_is_better"] = True
    out = out.reset_index().sort_values("value_shrunk", ascending=False)
    return out


# --------------------------------------------------------------------------- model
def prep_features(te):
    X = te.copy()
    for c in CAT_FEATS:
        X[c] = X[c].fillna("NA").astype(str).astype("category")
    X["pff_playAction"] = X.pff_playAction.astype(float)
    X["red_zone"] = X.red_zone.astype(float)
    X["down"] = X.down.astype(float)
    X["yardsToGo"] = X.yardsToGo.astype(float)
    X["eligible_i"] = X.eligible.astype(int)
    return X


def fit_expected(te):
    X = prep_features(te)
    tr = X[X.is_train]
    tuning = []
    best = None
    for i, params in enumerate(HGB_GRID):
        m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED,
                                           early_stopping=False, **params)
        pred, _ = stats.cross_fit(m, tr, FEATURES, "eligible_i", proba=True)
        ll = log_loss(tr.eligible_i, pred.clip(1e-6, 1 - 1e-6))
        tuning.append({"config": i, **params, "train_oof_logloss": round(ll, 5)})
        if best is None or ll < best[0]:
            best = (ll, params)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED,
                                           early_stopping=False, **best[1])
    pred, _ = stats.cross_fit(model, X, FEATURES, "eligible_i", proba=True)

    test = X[~X.is_train]
    y, pt = test.eligible_i.values, pred[test.index].values
    base = tr.eligible_i.mean()
    pb = np.full_like(pt, base)
    metrics = {
        "n_test": int(len(test)), "test_base_rate": float(y.mean()), "train_base_rate": float(base),
        "model": {"auc": roc_auc_score(y, pt), "log_loss": log_loss(y, pt.clip(1e-6, 1 - 1e-6)),
                  "brier": brier_score_loss(y, pt)},
        "baseline_train_rate": {"auc": 0.5, "log_loss": log_loss(y, pb), "brier": brier_score_loss(y, pb)},
        "selected_config": best[1], "tuning_train_oof": tuning,
    }
    bins = pd.qcut(pt, 5, labels=False, duplicates="drop")
    cal = pd.DataFrame({"bin": bins, "pred": pt, "obs": y}).groupby("bin").agg(
        n=("obs", "size"), mean_pred=("pred", "mean"), obs_rate=("obs", "mean")).reset_index()
    metrics["calibration_test_5bin"] = cal.round(4).to_dict("records")
    for k in ("model", "baseline_train_rate"):
        metrics[k] = {a: round(float(b), 5) for a, b in metrics[k].items()}
    return pred, metrics


# --------------------------------------------------------------------------- validation
def validation(te):
    """Target / completion rates on TE route snaps (throw plays) by eligibility."""
    rows = []
    v = te[te.is_route & te.is_throw]
    for split, sub in [("all", v), ("train", v[v.is_train]), ("test", v[~v.is_train])]:
        for name, col in [("eligible", "eligible")]:
            for flag in (True, False):
                s = sub[sub[col] == flag]
                n, kt, kc = len(s), int(s.is_target.sum()), int(s.is_catch.sum())
                tlo, thi = stats.wilson_ci(kt, n)
                clo, chi = stats.wilson_ci(kc, kt)
                rows.append({"split": split, "group": f"{name}={flag}", "n_snaps": n, "targets": kt,
                             "catches": kc, "target_rate": kt / n if n else np.nan,
                             "target_lo": float(tlo), "target_hi": float(thi),
                             "completion_rate": kc / kt if kt else np.nan,
                             "completion_lo": float(clo), "completion_hi": float(chi),
                             "catch_per_snap": kc / n if n else np.nan})
    # isolate the viability component: route snaps that released in time
    s0 = v[v.release_ok]
    for flag in (True, False):
        s = s0[s0.viable_any == flag]
        n, kt, kc = len(s), int(s.is_target.sum()), int(s.is_catch.sum())
        rows.append({"split": "all", "group": f"release_ok & viable={flag}", "n_snaps": n, "targets": kt,
                     "catches": kc, "target_rate": kt / n, "completion_rate": kc / kt if kt else np.nan,
                     "catch_per_snap": kc / n})
    return pd.DataFrame(rows).round(4)


def target_lift(te, elig):
    v = (te.is_route & te.is_throw).values
    e, t = elig[v], te.is_target.values[v]
    if e.sum() == 0 or (~e).sum() == 0:
        return np.nan
    return float(t[e].mean() / t[~e].mean())


def player_rank_corr(te, elig, ref_rate):
    df = pd.DataFrame({"nflId": te.nflId.values, "e": elig})
    g = df.groupby("nflId").e.agg(["mean", "count"])
    g = g[g["count"] >= MIN_N_RANK]
    return float(sps.spearmanr(g["mean"], ref_rate.reindex(g.index)).correlation), int(len(g))


def sensitivity(te, rf, win_default, win_capall, t_rel, ref_rate):
    grid = []
    viable_cache = {}
    for sep in GRID_SEP:
        for lane in GRID_LANE:
            viable_cache[(sep, lane)] = any_by_snap(rf, win_default & viable_mask(rf, sep, lane, MAX_AWAY_SPEED), te)
    for tr_ in GRID_T_REL:
        trv = t_rel if tr_ is None else tr_
        rel_ok = te.is_route.values & (te.release_t.values <= trv + EPS)
        for sep in GRID_SEP:
            for lane in GRID_LANE:
                e = rel_ok & viable_cache[(sep, lane)]
                rho, npl = player_rank_corr(te, e, ref_rate)
                grid.append({"T_rel": trv, "min_sep": sep, "min_lane": lane,
                             "is_default": (tr_ is None and sep == VIABLE_MIN_SEP and lane == VIABLE_MIN_LANE),
                             "league_etr": round(float(e.mean()), 4),
                             "league_etr_route": round(float(e[te.is_route.values].mean()), 4),
                             "spearman_vs_default": round(rho, 4), "n_players": npl,
                             "target_lift": round(target_lift(te, e), 3)})
    rel_ok = te.is_route.values & (te.release_t.values <= t_rel + EPS)
    extra = []
    for away in GRID_AWAY:
        e = rel_ok & any_by_snap(rf, win_default & viable_mask(rf, away=away), te)
        rho, npl = player_rank_corr(te, e, ref_rate)
        extra.append({"variant": f"max_away_speed={away}", "league_etr": round(float(e.mean()), 4),
                      "league_etr_route": round(float(e[te.is_route.values].mean()), 4),
                      "spearman_vs_default": round(rho, 4), "n_players": npl,
                      "target_lift": round(target_lift(te, e), 3)})
    e = rel_ok & any_by_snap(rf, win_capall & viable_mask(rf), te)
    rho, npl = player_rank_corr(te, e, ref_rate)
    extra.append({"variant": "T_cap applied to throw plays too", "league_etr": round(float(e.mean()), 4),
                  "league_etr_route": round(float(e[te.is_route.values].mean()), 4),
                  "spearman_vs_default": round(rho, 4), "n_players": npl,
                  "target_lift": round(target_lift(te, e), 3)})
    return pd.DataFrame(grid), pd.DataFrame(extra)


# --------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p, te, rf, n_all = load()
    t_rel, t_cap, n_rel_ref, n_tt_ref = derive_thresholds(p, te)
    print(f"T_rel={t_rel}s (n={n_rel_ref})  T_cap={t_cap}s (n={n_tt_ref})")

    win = window_mask(rf, t_cap)
    win_capall = window_mask(rf, t_cap, cap_throws=True)
    vmask = win & viable_mask(rf)

    te["release_ok"] = te.is_route & (te.release_t <= t_rel + EPS)
    te["viable_any"] = any_by_snap(rf, vmask, te)
    te["eligible"] = te.release_ok & te.viable_any
    first = rf.loc[vmask].sort_values("t").groupby(KEY).head(1)[KEY + ["t", "sep", "lane_clear", "x_rel", "v_away"]]
    first = first.rename(columns={"t": "first_viable_t", "sep": "sep_first_viable",
                                  "lane_clear": "lane_first_viable", "x_rel": "x_rel_first_viable",
                                  "v_away": "v_away_first_viable"})
    te = te.merge(first, on=KEY, how="left")
    nwin = pd.DataFrame({"n_window_frames": win}).groupby([rf.gameId, rf.playId, rf.nflId]).n_window_frames.sum()
    te = te.merge(nwin.reset_index(), on=KEY, how="left")
    te["window_end_t"] = np.where(te.is_throw, te.time_to_throw - 0.1, np.minimum(te.time_to_end, t_cap))
    te.loc[~te.is_route, "window_end_t"] = np.nan

    # priors
    prior = {}
    for name, sub in [("all", te), ("route", te[te.is_route])]:
        kn = rate_table(sub, "eligible")
        s, m, tau2 = beta_prior_strength(kn.k, kn.n)
        prior[name] = {"mean": float(sub.eligible.mean()), "strength": s, "mom_mean_players_n50": float(m),
                       "between_player_var": tau2}

    # expected model
    pred, model_metrics = fit_expected(te)
    te["expected"] = pred.values
    te["resid"] = te.eligible.astype(float) - te.expected

    players = player_table(te, prior)
    val = validation(te)

    ref = te.groupby("nflId").eligible.mean()
    grid, extra = sensitivity(te, rf, win, win_capall, t_rel, ref)

    rel = {}
    for col, lab in [("eligible", "etr_all_snaps"), ("resid", "etr_oe")]:
        for mn in (10, 25):
            r, n = stats.split_half_reliability(te.assign(eligible=te.eligible.astype(float)), "nflId", col, min_n=mn)
            rel[f"{lab}_min{mn}"] = {"spearman": None if pd.isna(r) else round(float(r), 4), "n_players": int(n)}
    tr_ = te[te.is_route].assign(eligible=lambda d: d.eligible.astype(float))
    for mn in (10, 25):
        r, n = stats.split_half_reliability(tr_, "nflId", "eligible", min_n=mn)
        rel[f"etr_route_snaps_min{mn}"] = {"spearman": None if pd.isna(r) else round(float(r), 4), "n_players": int(n)}

    # ---------------- outputs
    play_cols = KEY + ["displayName", "possessionTeam", "week", "is_train", "assignment", "alignment",
                       "pff_positionLinedUp", "release_t", "release_ok", "viable_any", "eligible",
                       "first_viable_t", "sep_first_viable", "lane_first_viable", "x_rel_first_viable",
                       "v_away_first_viable", "n_window_frames", "window_end_t", "end_event",
                       "time_to_throw", "expected", "resid", "is_target", "is_catch"] + \
                [c for c in FEATURES if c != "alignment"]
    te[play_cols].to_csv(OUT_DIR / "plays.csv", index=False)
    players.to_csv(OUT_DIR / "players.csv", index=False, float_format="%.5g")
    grid.to_csv(OUT_DIR / "sensitivity_grid.csv", index=False)
    val.to_csv(OUT_DIR / "validation.csv", index=False)

    by_align = te.groupby("alignment").eligible.agg(["mean", "count"]).round(4)
    by_assign = te.groupby("assignment").agg(etr=("eligible", "mean"), n=("eligible", "size"),
                                             release_ok=("release_ok", "mean"),
                                             viable_any=("viable_any", "mean")).round(4)
    top = players[players.n >= MIN_N_RANK].head(5)
    summary = {
        "metric": "Eligible Threat Rate", "slug": SLUG, "higher_is_better": True,
        "definition": {
            "denominator": "every TE dropback snap with tracking (te_plays rows whose play has a snap frame)",
            "eligible_if_all": [
                "assignment in {route, chip_release}",
                f"release_t <= T_rel ({t_rel} s); never-released routes are not eligible",
                "at some frame from release through the decision window the TE is viable",
            ],
            "decision_window": {
                "throw_plays": "release_t <= t < time_to_throw (frames strictly before the throw frame; no T_cap)",
                "non_throw_plays": f"release_t <= t <= min(time_to_end, T_cap={t_cap} s)",
            },
            "viable_position": {
                "min_sep_yd": VIABLE_MIN_SEP, "min_lane_clear_yd": VIABLE_MIN_LANE,
                "x_rel_range_yd": list(VIABLE_DEPTH), "sideline_buffer_yd": VIABLE_SIDELINE_BUFFER,
                "route_direction_rule": (f"v_away <= {MAX_AWAY_SPEED} yd/s, v_away = receiver velocity (vx,vy) "
                                         "projected on the unit vector QB->receiver (QB motion not subtracted); "
                                         "i.e. the TE is settling, working back, or crossing, not running away from the QB"),
            },
            "value": "k_eligible / n over all snaps; value_shrunk = beta_shrink toward league rate",
            "etr_route": "conditional rate among route + chip_release snaps",
            "etr_oe": "mean(eligible - P(eligible | context)) per TE; P from cross-fitted HGB classifier",
            "release_definition": {"min_disp_yd": RELEASE_MIN_DISP, "min_speed": RELEASE_MIN_SPEED,
                                   "sustain_frames": RELEASE_SUSTAIN},
        },
        "thresholds": {
            "T_rel_s": t_rel, "T_rel_rule": f"train-week (weeks {TRAIN_WEEKS}) p{int(T_REL_PCT*100)} of TE release_t "
                                            f"on assignment=='route' snaps", "T_rel_ref_n": n_rel_ref,
            "T_cap_s": t_cap, "T_cap_rule": f"train-week p{int(T_CAP_PCT*100)} of plays.time_to_throw (throw plays)",
            "T_cap_ref_n": n_tt_ref, "max_away_speed_yd_s": MAX_AWAY_SPEED,
            "direction_rule_choice": ("1.0 yd/s suggested by the brief; checked on train weeks only: target-rate lift "
                                      "(eligible vs not, route throw snaps) ~1.35x at 1.0 vs ~1.38x at 0.5, ~1.11x at "
                                      "2.0 and ~1.14x with no direction rule; 0.5 was sparser for similar lift"),
            "min_n_rank": MIN_N_RANK,
        },
        "sample": {
            "te_rows_total": int(n_all), "te_rows_used": int(len(te)),
            "te_rows_dropped_no_tracking": int(n_all - len(te)),
            "route_snaps": int(te.is_route.sum()), "eligible_snaps": int(te.eligible.sum()),
            "n_tes": int(te.nflId.nunique()), "n_tes_min50": int((players.n >= MIN_N_RANK).sum()),
            "train_rows": int(te.is_train.sum()), "test_rows": int((~te.is_train).sum()),
            "test_weeks": TEST_WEEKS,
        },
        "league": {
            "etr_all_snaps": round(float(te.eligible.mean()), 4),
            "etr_route_snaps": round(float(te[te.is_route].eligible.mean()), 4),
            "release_ok_rate_route": round(float(te[te.is_route].release_ok.mean()), 4),
            "viable_any_rate_route": round(float(te[te.is_route].viable_any.mean()), 4),
            "median_first_viable_t": round(float(te[te.eligible].first_viable_t.median()), 2),
            "by_alignment": by_align.to_dict("index"), "by_assignment": by_assign.to_dict("index"),
        },
        "shrinkage_prior": prior,
        "expected_model": {"features": FEATURES, "target": "eligible (all TE snaps)",
                           "estimator": "HistGradientBoostingClassifier (categorical from dtype), "
                                        "cross_fit GroupKFold(gameId) on train weeks", **model_metrics},
        "validation_target_completion": val.to_dict("records"),
        "sensitivity_grid": grid.to_dict("records"),
        "sensitivity_extra": extra.to_dict("records"),
        "sensitivity_summary": {
            "league_etr_range": [float(grid.league_etr.min()), float(grid.league_etr.max())],
            "spearman_min": float(grid.spearman_vs_default.min()),
            "spearman_median": float(grid.spearman_vs_default.median()),
        },
        "split_half_reliability": rel,
        "top5_min50": top[["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi",
                           "etr_route", "etr_oe"]].round(4).to_dict("records"),
        "caveats": [
            "chip_release snaps rarely pass T_rel (median chip release ~1.8 s vs T_rel), so ETR penalizes chip duty by design; see etr_route/release_ok_rate_route components",
            "a single 0.1 s frame of viability suffices; no dwell-time requirement",
            "viability uses nearest-defender separation and a straight QB->TE lane only; no ball-flight or defender-closing-speed model",
            "direction rule excludes vertical stems while running (only settling/working-back/crossing frames qualify)",
            "throw-play window is uncapped, so long-developing throws give more chances to be viable",
            "assignment (route vs block) is a play-call choice and is NOT in the expected model; etr_oe partly reflects deployment",
            "test weeks are only 2 weeks, so split-half reliability is based on small test-week samples",
            f"{n_all - len(te)} TE rows on plays without a snap frame (no tracking) were dropped from the denominator",
            "validity is modest: target-rate lift of eligible vs non-eligible route snaps is "
            + ", ".join(f"{sp}: {val[(val.split == sp) & (val.group == 'eligible=True')].target_rate.iat[0]:.3f} vs "
                        f"{val[(val.split == sp) & (val.group == 'eligible=False')].target_rate.iat[0]:.3f}"
                        for sp in ("train", "test"))
            + " (direction rule was checked on train weeks; test-week lift does not replicate)",
            "T_rel=0.8 s cells cut most releases (median TE release 1.0 s) and reshuffle ranks (Spearman ~0.2-0.4)",
            "split-half reliability is low; treat player ETR as descriptive, not a stable trait",
        ],
        "shared_code_notes": [
            "plays.time_to_throw is NaN for throw plays lacking a snap frame; such plays are dropped here",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(json.dumps({k: summary[k] for k in ["thresholds", "sample", "league", "split_half_reliability",
                                              "sensitivity_summary"]}, indent=1, default=str))
    print(json.dumps({k: model_metrics[k] for k in ["model", "baseline_train_rate", "selected_config",
                                                    "calibration_test_5bin"]}, indent=1, default=str))
    print(val.to_string())
    print(extra.to_string())
    print(top[["displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "etr_route", "etr_oe"]].to_string())


if __name__ == "__main__":
    main()

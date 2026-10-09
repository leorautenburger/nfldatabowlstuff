"""Chip-to-Separation Return (CSR).

CSR (per TE chip-and-release play) =
    separation at the throw frame (routes.sep_end)
    - E[separation | release time, route depth, time from release to throw, context]

The expectation is fit on ALL TE route-running plays (assignment in {route, chip_release}) from the
train weeks with stats.cross_fit, WITHOUT an is_chip flag, so chip plays are judged against the
general TE expectation for the same release time / depth / throw timing / coverage context.

Run from project root:  .venv/bin/python -m src.metrics.chip_to_separation_return
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from src.common import io, stats  # noqa: E402
from src.common.config import FIELD_W, OUT, SEED, TEST_WEEKS, TRAIN_WEEKS  # noqa: E402

SLUG = "chip_to_separation_return"
OUT_DIR = OUT / SLUG
KEY = ["gameId", "playId", "nflId"]

MIN_CHIP_TOP = 8           # min chip plays for the top-5 / leaderboard flag
REL_MIN_N_CANDIDATES = [10, 6, 4]  # split-half reliability thresholds tried (chip plays per half)
PD_GRID = np.round(np.arange(0.5, 3.01, 0.25), 2)  # release_t grid for partial dependence
PD_MIN_T_END = 3.5         # PD population: plays whose throw came at t_end >= 3.5 s (so every grid
                           # point leaves >= 0.5 s between release and throw)

CAT_FEATS = ["alignment", "pff_passCoverageType", "pff_passCoverage", "dropBackType"]
NUM_FEATS = ["release_t", "x_rel_end", "max_depth", "t_rel_to_throw", "t_end",
             "receiver_num", "n_rec_side", "pff_playAction",
             "lat_from_qb_end", "lat_from_ball_end", "sideline_dist_end"]
GBM_FEATS = NUM_FEATS + CAT_FEATS
LIN_FEATS = ["release_t", "x_rel_end", "max_depth", "t_rel_to_throw"]

GBM_GRID = [
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
    dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0),
    dict(max_iter=250, learning_rate=0.05, max_depth=4, min_samples_leaf=60, l2_regularization=2.0),
]


# --------------------------------------------------------------------------------------------- data
def load_frame() -> pd.DataFrame:
    tp = io.te_plays()
    tp = tp[tp.assignment.isin(["route", "chip_release"])].copy()
    r = io.routes()[KEY + ["t_end", "y_end", "end_event", "released_flag"]]
    p = io.plays()[["gameId", "playId", "possessionTeam", "pff_passCoverageType", "pff_passCoverage",
                    "pff_playAction", "dropBackType"]]
    df = tp.merge(r, on=KEY, how="left").merge(p, on=["gameId", "playId"], how="left")

    # QB position at the throw/end frame (lateral offset of TE from the passer at the throw).
    rf = io.receiver_frames(columns=["gameId", "playId", "nflId", "t", "qb_y"],
                            filters=[("nflId", "in", df.nflId.unique().tolist())])
    rf = rf.merge(df[KEY + ["t_end"]], on=KEY, how="inner")
    rf = rf[np.isclose(rf.t, rf.t_end, atol=1e-6)].drop_duplicates(KEY)[KEY + ["qb_y"]]
    df = df.merge(rf, on=KEY, how="left")

    ball_y = df.ball_y.where(df.ball_y.between(0, FIELD_W))  # a few ball_y values lie off the field
    df["lat_from_qb_end"] = (df.y_end - df.qb_y).abs()
    df["lat_from_ball_end"] = (df.y_end - ball_y).abs()
    df["sideline_dist_end"] = np.minimum(df.y_end, FIELD_W - df.y_end)
    df["t_rel_to_throw"] = df.t_end - df.release_t
    df["n_rec_side"] = df.n_rec_side.astype(float)
    df["pff_playAction"] = df.pff_playAction.astype(float)
    for c in CAT_FEATS:
        df[c] = df[c].astype("category")
    df["is_chip"] = df.assignment.eq("chip_release")
    df["has_tracking"] = df.t_end.notna() & df.sep_end.notna()
    df["is_throw"] = df.end_event.eq("throw")
    df["released"] = df.release_t.notna()
    return df.reset_index(drop=True)


# -------------------------------------------------------------------------------------------- model
def reg_metrics(y, yhat):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, yhat))), mae=float(mean_absolute_error(y, yhat)),
                r2=float(r2_score(y, yhat)), n=int(len(y)))


def test_report(df, pred_col, base):
    te = df[~df.is_train]
    out = {"all_te_routes": reg_metrics(te.sep_end, te[pred_col]),
           "chip_only": reg_metrics(te[te.is_chip].sep_end, te[te.is_chip][pred_col]),
           "route_only": reg_metrics(te[~te.is_chip].sep_end, te[~te.is_chip][pred_col])}
    if base is not None:
        out["baseline_all"] = reg_metrics(te.sep_end, np.full(len(te), base))
        out["baseline_chip"] = reg_metrics(te[te.is_chip].sep_end, np.full(te.is_chip.sum(), base))
    return out


def fit_models(m: pd.DataFrame):
    """Tune GBM on train OOF only, cross-fit GBM + linear, return preds and diagnostics."""
    tr = m[m.is_train]
    tuning = []
    best = None
    for i, params in enumerate(GBM_GRID):
        gbm = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED, **params)
        pred, full = stats.cross_fit(gbm, m, GBM_FEATS, "sep_end")
        oof_rmse = float(np.sqrt(mean_squared_error(tr.sep_end, pred[tr.index])))
        tuning.append({"params": params, "train_oof_rmse": oof_rmse})
        if best is None or oof_rmse < best[0]:
            best = (oof_rmse, i, pred, full)
    gbm_pred, gbm_full = best[2], best[3]
    gbm_params = GBM_GRID[best[1]]

    lin = LinearRegression()
    lin_pred, lin_full = stats.cross_fit(lin, m, LIN_FEATS, "sep_end")
    return gbm_pred, gbm_full, gbm_params, tuning, lin_pred, lin_full


def partial_dependence_release(model, pop: pd.DataFrame):
    """Average prediction as release_t varies with throw time held fixed (t_rel_to_throw adjusts)."""
    rows = []
    for v in PD_GRID:
        X = pop[GBM_FEATS].copy()
        X["release_t"] = v
        X["t_rel_to_throw"] = pop.t_end - v
        rows.append({"release_t": float(v), "exp_sep": float(model.predict(X).mean())})
    out = pd.DataFrame(rows)
    out["delta_vs_first"] = out.exp_sep - out.exp_sep.iloc[0]
    return out


def safe_split_half(df, player_col, value_col, half_col="is_train", min_n=10):
    """Workaround for stats.split_half_reliability: it raises KeyError when no player reaches min_n in
    one of the halves (pivot then lacks that column). Return (nan, n_both_halves) in that case."""
    g = df.groupby([player_col, half_col])[value_col].count().unstack()
    n_both = int(((g.get(True, 0) >= min_n) & (g.get(False, 0) >= min_n)).sum()) if len(g) else 0
    if n_both < 5:
        return np.nan, n_both
    return stats.split_half_reliability(df, player_col, value_col, half_col=half_col, min_n=min_n)


# ------------------------------------------------------------------------------------------- player
def player_table(d: pd.DataFrame, resid: str, pooled_sd_chip: float, pooled_sd_route: float):
    chip = d[d.is_chip]
    route = d[~d.is_chip]
    g = chip.groupby("nflId")[resid].agg(n="count", value="mean", sd="std")
    g["value_shrunk"] = stats.eb_shrink_mean(g.value, g.n, np.full(len(g), pooled_sd_chip))
    se = pooled_sd_chip / np.sqrt(g.n)
    g["lo"], g["hi"] = g.value - 1.96 * se, g.value + 1.96 * se
    extras = chip.groupby("nflId").agg(chip_sep_mean=("sep_end", "mean"), chip_exp_sep_mean=("exp_sep", "mean"),
                                       chip_release_t_mean=("release_t", "mean"),
                                       chip_depth_mean=("x_rel_end", "mean"),
                                       chip_targets=("is_target", "sum"), chip_catches=("is_catch", "sum"))
    rg = route.groupby("nflId")[resid].agg(n_route="count", route_resid_mean="mean")
    out = g.join(extras).join(rg, how="left")
    out["route_resid_shrunk"] = np.nan
    ok = out.n_route.fillna(0) > 0
    out.loc[ok, "route_resid_shrunk"] = stats.eb_shrink_mean(out.loc[ok, "route_resid_mean"], out.loc[ok, "n_route"],
                                                             np.full(ok.sum(), pooled_sd_route))
    out["chip_minus_route"] = out.value - out.route_resid_mean
    se_diff = np.sqrt(pooled_sd_chip**2 / out.n + pooled_sd_route**2 / out.n_route)
    out["chip_minus_route_lo"] = out.chip_minus_route - 1.96 * se_diff
    out["chip_minus_route_hi"] = out.chip_minus_route + 1.96 * se_diff
    out["chip_target_rate"] = out.chip_targets / out.n
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    counts = {
        "te_route_running_plays": int(len(df)),
        "chip_release_plays": int(df.is_chip.sum()),
        "route_plays": int((~df.is_chip).sum()),
        "missing_route_tracking": int((~df.has_tracking).sum()),
        "chip_missing_tracking": int((df.is_chip & ~df.has_tracking).sum()),
        "never_released_chip": int((df.is_chip & df.has_tracking & ~df.released).sum()),
        "never_released_route": int((~df.is_chip & df.has_tracking & ~df.released).sum()),
        "released_chip_no_throw_(sack/scramble)": int((df.is_chip & df.has_tracking & df.released & ~df.is_throw).sum()),
        "released_route_no_throw_(sack/scramble)": int((~df.is_chip & df.has_tracking & df.released & ~df.is_throw).sum()),
    }

    # Primary modelling set: released, tracked, play ended in a throw (sep_end is the throw frame).
    base_mask = df.has_tracking & df.released
    m = df[base_mask & df.is_throw].copy()
    counts["model_rows"] = int(len(m))
    counts["model_rows_train"] = int(m.is_train.sum())
    counts["model_rows_test"] = int((~m.is_train).sum())
    counts["chip_csr_plays"] = int(m.is_chip.sum())
    counts["chip_csr_plays_train"] = int((m.is_chip & m.is_train).sum())
    counts["chip_csr_plays_test"] = int((m.is_chip & ~m.is_train).sum())

    gbm_pred, gbm_full, gbm_params, tuning, lin_pred, lin_full = fit_models(m)
    m["exp_sep_gbm"], m["exp_sep_lin"] = gbm_pred, lin_pred
    base = float(m[m.is_train].sep_end.mean())
    rep_gbm = test_report(m, "exp_sep_gbm", base)
    rep_lin = test_report(m, "exp_sep_lin", None)
    primary = "gbm" if rep_gbm["all_te_routes"]["rmse"] < rep_lin["all_te_routes"]["rmse"] else "linear"
    m["exp_sep"] = m.exp_sep_gbm if primary == "gbm" else m.exp_sep_lin
    m["csr"] = m.sep_end - m.exp_sep
    m["resid_gbm"] = m.sep_end - m.exp_sep_gbm
    m["resid_lin"] = m.sep_end - m.exp_sep_lin

    # ---- league level
    chip, route = m[m.is_chip], m[~m.is_chip]
    league = {
        "chip_mean_csr": stats.mean_ci(chip.csr).to_dict(),
        "chip_mean_csr_test_weeks": stats.mean_ci(chip[~chip.is_train].csr).to_dict(),
        "route_mean_resid": stats.mean_ci(route.csr).to_dict(),
        "chip_mean_sep": float(chip.sep_end.mean()), "chip_mean_exp_sep": float(chip.exp_sep.mean()),
        "route_mean_sep": float(route.sep_end.mean()), "route_mean_exp_sep": float(route.exp_sep.mean()),
        "chip_mean_release_t": float(chip.release_t.mean()), "route_mean_release_t": float(route.release_t.mean()),
        "chip_mean_depth_x_rel_end": float(chip.x_rel_end.mean()),
        "route_mean_depth_x_rel_end": float(route.x_rel_end.mean()),
        "chip_csr_by_blocktype": m[m.is_chip].groupby("pff_blockType").csr.agg(["mean", "count"]).round(3)
        .to_dict(orient="index"),
        "chip_csr_by_alignment": m[m.is_chip].groupby("alignment", observed=True).csr.agg(["mean", "count"])
        .round(3).to_dict(orient="index"),
    }

    # Target / catch rates (all released chip / route plays that ended in a throw).
    def rates(d):
        nt, nc, n = int(d.is_target.sum()), int(d.is_catch.sum()), len(d)
        tlo, thi = stats.wilson_ci(nt, n)
        clo, chi = stats.wilson_ci(nc, nt)
        return {"n": n, "targets": nt, "catches": nc, "target_rate": nt / n, "target_rate_ci": [float(tlo), float(thi)],
                "catch_rate_given_target": nc / nt if nt else None, "catch_rate_ci": [float(clo), float(chi)]}
    all_chip_tracked = df[df.is_chip]
    league["rates"] = {
        "chip_all_assignments_incl_unreleased_and_sacks": rates(all_chip_tracked),
        "chip_csr_plays": rates(chip), "route_plays": rates(route),
        "chip_csr_positive": rates(chip[chip.csr > 0]), "chip_csr_nonpositive": rates(chip[chip.csr <= 0]),
        "route_resid_positive": rates(route[route.csr > 0]), "route_resid_nonpositive": rates(route[route.csr <= 0]),
    }

    # Partial dependence of release_t (throw time held fixed), train-fit full model.
    pd_chip = partial_dependence_release(gbm_full, chip[chip.t_end >= PD_MIN_T_END])
    pd_all = partial_dependence_release(gbm_full, m[m.t_end >= PD_MIN_T_END])
    pd_out = pd_chip.rename(columns={"exp_sep": "exp_sep_chip_pop", "delta_vs_first": "delta_chip_pop"}).merge(
        pd_all.rename(columns={"exp_sep": "exp_sep_all_pop", "delta_vs_first": "delta_all_pop"}), on="release_t")
    pd_out.to_csv(OUT_DIR / "release_t_partial_dependence.csv", index=False)
    lin_coefs = dict(zip(LIN_FEATS, map(float, lin_full.coef_)))
    lin_coefs["intercept"] = float(lin_full.intercept_)
    league["release_t_partial_dependence"] = {
        "note": f"GBM, release_t varied with t_end fixed (t_rel_to_throw = t_end - release_t); populations "
                f"restricted to t_end >= {PD_MIN_T_END}s",
        "n_chip_pop": int((chip.t_end >= PD_MIN_T_END).sum()), "n_all_pop": int((m.t_end >= PD_MIN_T_END).sum()),
        "table": pd_out.round(3).to_dict(orient="records"),
        "chip_pop_change_per_sec_0.5_to_3.0": float((pd_chip.exp_sep.iloc[-1] - pd_chip.exp_sep.iloc[0]) / 2.5),
        "all_pop_change_per_sec_0.5_to_3.0": float((pd_all.exp_sep.iloc[-1] - pd_all.exp_sep.iloc[0]) / 2.5),
        "linear_model_coefficients": lin_coefs,
        "linear_note": "linear: one extra second of release delay at fixed throw time changes E[sep] by "
                       "coef(release_t) - coef(t_rel_to_throw)",
        "linear_delay_effect_per_sec_fixed_throw": lin_coefs["release_t"] - lin_coefs["t_rel_to_throw"],
    }

    # ---- player table
    sd_chip, sd_route = float(chip.csr.std()), float(route.csr.std())
    pt = player_table(m, "csr", sd_chip, sd_route)
    names = df.groupby("nflId").agg(displayName=("displayName", "first"),
                                    team=("possessionTeam", lambda s: s.mode().iat[0]))
    pt = names.join(pt, how="inner").reset_index()
    pt["qualifies_top"] = pt.n >= MIN_CHIP_TOP
    pt["higher_is_better"] = True
    # Linear-model version for comparison
    lin_pt = player_table(m.assign(exp_sep=m.exp_sep_lin), "resid_lin",
                          float(chip.resid_lin.std()), float(route.resid_lin.std()))
    pt = pt.merge(lin_pt[["value", "value_shrunk"]].rename(columns={"value": "value_linear",
                                                                    "value_shrunk": "value_shrunk_linear"}),
                  left_on="nflId", right_index=True, how="left")
    pt = pt.sort_values(["qualifies_top", "value_shrunk"], ascending=[False, False])
    cols = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "higher_is_better",
            "qualifies_top", "chip_sep_mean", "chip_exp_sep_mean", "chip_release_t_mean", "chip_depth_mean",
            "chip_targets", "chip_catches", "chip_target_rate", "n_route", "route_resid_mean",
            "route_resid_shrunk", "chip_minus_route", "chip_minus_route_lo", "chip_minus_route_hi",
            "value_linear", "value_shrunk_linear", "sd"]
    pt[cols].to_csv(OUT_DIR / "players.csv", index=False, float_format="%.4f")

    # ---- reliability (chip residuals train weeks vs test weeks)
    reliability = {}
    for k in REL_MIN_N_CANDIDATES:
        rho, n = safe_split_half(chip, "nflId", "csr", min_n=k)
        reliability[f"chip_csr_min_n_{k}"] = {"spearman": None if pd.isna(rho) else float(rho), "n_players": int(n)}
    for k in [10, 20]:
        rho, n = safe_split_half(route, "nflId", "csr", min_n=k)
        reliability[f"route_resid_min_n_{k}"] = {"spearman": None if pd.isna(rho) else float(rho), "n_players": int(n)}
    # Odd/even-game split within all weeks (more chip plays per half than train/test weeks)
    ch = chip.copy()
    gid_rank = ch.groupby("nflId").gameId.rank(method="dense")
    ch["half"] = (gid_rank % 2 == 0)
    for k in [6, 4]:
        rho, n = safe_split_half(ch, "nflId", "csr", half_col="half", min_n=k)
        reliability[f"chip_csr_oddeven_games_min_n_{k}"] = {"spearman": None if pd.isna(rho) else float(rho),
                                                            "n_players": int(n)}

    # ---- sensitivity
    sens = {}
    q = pt[pt.qualifies_top]
    sens["gbm_vs_linear_player_csr_spearman_qualified"] = float(q.value.corr(q.value_linear, method="spearman"))
    sens["gbm_vs_linear_league_chip_mean"] = {"gbm": float(chip.resid_gbm.mean()), "linear": float(chip.resid_lin.mean())}
    # include sack/scramble plays (separation at end of dropback instead of throw)
    m2 = df[base_mask].copy()
    gbm2 = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED, **gbm_params)
    p2, _ = stats.cross_fit(gbm2, m2, GBM_FEATS, "sep_end")
    m2["csr"] = m2.sep_end - p2
    c2 = m2[m2.is_chip]
    g2 = c2.groupby("nflId").csr.agg(["mean", "count"])
    j = q.set_index("nflId")[["value"]].join(g2)
    sens["incl_sack_scramble_plays"] = {"n_chip": int(len(c2)), "chip_mean_csr": float(c2.csr.mean()),
                                        "player_spearman_vs_primary_qualified": float(j.value.corr(j["mean"], method="spearman"))}
    # different min-n for leaderboard
    sens["top5_by_min_n"] = {str(k): pt[pt.n >= k].sort_values("value_shrunk", ascending=False)
                             .head(5).displayName.tolist() for k in [5, 8, 12]}
    sens["n_players_by_min_n"] = {str(k): int((pt.n >= k).sum()) for k in [1, 4, 5, 8, 12]}

    # ---- play output
    play_cols = ["gameId", "playId", "nflId", "displayName", "possessionTeam", "week", "is_train", "assignment",
                 "pff_blockType", "alignment", "release_t", "t_end", "t_rel_to_throw", "x_rel_end", "max_depth",
                 "sep_end", "exp_sep_gbm", "exp_sep_lin", "exp_sep", "csr", "is_target", "is_catch"]
    m.sort_values(["is_chip", "nflId", "gameId", "playId"], ascending=[False, True, True, True])[play_cols] \
        .rename(columns={"csr": "residual"}).to_csv(OUT_DIR / "plays.csv", index=False, float_format="%.4f")

    top5 = pt[pt.qualifies_top].head(5)[["displayName", "team", "n", "value", "value_shrunk", "lo", "hi",
                                         "route_resid_mean", "chip_minus_route"]]
    summary = {
        "metric": "Chip-to-Separation Return (CSR)", "slug": SLUG,
        "definition": "Per TE chip_release play: sep_end (nearest-defender distance at the throw frame) minus "
                      "expected sep_end from a model fit on all TE route-running plays (route + chip_release) "
                      "with no chip indicator. Player CSR = mean residual over the TE's chip_release plays.",
        "plays_included": "TE assignment chip_release (PFF Pass Route with blockType CH or SR) or route; released "
                          "(release_t not NaN); play ended in a throw (end_event == 'throw'); tracking present.",
        "exclusions": counts,
        "features": {"gbm": GBM_FEATS, "gbm_categorical": CAT_FEATS, "linear": LIN_FEATS,
                     "derived": {"t_rel_to_throw": "t_end - release_t",
                                 "lat_from_qb_end": "|y_end - QB y at throw frame|",
                                 "lat_from_ball_end": "|y_end - ball_y at snap| (off-field ball_y set NaN)",
                                 "sideline_dist_end": "min(y_end, 53.3 - y_end)"}},
        "parameters": {"train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS, "min_chip_top": MIN_CHIP_TOP,
                       "gbm_grid": GBM_GRID, "gbm_selected": gbm_params, "pd_grid": PD_GRID.tolist(),
                       "pd_min_t_end": PD_MIN_T_END,
                       "shrinkage": "normal-normal EB (stats.eb_shrink_mean) with pooled chip-residual SD",
                       "ci": "mean +/- 1.96 * pooled_sd / sqrt(n) (pooled SD because per-player chip n is small)",
                       "pooled_sd_chip_resid": sd_chip, "pooled_sd_route_resid": sd_route,
                       "eb_tau2_chip_implied": float(max(np.var(pt.value) - np.mean(sd_chip**2 / pt.n), 1e-6))},
        "model": {"primary": primary, "gbm_tuning_train_oof": tuning, "test_gbm": rep_gbm, "test_linear": rep_lin,
                  "train_mean_baseline": base},
        "league": league,
        "reliability": reliability,
        "sensitivity": sens,
        "top5_min_n": top5.round(3).to_dict(orient="records"),
        "caveats": [
            "Separation = distance to nearest defender at the throw frame; ball flight/catch point is untracked.",
            "Chip plays are scarce per TE (43 TEs with >= 8); CIs are wide. Implied between-player variance "
            "(tau2 ~0.02 yd^2) is tiny vs sampling noise, so EB shrinks nearly everyone to the league chip mean; "
            "rank on value_shrunk is weakly informative and split-half reliability is low.",
            "Model includes no chip flag by design, so league mean CSR on chips measures the chip-vs-general-TE "
            "difference after context; it may partly reflect route types chips lead to (flats/checkdowns) that "
            "depth/lateral features do not fully capture.",
            "chip_release relies on PFF blockType CH/SR on a Pass Route role; release time uses the shared "
            "1.5 yd / 2.5 yd/s / 3-frame rule.",
            "Sack/scramble plays (no throw frame) are excluded from the primary metric; see sensitivity.",
            "dropBackType includes post-snap QB behaviour (scramble), kept per spec as non-player context.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- console
    print(json.dumps(counts, indent=1))
    print("primary:", primary, "| selected gbm:", gbm_params)
    for name, rep in [("GBM", rep_gbm), ("LIN", rep_lin)]:
        print(name, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rep.items()})
    print("league chip CSR:", {k: round(v, 3) for k, v in league["chip_mean_csr"].items()})
    print("route resid:", {k: round(v, 3) for k, v in league["route_mean_resid"].items()})
    print("rates:", json.dumps(league["rates"], indent=1, default=float))
    print(pd_out.round(3).to_string(index=False))
    print("linear delay effect/sec:", round(league["release_t_partial_dependence"]["linear_delay_effect_per_sec_fixed_throw"], 3))
    print("reliability:", reliability)
    print("sensitivity:", json.dumps(sens, indent=1, default=float))
    print(top5.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

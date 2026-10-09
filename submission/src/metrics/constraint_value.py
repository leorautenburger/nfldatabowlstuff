"""Constraint Value: how TE alignment changes the pre-snap defense.

Pipeline
  1. Defensive outcomes per play: defendersInBox (PFF), deep-defender count / two-high / deepest
     depth from defender tracking at the snap frame (t == 0), man coverage, PFF two-high shell,
     blitz (n_rushers >= 5).
  2. Expected outcome from context only (down, distance, field position, quarter, score
     differential, seconds left in half, RB / WR counts from personnelO, detached non-TE receivers).
     No TE count / TE alignment. HistGradientBoosting, cross-fit on train weeks (stats.cross_fit).
  3. Residual = observed - expected.
       League : mean residual on TE-inline / TE-detached / no-TE plays (game-bootstrap CIs).
       Player : box-count residual on the TE's inline snaps minus on his detached snaps.
                Components: same difference for n_deep, two_high, man.

Run: .venv/bin/python -m src.metrics.constraint_value
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
from src.common.config import OUT, RAW, SEED

SLUG = "constraint_value"
OUT_DIR = OUT / SLUG

# ---------------------------------------------------------------- parameters
DEEP_X = 10.0                 # yards past LOS at snap -> "deep" defender
TWO_HIGH_MIN = 2              # >= 2 deep defenders -> two-high
BLITZ_MIN = 5                 # n_rushers >= 5
PFF_TWO_HIGH = ["Cover-2", "2-Man", "Quarters", "Cover-6"]
INLINE_ALIGN = ["inline", "wing"]
MIN_TOP = 25                  # min snaps in EACH state (inline, detached) for top-5
MIN_REL = 10                  # min snaps per state per half for reliability
N_BOOT = 1000

BASE_FEATS = ["down", "yardsToGo", "yards_to_goal", "quarter", "score_diff", "secs_left_half",
              "n_rb", "n_wr", "n_detached_nonte"]
SENS_FEATS = BASE_FEATS + ["formation_code"]

OUTCOMES = {  # name: kind
    "box": "reg", "n_deep": "reg", "deepest": "reg",
    "two_high": "clf", "man": "clf", "pff_two_high": "clf", "blitz": "clf",
}
COMPONENTS = ["n_deep", "two_high", "man"]

HGB_KW = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
              l2_regularization=1.0, random_state=SEED)


# ---------------------------------------------------------------- data
def _clock_secs(s):
    mm, ss = s.str.split(":", expand=True).astype(float).T.values
    return mm * 60 + ss


def build_plays():
    p = io.plays()
    g = pd.read_csv(RAW / "games.csv", usecols=["gameId", "homeTeamAbbr"]) \
        if "homeTeamAbbr" not in p.columns else None
    if g is not None:
        p = p.merge(g, on="gameId", how="left")
    home = p["possessionTeam"] == p["homeTeamAbbr"]
    p["score_diff"] = np.where(home, p["preSnapHomeScore"] - p["preSnapVisitorScore"],
                               p["preSnapVisitorScore"] - p["preSnapHomeScore"])
    clk = _clock_secs(p["gameClock"])
    p["secs_left_half"] = np.where(p["quarter"].isin([1, 3]), clk + 900, clk)
    pers = p["personnelO"].fillna("")
    p["n_rb"] = pers.str.extract(r"(\d+) RB")[0].astype(float).fillna(0)
    p["n_wr"] = pers.str.extract(r"(\d+) WR")[0].astype(float).fillna(0)
    p["formation_code"] = p["offenseFormation"].astype("category").cat.codes.replace(-1, np.nan)

    # detached non-TE receivers (WR / RB / FB lined up detached at snap)
    pp = io.player_plays()[["gameId", "playId", "officialPosition", "alignment"]]
    det = pp[(pp["alignment"] == "detached") & pp["officialPosition"].isin(["WR", "RB", "FB"])]
    nd = det.groupby(["gameId", "playId"]).size().rename("n_detached_nonte")
    p = p.merge(nd, on=["gameId", "playId"], how="left")
    p["n_detached_nonte"] = p["n_detached_nonte"].fillna(0)

    # snap-frame defender depth
    tr = io.tracking(columns=["gameId", "playId", "x_rel"],
                     filters=[("t", "==", 0.0), ("is_def", "==", True)])
    d = tr.groupby(["gameId", "playId"])["x_rel"].agg(
        n_deep=lambda x: int((x >= DEEP_X).sum()), deepest="max", n_def="size").reset_index()
    p = p.merge(d, on=["gameId", "playId"], how="left")

    p["box"] = p["defendersInBox"]
    p["two_high"] = (p["n_deep"] >= TWO_HIGH_MIN).astype(float).where(p["n_deep"].notna())
    p["man"] = (p["pff_passCoverageType"] == "Man").astype(float)
    p["pff_two_high"] = p["pff_passCoverage"].isin(PFF_TWO_HIGH).astype(float)
    p["blitz"] = (p["n_rushers"] >= BLITZ_MIN).astype(float)
    return p, pp


def te_states(p):
    te = io.te_plays()[["gameId", "playId", "nflId", "displayName", "alignment", "is_train"]]
    te = te.copy()
    te["state"] = np.select([te["alignment"].isin(INLINE_ALIGN), te["alignment"] == "detached",
                             te["alignment"] == "backfield"], ["inline", "detached", "backfield"],
                            "other")
    f = te.pivot_table(index=["gameId", "playId"], columns="state", values="nflId",
                       aggfunc="size", fill_value=0)
    for c in ["inline", "detached", "backfield"]:
        if c not in f:
            f[c] = 0
    f = f.add_prefix("n_te_").reset_index()
    p = p.merge(f, on=["gameId", "playId"], how="left")
    for c in ["n_te_inline", "n_te_detached", "n_te_backfield"]:
        p[c] = p[c].fillna(0)
    inl, det = p["n_te_inline"] > 0, p["n_te_detached"] > 0
    p["play_state"] = np.select(
        [p["n_te_on_field"] == 0, inl & ~det, det & ~inl, inl & det],
        ["no_te", "te_inline", "te_detached", "te_mixed"], "te_backfield_only")
    return p, te


# ---------------------------------------------------------------- models
def _reg_metrics(y, p, base):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.full_like(y, base)
    f = lambda q: {"rmse": float(np.sqrt(mean_squared_error(y, q))),
                   "mae": float(mean_absolute_error(y, q)), "r2": float(r2_score(y, q))}
    return {"n": int(len(y)), "model": f(p), "baseline_train_mean": f(b)}


def _clf_metrics(y, p, base):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    b = np.full(len(y), base)
    cal = pd.DataFrame({"p": p, "y": y})
    cal["bin"] = pd.qcut(cal["p"], 5, labels=False, duplicates="drop")
    cal = cal.groupby("bin").agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
    return {"n": int(len(y)), "base_rate_test": float(y.mean()),
            "model": {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
                      "brier": float(brier_score_loss(y, p))},
            "baseline_train_rate": {"auc": 0.5, "log_loss": float(log_loss(y, b, labels=[0, 1])),
                                    "brier": float(brier_score_loss(y, b))},
            "calibration_5bin": cal.round(4).reset_index().to_dict(orient="records")}


def fit_expected(p, target, kind, feats):
    sub = p[p[target].notna()]
    if kind == "reg":
        m = HistGradientBoostingRegressor(**HGB_KW)
        pred, _ = stats.cross_fit(m, sub, feats, target)
    else:
        m = HistGradientBoostingClassifier(**HGB_KW)
        pred, _ = stats.cross_fit(m, sub, feats, target, proba=True)
    tr, te = sub["is_train"], ~sub["is_train"]
    base = sub.loc[tr, target].mean()
    met = (_reg_metrics if kind == "reg" else _clf_metrics)(sub.loc[te, target], pred[te], base)
    out = pd.Series(np.nan, index=p.index)
    out.loc[sub.index] = pred
    return out, met


# ---------------------------------------------------------------- bootstrap helpers
def _boot_weights(n_games, rng):
    return rng.multinomial(n_games, np.full(n_games, 1 / n_games), size=N_BOOT)


def league_effects(p, res_cols, rng):
    """Mean residual by play_state with game-bootstrap 95% CI, plus inline - detached and
    TE-present - no-TE contrasts."""
    games = np.sort(p["gameId"].unique())
    W = _boot_weights(len(games), rng)
    out = {}
    for col in res_cols:
        d = p[p[col].notna()]
        S = d.pivot_table(index="gameId", columns="play_state", values=col, aggfunc="sum",
                          fill_value=0).reindex(games, fill_value=0)
        N = d.pivot_table(index="gameId", columns="play_state", values=col, aggfunc="count",
                          fill_value=0).reindex(games, fill_value=0)
        d2 = d.assign(te_present=np.where(d["play_state"] == "no_te", "no_te", "te_present"))
        S2 = d2.pivot_table(index="gameId", columns="te_present", values=col, aggfunc="sum",
                            fill_value=0).reindex(games, fill_value=0)
        N2 = d2.pivot_table(index="gameId", columns="te_present", values=col, aggfunc="count",
                            fill_value=0).reindex(games, fill_value=0)
        res, boots = {}, {}
        for s in S.columns:
            pt = S[s].sum() / N[s].sum()
            bs = (W @ S[s].values) / np.maximum(W @ N[s].values, 1)
            boots[s] = bs
            res[s] = {"mean": float(pt), "lo": float(np.percentile(bs, 2.5)),
                      "hi": float(np.percentile(bs, 97.5)), "n": int(N[s].sum())}
        if "te_inline" in res and "te_detached" in res:
            diff = res["te_inline"]["mean"] - res["te_detached"]["mean"]
            bd = boots["te_inline"] - boots["te_detached"]
            res["inline_minus_detached"] = {"mean": float(diff), "lo": float(np.percentile(bd, 2.5)),
                                            "hi": float(np.percentile(bd, 97.5))}
        bp = (W @ S2["te_present"].values) / np.maximum(W @ N2["te_present"].values, 1)
        bn = (W @ S2["no_te"].values) / np.maximum(W @ N2["no_te"].values, 1)
        pt = S2["te_present"].sum() / N2["te_present"].sum() - S2["no_te"].sum() / N2["no_te"].sum()
        res["te_present_minus_no_te"] = {"mean": float(pt),
                                         "lo": float(np.percentile(bp - bn, 2.5)),
                                         "hi": float(np.percentile(bp - bn, 97.5))}
        out[col] = res
    return out


def player_table(te, rng):
    rows = []
    d = te[te["state"].isin(["inline", "detached"])]
    for pid, g in d.groupby("nflId"):
        games = g["gameId"].unique()
        gi = pd.Index(games)
        r = {"nflId": pid, "displayName": g["displayName"].iloc[0],
             "team": g["possessionTeam"].mode().iloc[0],
             "n_inline": int((g["state"] == "inline").sum()),
             "n_detached": int((g["state"] == "detached").sum()),
             "n_games": len(games)}
        W = _boot_weights(len(games), rng) if len(games) > 1 else None
        for col in ["res_box"] + [f"res_{c}" for c in COMPONENTS]:
            gg = g[g[col].notna()]
            S = gg.pivot_table(index="gameId", columns="state", values=col, aggfunc="sum",
                               fill_value=0).reindex(gi, fill_value=0)
            N = gg.pivot_table(index="gameId", columns="state", values=col, aggfunc="count",
                               fill_value=0).reindex(gi, fill_value=0)
            name = "value" if col == "res_box" else f"{col[4:]}_diff"
            if "inline" not in S or "detached" not in S or N["inline"].sum() == 0 \
                    or N["detached"].sum() == 0:
                r[name] = np.nan
                if col == "res_box":
                    r["box_res_inline"] = S["inline"].sum() / N["inline"].sum() \
                        if "inline" in S and N["inline"].sum() else np.nan
                    r["box_res_detached"] = S["detached"].sum() / N["detached"].sum() \
                        if "detached" in S and N["detached"].sum() else np.nan
                    r["lo"] = r["hi"] = np.nan
                continue
            mi = S["inline"].sum() / N["inline"].sum()
            md = S["detached"].sum() / N["detached"].sum()
            r[name] = mi - md
            if col == "res_box":
                r["box_res_inline"], r["box_res_detached"] = mi, md
                if W is not None:
                    ni, nd = W @ N["inline"].values, W @ N["detached"].values
                    ok = (ni > 0) & (nd > 0)
                    bs = (W @ S["inline"].values)[ok] / ni[ok] - (W @ S["detached"].values)[ok] / nd[ok]
                    r["lo"], r["hi"] = (np.percentile(bs, [2.5, 97.5]) if ok.sum() > 50
                                        else (np.nan, np.nan))
                else:
                    r["lo"] = r["hi"] = np.nan
        rows.append(r)
    pl = pd.DataFrame(rows)
    pl["n"] = pl["n_inline"] + pl["n_detached"]

    # EB shrinkage of the difference: effective n = harmonic combination of the two states
    sd = stats.pooled_within_sd(d.assign(k=d["nflId"].astype(str) + d["state"]), "k", "res_box")
    n_eff = 1 / (1 / pl["n_inline"].clip(lower=1) + 1 / pl["n_detached"].clip(lower=1))
    n_eff = n_eff.where(pl["value"].notna(), 0)
    pl["n_eff"] = n_eff
    pl["value_shrunk"] = np.where(pl["value"].notna(),
                                  stats.eb_shrink_mean(pl["value"].values, n_eff.values, sd), np.nan)
    pl["higher_is_better"] = True
    pl["top_eligible"] = (pl["n_inline"] >= MIN_TOP) & (pl["n_detached"] >= MIN_TOP)
    return pl, sd


def _diff_reliability(te, half_col):
    """Spearman of per-player (inline - detached) box residual between two halves."""
    d = te[te["state"].isin(["inline", "detached"])]
    g = d.groupby(["nflId", half_col, "state"])["res_box"].agg(["mean", "count"]).unstack("state")
    ok = (g[("count", "inline")] >= MIN_REL) & (g[("count", "detached")] >= MIN_REL)
    v = (g[("mean", "inline")] - g[("mean", "detached")])[ok].unstack(half_col).dropna()
    if v.shape[1] < 2 or len(v) < 5:
        return {"rho": None, "n_players": int(len(v))}
    return {"rho": float(sps.spearmanr(v.iloc[:, 0], v.iloc[:, 1]).correlation),
            "n_players": int(len(v))}


def reliability(te):
    t = te.copy()
    t["_odd"] = t.groupby("nflId")["gameId"].rank(method="dense") % 2 == 0
    out = {"min_snaps_per_state_per_half": MIN_REL,
           "value_diff_split_half_weeks1-6_vs_7-8": _diff_reliability(t, "is_train"),
           "value_diff_odd_even_games": _diff_reliability(t, "_odd")}
    for s in ["inline", "detached"]:
        ds = t[t["state"] == s]
        sh = stats.split_half_reliability(ds, "nflId", "res_box")
        oe = stats.odd_even_reliability(ds, "nflId", "res_box")
        out[f"box_res_{s}_snaps_split_half"] = {"rho": _f(sh[0]), "n_players": int(sh[1])}
        out[f"box_res_{s}_snaps_odd_even"] = {"rho": _f(oe[0]), "n_players": int(oe[1])}
    return out


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    p, _ = build_plays()
    p, te = te_states(p)

    model_metrics, sens_metrics = {}, {}
    for y, kind in OUTCOMES.items():
        p[f"exp_{y}"], model_metrics[y] = fit_expected(p, y, kind, BASE_FEATS)
        p[f"res_{y}"] = p[y] - p[f"exp_{y}"]
    res_cols = [f"res_{y}" for y in OUTCOMES]
    league = league_effects(p, res_cols, rng)
    raw_means = p.groupby("play_state")[list(OUTCOMES)].mean().round(4).to_dict(orient="index")
    print(f"[{time.time() - t0:.0f}s] primary models done")

    # sensitivity: add offenseFormation
    ps = p[["gameId", "playId", "is_train", "play_state"] + SENS_FEATS + list(OUTCOMES)].copy()
    for y, kind in OUTCOMES.items():
        e, sens_metrics[y] = fit_expected(ps, y, kind, SENS_FEATS)
        ps[f"res_{y}"] = ps[y] - e
    league_sens = league_effects(ps, res_cols, rng)
    sens_shift = {c: {k: {"primary": league[c][k]["mean"], "with_formation": league_sens[c][k]["mean"]}
                      for k in ["te_inline", "te_detached", "no_te", "inline_minus_detached",
                                "te_present_minus_no_te"] if k in league[c]}
                  for c in res_cols}
    print(f"[{time.time() - t0:.0f}s] sensitivity models done")

    # TE-play level
    te = te.merge(p[["gameId", "playId", "possessionTeam", "play_state"] + res_cols],
                  on=["gameId", "playId"], how="left")
    pl, pooled_sd = player_table(te, rng)
    pl = pl.sort_values("value_shrunk", ascending=False, na_position="last")
    cols = ["nflId", "displayName", "team", "n", "n_inline", "n_detached", "n_eff", "n_games",
            "value", "value_shrunk", "lo", "hi", "box_res_inline", "box_res_detached"] + \
           [f"{c}_diff" for c in COMPONENTS] + ["top_eligible", "higher_is_better"]
    n_bf = te[te["state"] == "backfield"].groupby("nflId").size()
    pl["n_backfield"] = pl["nflId"].map(n_bf).fillna(0).astype(int)
    pl[cols + ["n_backfield"]].to_csv(OUT_DIR / "players.csv", index=False)
    rel = reliability(te)

    # team table: TE present vs absent (offense)
    tt = p.assign(te_present=p["n_te_on_field"] > 0).groupby(["possessionTeam", "te_present"])
    team = tt[res_cols].mean().unstack("te_present")
    team.columns = [f"{c}_{'te' if f else 'no_te'}" for c, f in team.columns]
    team["n_te"] = tt.size().unstack().get(True)
    team["n_no_te"] = tt.size().unstack().get(False)
    for c in res_cols:
        team[f"{c}_diff"] = team[f"{c}_te"] - team[f"{c}_no_te"]
    team = team.fillna({"n_no_te": 0}).sort_values("res_box_diff", ascending=False)
    team.round(4).to_csv(OUT_DIR / "teams.csv")

    # play-level output
    pcols = ["gameId", "playId", "week", "is_train", "possessionTeam", "defensiveTeam",
             "play_state", "n_te_on_field", "n_te_inline", "n_te_detached", "n_te_backfield",
             "offenseFormation"] + BASE_FEATS + list(OUTCOMES) + \
            [f"exp_{y}" for y in OUTCOMES] + res_cols
    p[pcols].to_csv(OUT_DIR / "plays.csv", index=False)

    top = pl[pl["top_eligible"]].head(5)[["displayName", "team", "n_inline", "n_detached",
                                          "value", "value_shrunk", "lo", "hi"]]
    summary = {
        "metric": "Constraint Value",
        "definition": ("Per TE: mean box-count residual (observed defendersInBox - context-expected) "
                       "on his inline/wing snaps minus on his detached snaps. Positive = defenses "
                       "put more men in the box (vs expectation) when he is inline than when he is "
                       "detached. Components: same inline-detached difference for deep-defender "
                       "count, two-high (tracking) and man-coverage residuals. League effects: mean "
                       "residual per play state with game-bootstrap CIs."),
        "higher_is_better": True,
        "parameters": {"deep_x_rel_at_snap": DEEP_X, "two_high_min_deep": TWO_HIGH_MIN,
                       "blitz_min_rushers": BLITZ_MIN, "pff_two_high_shells": PFF_TWO_HIGH,
                       "inline_alignments": INLINE_ALIGN, "min_snaps_each_state_top5": MIN_TOP,
                       "n_bootstrap": N_BOOT, "bootstrap_unit": "gameId",
                       "context_features_primary": BASE_FEATS,
                       "context_features_sensitivity": SENS_FEATS, "hgb": HGB_KW,
                       "play_state_rules": ("no_te: n_te_on_field==0; te_inline: >=1 TE inline/wing "
                                            "and none detached; te_detached: >=1 detached and none "
                                            "inline/wing; te_mixed: both; te_backfield_only: rest"),
                       "man_definition": "pff_passCoverageType == 'Man' (Zone and Other = 0)",
                       "eb_shrinkage": ("eb_shrink_mean on the difference with n_eff = "
                                        "1/(1/n_inline + 1/n_detached), pooled within player-state SD"),
                       "pooled_within_sd_box_res": pooled_sd},
        "sample_sizes": {"plays": int(len(p)),
                         "plays_with_snap_tracking": int(p["n_deep"].notna().sum()),
                         "play_state_counts": p["play_state"].value_counts().to_dict(),
                         "te_play_rows": int(len(te)),
                         "te_state_counts": te["state"].value_counts().to_dict(),
                         "tes": int(pl["nflId"].nunique()),
                         "tes_top_eligible": int(pl["top_eligible"].sum())},
        "test_set_model_metrics_primary": model_metrics,
        "test_set_model_metrics_with_formation": {
            k: {kk: v[kk] for kk in v if kk != "calibration_5bin"} for k, v in sens_metrics.items()},
        "league_effects_primary": league,
        "league_raw_observed_means_by_state": raw_means,
        "sensitivity_with_offenseFormation_league_means": sens_shift,
        "reliability": rel,
        "top5_min25_each_state": top.round(3).to_dict(orient="records"),
        "caveats": [
            "Observational: TE alignment is chosen by the offense, partly in response to expected "
            "defense; residuals are associations, not causal effects.",
            "Context model omits TE count/alignment by design, so personnel-driven differences "
            "(e.g. 12 vs 11 personnel) are partly absorbed via n_rb/n_wr/n_detached_nonte.",
            "No-TE plays are rare (~3%) so no-TE and team-level no-TE estimates are noisy; most "
            "teams have < 20 no-TE plays (see n_no_te in teams.csv).",
            "Data are dropbacks only (no designed runs), weeks 1-8 of 2021.",
            "defendersInBox is a PFF/NGS pre-snap label; deep counts use snap-frame tracking "
            "(x_rel >= 10), which misses late rotation after the snap.",
            "Player value is a within-player inline-vs-detached contrast; low reliability means it "
            "mostly reflects scheme/opponent noise rather than a stable player trait.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(json.dumps({"model": {k: v["model"] for k, v in model_metrics.items()},
                      "league_box": league["res_box"], "rel": rel}, indent=1, default=str))
    print(top.to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

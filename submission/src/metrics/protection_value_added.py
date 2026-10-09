"""Protection Value Added (PVA) for tight ends.

Play-level expected-pressure model (target = plays.pressure, any PFF hurry/hit/sack) fit with a
monotone-constrained HistGradientBoostingClassifier on train weeks and cross-fit by game.
Counterfactual: remove all TE help (full blockers + chips), recompute blocker counts and
rushers-minus-blockers, and re-predict with the SAME honest model (fold model for train rows,
full-train model for test rows).

    PVA_scheme    = E[pressure | no TE help] - E[pressure | observed]
    protection_oe = E[pressure | observed]   - actual pressure
    PVA_total     = E[pressure | no TE help] - actual pressure   (= scheme + oe)

Play values are split equally among the TEs who pass-blocked or chip-released on the play.
Higher = better (pressure prevented). Run: .venv/bin/python -m src.metrics.protection_value_added
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.common import io, stats  # noqa: E402
from src.common.config import N_FOLDS, OUT, SEED, TEST_WEEKS, TRAIN_WEEKS  # noqa: E402

SLUG = "protection_value_added"
OUT_DIR = OUT / SLUG

# ---------------------------------------------------------------- parameters
OL_POS = {"T", "G", "C"}
RB_POS = {"RB", "FB"}
EDGE_LABELS = {"LEO", "REO", "LOLB", "ROLB", "LE", "RE"}       # PFF edge alignments
DB_LABEL_PREFIX = ("SCB", "LCB", "RCB", "CB", "SS", "FS")       # DB blitzers = outside
OUTSIDE_GEOM_BUFFER = 1.0   # yd: rusher |lat| > widest OL |lat| + buffer also counts as outside
RUSHER_PRIOR_MIN_N = 50     # rush snaps for method-of-moments beta prior
MIN_TOP_SNAPS = 25          # assisted snaps required for top-5 list
HGB_PARAMS = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=SEED)

NUM_FEATS = ["n_ol_block", "n_rb_block", "n_te_block", "n_te_chip", "n_rushers",
             "rushers_minus_blockers", "n_rush_outside", "rusher_q_mean", "rusher_q_max",
             "pff_playAction", "down", "yardsToGo", "red_zone"]
CAT_FEATS = ["dropBackType", "offenseFormation", "pff_passCoverageType"]
FEATURES = NUM_FEATS + CAT_FEATS
MONO = {"n_ol_block": -1, "n_rb_block": -1, "n_te_block": -1, "n_te_chip": -1,
        "n_rushers": 1, "rushers_minus_blockers": 1}


def make_model(constrained=True):
    cst = [MONO.get(f, 0) if constrained else 0 for f in FEATURES]
    cat = [f in CAT_FEATS for f in FEATURES]
    return HistGradientBoostingClassifier(monotonic_cst=cst, categorical_features=cat, **HGB_PARAMS)


# ---------------------------------------------------------------- features
def rusher_quality(pp):
    """Per rusher-play beta-shrunk PFF pressure rate from train weeks; leave-own-game-out on train."""
    r = pp.loc[pp.pff_role == "Pass Rush", ["gameId", "playId", "nflId", "is_train",
                                            "pff_hurry", "pff_hit", "pff_sack"]].copy()
    r["pr"] = r[["pff_hurry", "pff_hit", "pff_sack"]].fillna(0).max(axis=1)
    tr = r[r.is_train]
    m = float(tr.pr.mean())
    tot = tr.groupby("nflId").pr.agg(k="sum", n="size").reset_index()
    pg = tr.groupby(["nflId", "gameId"]).pr.agg(kg="sum", ng="size").reset_index()
    big = tot[tot.n >= RUSHER_PRIOR_MIN_N]
    tau2 = max(float((big.k / big.n).var() - (m * (1 - m) / big.n).mean()), 1e-5)
    strength = max(m * (1 - m) / tau2 - 1, 1.0)
    r = r.merge(tot, on="nflId", how="left").merge(pg, on=["nflId", "gameId"], how="left")
    k = r.k.fillna(0) - r.kg.fillna(0)   # kg is NaN for test games -> nothing removed
    n = r.n.fillna(0) - r.ng.fillna(0)
    r["q"] = stats.beta_shrink(k, n, m, strength)
    return r, {"league_rush_pressure_rate_train": m, "prior_strength": strength,
               "n_rushers_prior": int(len(big))}


def build_play_frame(p, pp, te):
    key = ["gameId", "playId"]
    blk = pp[pp.pff_role == "Pass Block"]
    rush = pp[pp.pff_role == "Pass Rush"]
    g = pd.DataFrame({
        "n_ol_block": blk[blk.officialPosition.isin(OL_POS)].groupby(key).size(),
        "n_rb_block": blk[blk.officialPosition.isin(RB_POS)].groupby(key).size(),
        "n_te_block": te[te.assignment == "pass_block"].groupby(key).size(),
        "n_te_chip": te[te.assignment == "chip_release"].groupby(key).size(),
        "n_blockers_total": blk.groupby(key).size(),
    })
    # rushers aligned outside the tackles: PFF label OR geometry vs widest OL at snap
    ol_w = blk[blk.officialPosition.isin(OL_POS)].assign(a=lambda d: d.lat_from_ball.abs()) \
        .groupby(key).a.max().rename("ol_width")
    rr = rush.merge(ol_w.reset_index(), on=key, how="left")
    lab = rr.pff_positionLinedUp.fillna("")
    geo = rr.lat_from_ball.abs() > (rr.ol_width + OUTSIDE_GEOM_BUFFER)
    rr["outside"] = lab.isin(EDGE_LABELS) | lab.str.startswith(DB_LABEL_PREFIX) | geo.fillna(False)
    g["n_rush_outside"] = rr.groupby(key).outside.sum()
    g = g.reset_index()

    rq, rq_info = rusher_quality(pp)
    q = rq.groupby(key).q.agg(rusher_q_mean="mean", rusher_q_max="max").reset_index()

    # PFF label breakdown (any defender) for validation
    lab_df = pp.groupby(key)[["pff_hurry", "pff_hit", "pff_sack"]].max().fillna(0).astype(int) \
        .rename(columns=lambda c: c.replace("pff_", "any_")).reset_index()

    d = p[key + ["week", "is_train", "possessionTeam", "pressure", "n_rushers", "down", "yardsToGo",
                 "red_zone", "pff_playAction", "dropBackType", "offenseFormation",
                 "pff_passCoverageType"]].merge(g, on=key, how="left") \
        .merge(q, on=key, how="left").merge(lab_df, on=key, how="left")
    cnt = ["n_ol_block", "n_rb_block", "n_te_block", "n_te_chip", "n_blockers_total", "n_rush_outside"]
    d[cnt] = d[cnt].fillna(0).astype(int)
    d["rushers_minus_blockers"] = d.n_rushers - d.n_blockers_total
    d["red_zone"] = d.red_zone.astype(float)
    d["pff_playAction"] = d.pff_playAction.astype(float)
    d["pressure"] = d.pressure.astype(int)
    for c in CAT_FEATS:  # ordinal codes (no target info), NaN -> its own level
        d[c] = pd.Categorical(d[c].fillna("NA").astype(str)).codes.astype(int)
    return d.reset_index(drop=True), rq_info


# ---------------------------------------------------------------- honest predictions
class HonestPredictor:
    """Reproduces stats.cross_fit folds so counterfactual rows use the same honest model:
    OOF fold model for train rows, full-train model for test rows."""

    def __init__(self, model, df):
        self.df = df
        tr_idx = df.index[df.is_train]
        tr = df.loc[tr_idx]
        self.folds = []
        for fit_i, oof_i in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr.gameId):
            m = clone(model).fit(tr.iloc[fit_i][FEATURES], tr.iloc[fit_i].pressure)
            self.folds.append((tr_idx[oof_i], m))
        self.full = clone(model).fit(tr[FEATURES], tr.pressure)
        self.test_idx = df.index[~df.is_train]

    def predict(self, X):
        out = pd.Series(np.nan, index=X.index)
        for idx, m in self.folds + [(self.test_idx, self.full)]:
            idx = idx.intersection(X.index)
            if len(idx):
                out.loc[idx] = m.predict_proba(X.loc[idx, FEATURES])[:, 1]
        return out


def remove_te(X, n_block_removed, n_chip_removed):
    Xc = X.copy()
    Xc["n_te_block"] = Xc.n_te_block - n_block_removed
    Xc["n_te_chip"] = Xc.n_te_chip - n_chip_removed
    Xc["rushers_minus_blockers"] = Xc.n_rushers - (Xc.n_blockers_total - n_block_removed)
    return Xc


# ---------------------------------------------------------------- validation
def clf_report(y, pr, base):
    y, pr = np.asarray(y), np.asarray(pr)
    b = np.full_like(pr, base)
    out = {"n": int(len(y)), "pressure_rate": float(y.mean()), "base_rate_train": float(base),
           "auc": float(roc_auc_score(y, pr)), "log_loss": float(log_loss(y, pr)),
           "brier": float(brier_score_loss(y, pr)),
           "baseline_log_loss": float(log_loss(y, b)), "baseline_brier": float(brier_score_loss(y, b))}
    out["brier_skill"] = 1 - out["brier"] / out["baseline_brier"]
    bins = pd.qcut(pr, 5, labels=False, duplicates="drop")
    cal = pd.DataFrame({"bin": bins, "pred": pr, "y": y}).groupby("bin").agg(
        n=("y", "size"), mean_pred=("pred", "mean"), actual=("y", "mean")).reset_index()
    cal["base_rate"] = base
    out["calibration_5bin"] = cal.round(4).to_dict(orient="records")
    return out


def posterior_ci(means, ns, sd, min_n_hyper=10, z=1.96):
    """Same normal-normal hyper-parameters as stats.eb_shrink_mean, returning posterior SD."""
    means, ns = np.asarray(means, float), np.asarray(ns, float)
    se2 = sd**2 / np.maximum(ns, 1)
    h = ns >= min_n_hyper
    if h.sum() < 3:
        h = ns > 0
    tau2 = max(np.var(means[h]) - np.mean(se2[h]), 1e-6)
    shr = stats.eb_shrink_mean(means, ns, sd, min_n_hyper)
    post_sd = np.sqrt(tau2 * se2 / (tau2 + se2))
    return shr, shr - z * post_sd, shr + z * post_sd, tau2


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = io.plays()
    pp_cols = ["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp", "pff_hit", "pff_hurry",
               "pff_sack", "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed", "officialPosition",
               "lat_from_ball", "is_train"]
    pp = io.player_plays()[pp_cols]
    te = io.te_plays()[["gameId", "playId", "nflId", "displayName", "assignment", "alignment",
                        "pff_hurryAllowed", "pff_hitAllowed", "pff_sackAllowed", "is_train", "week"]]
    d, rq_info = build_play_frame(p, pp, te)
    base = float(d.loc[d.is_train, "pressure"].mean())

    # ---- expected pressure (shared cross_fit) + honest counterfactuals on identical folds
    model = make_model(True)
    d["e_obs"], _ = stats.cross_fit(model, d, FEATURES, "pressure", proba=True)
    hp = HonestPredictor(model, d)
    chk = float(np.abs(hp.predict(d) - d.e_obs).max())
    assert chk < 1e-9, f"fold replication mismatch {chk}"

    d["n_assist"] = d.n_te_block + d.n_te_chip
    d["e_cf"] = d.e_obs
    a = d.n_assist > 0
    d.loc[a, "e_cf"] = hp.predict(remove_te(d.loc[a], d.loc[a, "n_te_block"], d.loc[a, "n_te_chip"]))
    d["pva_scheme"] = d.e_cf - d.e_obs
    d["protection_oe"] = d.e_obs - d.pressure
    d["pva_total"] = d.e_cf - d.pressure

    # sanity: marginal value of ONE TE full block vs ONE TE chip (league)
    mb, mc = d.n_te_block > 0, d.n_te_chip > 0
    marg_block = hp.predict(remove_te(d[mb], 1, 0)) - d.loc[mb, "e_obs"]
    marg_chip = hp.predict(remove_te(d[mc], 0, 1)) - d.loc[mc, "e_obs"]
    sanity = {"league_pva_scheme_one_full_block": float(marg_block.mean()), "n_block_plays": int(mb.sum()),
              "league_pva_scheme_one_chip": float(marg_chip.mean()), "n_chip_plays": int(mc.sum()),
              "share_negative_play_pva_scheme": float((d.loc[a, "pva_scheme"] < -1e-12).mean())}
    sanity["pass_block_gt_chip_gt_0"] = bool(sanity["league_pva_scheme_one_full_block"]
                                             > sanity["league_pva_scheme_one_chip"] > 0)

    # ---- test-set validation
    tst = d[~d.is_train]
    val = {"test_constrained": clf_report(tst.pressure, tst.e_obs, base),
           "train_oof_constrained": {k: v for k, v in clf_report(
               d.loc[d.is_train, "pressure"], d.loc[d.is_train, "e_obs"], base).items()
               if k != "calibration_5bin"},
           "test_auc_vs_pff_labels": {lab: float(roc_auc_score(tst[f"any_{lab}"], tst.e_obs))
                                     for lab in ["hurry", "hit", "sack"]},
           "test_label_rates": {lab: float(tst[f"any_{lab}"].mean()) for lab in ["hurry", "hit", "sack"]},
           "test_te_assisted_plays": {k: v for k, v in clf_report(
               tst.loc[tst.n_assist > 0, "pressure"], tst.loc[tst.n_assist > 0, "e_obs"], base).items()
               if k != "calibration_5bin"}}

    # sensitivity: unconstrained model (cost of constraints + scheme effect without them)
    e_unc, _ = stats.cross_fit(make_model(False), d, FEATURES, "pressure", proba=True)
    hp_u = HonestPredictor(make_model(False), d)
    tu = ~d.is_train
    sens = {"unconstrained_test_auc": float(roc_auc_score(d.pressure[tu], e_unc[tu])),
            "unconstrained_test_log_loss": float(log_loss(d.pressure[tu], e_unc[tu])),
            "unconstrained_league_pva_scheme_one_full_block":
                float((hp_u.predict(remove_te(d[mb], 1, 0)) - e_unc[mb]).mean()),
            "unconstrained_league_pva_scheme_one_chip":
                float((hp_u.predict(remove_te(d[mc], 0, 1)) - e_unc[mc]).mean())}

    # ---- TE-play rows: equal split of play value among assisting TEs
    tp = te[te.assignment.isin(["pass_block", "chip_release"])].merge(
        d[["gameId", "playId", "possessionTeam", "pressure", "e_obs", "e_cf", "n_assist", "n_te_block",
           "n_te_chip", "pva_scheme", "protection_oe", "pva_total", "any_hurry", "any_hit", "any_sack"]],
        on=["gameId", "playId"])
    for c in ["pva_scheme", "protection_oe", "pva_total"]:
        tp[c] = tp[c] / tp.n_assist
    tp["inline"] = tp.alignment == "inline"
    al = tp[["pff_hurryAllowed", "pff_hitAllowed", "pff_sackAllowed"]]
    tp["te_allowed"] = al.max(axis=1).where(al.notna().any(axis=1))

    # OL/RB allowed-pressure rate on the same plays (per blocker snap)
    olrb = pp[(pp.pff_role == "Pass Block") & pp.officialPosition.isin(OL_POS | RB_POS)].copy()
    ala = olrb[["pff_hurryAllowed", "pff_hitAllowed", "pff_sackAllowed"]]
    olrb["allowed"] = ala.max(axis=1).where(ala.notna().any(axis=1))
    olrb_play = olrb.groupby(["gameId", "playId"]).allowed.agg(olrb_k="sum", olrb_n="count").reset_index()
    tp = tp.merge(olrb_play, on=["gameId", "playId"], how="left")

    # ---- player table
    sd = stats.pooled_within_sd(tp, "nflId", "pva_total")
    g = tp.groupby("nflId")
    pl = pd.DataFrame({
        "displayName": g.displayName.first(),
        "team": g.possessionTeam.agg(lambda s: s.mode().iat[0]),
        "n": g.size(),
        "n_block": g.assignment.agg(lambda s: int((s == "pass_block").sum())),
        "n_chip": g.assignment.agg(lambda s: int((s == "chip_release").sum())),
        "value": g.pva_total.mean(),
        "pva_scheme": g.pva_scheme.mean(),
        "protection_oe": g.protection_oe.mean(),
        "pressure_rate": g.pressure.mean(),
        "te_allowed_rate": g.te_allowed.mean(),
        "n_allowed_snaps": g.te_allowed.count(),
        "olrb_allowed_rate_same_plays": g.olrb_k.sum() / g.olrb_n.sum(),
    })
    pl["value_volume"] = pl.value * pl.n
    for lab, mask in [("inline", tp.inline), ("other", ~tp.inline)]:
        s = tp[mask].groupby("nflId")
        pl[f"value_{lab}"] = s.pva_total.mean()
        pl[f"pva_scheme_{lab}"] = s.pva_scheme.mean()
        pl[f"n_{lab}"] = s.size()
    pl[["n_inline", "n_other"]] = pl[["n_inline", "n_other"]].fillna(0).astype(int)
    pl["value_shrunk"], pl["lo_shrunk"], pl["hi_shrunk"], tau2 = posterior_ci(pl.value.values, pl.n.values, sd)
    # lo/hi = raw-mean 95% CI with pooled within-player SD (posterior CI is prior-dominated when tau2 ~ 0)
    pl["lo"] = pl.value - 1.96 * sd / np.sqrt(pl.n)
    pl["hi"] = pl.value + 1.96 * sd / np.sqrt(pl.n)
    pl["higher_is_better"] = True
    pl = pl.reset_index().sort_values("value_shrunk", ascending=False)
    cols = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "lo_shrunk",
            "hi_shrunk", "higher_is_better",
            "pva_scheme", "protection_oe", "value_volume", "n_block", "n_chip", "value_inline", "n_inline",
            "pva_scheme_inline", "value_other", "n_other", "pva_scheme_other", "pressure_rate",
            "te_allowed_rate", "n_allowed_snaps", "olrb_allowed_rate_same_plays"]
    pl[cols].to_csv(OUT_DIR / "players.csv", index=False, float_format="%.5f")
    tp.drop(columns=["inline"]).to_csv(OUT_DIR / "plays.csv", index=False, float_format="%.5f")
    d[["gameId", "playId", "week", "is_train", "pressure", "any_hurry", "any_hit", "any_sack", "e_obs",
       "e_cf", "pva_scheme", "protection_oe", "pva_total", "n_assist", "n_blockers_total"] + FEATURES] \
        .to_csv(OUT_DIR / "plays_model.csv", index=False, float_format="%.5f")

    top = pl[pl.n >= MIN_TOP_SNAPS].head(5)[["displayName", "team", "n", "n_block", "n_chip", "value",
                                             "value_shrunk", "lo", "hi", "pva_scheme", "protection_oe"]]

    # ---- reliability
    rel = {}
    for c in ["pva_total", "pva_scheme", "protection_oe", "te_allowed"]:
        sh, nsh = stats.split_half_reliability(tp, "nflId", c)
        oe, noe = stats.odd_even_reliability(tp, "nflId", c)
        rel[c] = {"split_half_weeks_spearman": sh, "split_half_n_players": nsh,
                  "odd_even_games_spearman": oe, "odd_even_n_players": noe}

    # ---- allowed-pressure rates (league)
    tb = tp[tp.te_allowed.notna()]
    allowed = {"te_allowed_rate_per_snap": float(tb.te_allowed.mean()), "te_n_snaps": int(len(tb)),
               "te_n_snaps_by_assignment": tb.assignment.value_counts().to_dict(),
               "olrb_allowed_rate_same_plays": float(tb.drop_duplicates(["gameId", "playId"]).olrb_k.sum()
                                                     / tb.drop_duplicates(["gameId", "playId"]).olrb_n.sum())}

    summary = {
        "metric": "Protection Value Added (PVA)", "slug": SLUG,
        "definition": {
            "target": "plays.pressure (any PFF hurry/hit/sack on the play)",
            "pva_scheme": "E[pressure | TE blockers & chips set to 0] - E[pressure | observed]",
            "protection_oe": "E[pressure | observed] - actual pressure",
            "pva_total": "E[pressure | no TE help] - actual pressure (player value, per assisted snap)",
            "credit": "play values split equally among TEs with assignment pass_block or chip_release",
            "counterfactual": "n_te_block=0, n_te_chip=0, rushers_minus_blockers = n_rushers - "
                              "(pass blockers - TE blockers); chips are not counted as blockers",
            "honesty": "counterfactuals use the same fold model as each play's OOF prediction "
                       "(train) or the full-train model (test); fold replication checked (max diff "
                       f"{chk:.1e})",
            "higher_is_better": True,
        },
        "features": FEATURES, "categorical_features": CAT_FEATS,
        "monotonic_constraints": {f: MONO.get(f, 0) for f in FEATURES},
        "constraint_note": "rushers_minus_blockers (+1) added beyond the requested set so that removing "
                           "a TE blocker cannot lower expected pressure through the derived feature",
        "excluded_features": ["time_to_throw (caused by pressure)", "player identity"],
        "parameters": {"hgb": HGB_PARAMS, "edge_labels": sorted(EDGE_LABELS),
                       "db_label_prefixes": list(DB_LABEL_PREFIX),
                       "outside_geom_buffer_yd": OUTSIDE_GEOM_BUFFER,
                       "rusher_quality": {**rq_info, "prior_min_n": RUSHER_PRIOR_MIN_N,
                                          "method": "beta-shrunk PFF pressure rate, train weeks only, "
                                                    "leave-own-game-out on train rows; per-play mean & max"},
                       "min_top_snaps": MIN_TOP_SNAPS, "eb_pooled_within_sd": sd, "eb_tau2": float(tau2),
                       "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS, "n_folds": N_FOLDS},
        "sample": {"plays": int(len(d)), "train_plays": int(d.is_train.sum()), "test_plays": int((~d.is_train).sum()),
                   "te_assisted_plays": int(a.sum()), "te_assisted_snaps": int(len(tp)),
                   "te_block_snaps": int((tp.assignment == "pass_block").sum()),
                   "te_chip_snaps": int((tp.assignment == "chip_release").sum()),
                   "tes": int(len(pl)), "tes_with_min_snaps": int((pl.n >= MIN_TOP_SNAPS).sum())},
        "validation": val, "sanity": sanity, "sensitivity": sens, "reliability": rel,
        "allowed_pressure": allowed,
        "league_means_per_te_snap": tp.groupby("assignment")[["pva_scheme", "protection_oe", "pva_total"]]
            .mean().to_dict(orient="index"),
        "top5_min_snaps": top.round(4).to_dict(orient="records"),
        "caveats": [
            "Observational: TE stay-in decisions are scheme choices correlated with expected pressure; "
            "the model controls for listed context only.",
            "protection_oe credits the whole protection unit's over/under-performance to the assisting TEs.",
            "Equal split among TEs ignores who actually faced rushers.",
            "Monotone constraints force PVA_scheme >= 0 by construction; its size, not sign, is informative.",
            "Chip sanity check: the constrained model never lowers pressure for a chip (effect clamps to 0); "
            "the unconstrained model estimates a chip RAISES expected pressure, i.e. chips are used on "
            "plays that are harder to protect in ways the features do not capture.",
            "EB tau2 hits its floor (player PVA is not separable from noise in 8 weeks), so value_shrunk "
            "is ~the grand mean for everyone; lo/hi are raw-mean CIs, lo_shrunk/hi_shrunk posterior.",
            "Time-to-threat variant skipped for runtime.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: summary[k] for k in ["sample", "sanity", "sensitivity", "reliability",
                                              "allowed_pressure", "league_means_per_te_snap"]},
                     indent=1, default=float))
    v = val["test_constrained"]
    print({k: round(v[k], 4) for k in ["auc", "log_loss", "baseline_log_loss", "brier", "baseline_brier"]})
    print(pd.DataFrame(v["calibration_5bin"]).to_string(index=False))
    print(val["test_auc_vs_pff_labels"], val["test_te_assisted_plays"]["auc"])
    print(top.to_string(index=False))
    print(f"pooled sd {sd:.4f} tau2 {tau2:.2e}; runtime {summary['runtime_sec']}s")


if __name__ == "__main__":
    main()

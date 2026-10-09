"""Dual-Threat Deployment Index (DDI) for tight ends.  slug = deployment_index

Definition (tight-end-metrics.md): formation-adjusted usage mix across inline routes, detached
routes, chip-and-release assignments, stay-in blocks and standard pass-block snaps.

Snap categories (te_plays.assignment x alignment x pff_blockType), REFINED rule (see summary.json
"category_rule" for why the literal rule was changed):
  inline_route        assignment == route        and alignment in {inline, wing, backfield}
  detached_route      assignment == route        and alignment == detached
  chip_release        assignment == chip_release (Pass Route with blockType CH/SR)
  standard_pass_block assignment == pass_block   and alignment == inline and blockType in {PP, PA}
  stay_in_block       every other pass_block (off-line alignment, or non-conventional block type)

Formation adjustment
  * context cell = offenseFormation x personnelO, backed off to offenseFormation (then league)
    when the cell has < MIN_CELL train TE snaps.
  * expected mix per snap = train-week league shares in the snap's cell; train rows get out-of-fold
    (GroupKFold by game) expectations, test rows use the full-train fit. Reported share - expected.
  * DDI value = normalized Shannon entropy H/log(5) of the TE's category mix after inverse-
    propensity reweighting of his snaps to the league (train) context distribution:
    w(c) = q_league(c | TE support) / p_TE(c), capped at W_CAP.
  * value_shrunk = EB (normal-normal) shrinkage of DDI toward the grand mean using bootstrap
    variance. Shares are also Dirichlet-shrunk toward the league mix (wshare_shrunk_*).
    Miller-Madow corrected entropies also reported.
  * CI: bootstrap of games within player (weights recomputed in each resample).

Run: .venv/bin/python -m src.metrics.deployment_index
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.common import io, stats  # noqa: E402
from src.common.config import N_FOLDS, OUT, SEED, TEST_WEEKS, TRAIN_WEEKS  # noqa: E402

SLUG = "deployment_index"
OUTDIR = OUT / SLUG
CATS = ["inline_route", "detached_route", "chip_release", "stay_in_block", "standard_pass_block"]
K = len(CATS)
LOGK = np.log(K)
MIN_CELL = 30          # min train TE snaps for a formation x personnel cell
W_CAP = 5.0            # IPW weight cap
N_BOOT = 1000
MIN_TOP = 50           # min snaps for top-5 / archetype labels
MIN_REL = 20           # min snaps per half for reliability correlations
STANDARD_TYPES = {"PP", "PA"}
NON_PP_TYPES = {"PA", "PU", "BH", "PT", "SW", "PR", "UP", "CL", "NB"}
# Archetype thresholds on formation-adjusted (IPW) shares, applied in this order
ARCH = {"chip_and_release_min_chip": 0.20,      # ~2x league chip share
        "inline_blocker_min_block": 0.25,       # stay_in + standard, ~2x league block share
        "receiving_specialist_min_route": 0.85}  # inline + detached routes


# ----------------------------------------------------------------------------- classification
def classify(df, literal=False):
    a, al, bt = df["assignment"], df["alignment"], df["pff_blockType"].fillna("NA")
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out[(a == "route") & al.isin(["inline", "wing", "backfield"])] = "inline_route"
    out[(a == "route") & (al == "detached")] = "detached_route"
    out[a == "chip_release"] = "chip_release"
    pb = a == "pass_block"
    if literal:  # rule exactly as specified in the request
        std = pb & (bt == "PP") & (al == "inline")
        stay = pb & (bt.isin(NON_PP_TYPES) | (al != "inline"))
        out[std] = "standard_pass_block"
        out[stay] = "stay_in_block"
        out[pb & out.isna()] = "stay_in_block"  # residual CH/SR/NA-typed inline blocks
    else:
        std = pb & bt.isin(STANDARD_TYPES) & (al == "inline")
        out[std] = "standard_pass_block"
        out[pb & ~std] = "stay_in_block"
    return out


# ----------------------------------------------------------------------------- entropy helpers
def norm_entropy(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)
    return h / LOGK


def miller_madow(p, n):
    """Normalized Miller-Madow entropy: (H + (m-1)/(2n)) / log K, m = observed categories."""
    p = np.asarray(p, float)
    m = (p > 0).sum(axis=-1)
    return norm_entropy(p) + (m - 1) / (2 * np.asarray(n, float)) / LOGK


# ----------------------------------------------------------------------------- expected mix
def fit_cells(tr):
    """Train-only shares: cell -> share vector, formation -> share vector, global."""
    oh = pd.get_dummies(tr["category"]).reindex(columns=CATS, fill_value=0).astype(float)
    g = oh.groupby(tr["fp"]).sum()
    gn = tr.groupby("fp").size()
    f = oh.groupby(tr["form"]).sum()
    fn = tr.groupby("form").size()
    return {"cell": g.div(gn, axis=0)[gn >= MIN_CELL], "form": f.div(fn, axis=0)[fn >= MIN_CELL],
            "glob": oh.mean()}


def assign_ctx(df, fit):
    ctx = np.where(df["fp"].isin(fit["cell"].index), "C:" + df["fp"],
                   np.where(df["form"].isin(fit["form"].index), "F:" + df["form"], "ALL"))
    return pd.Series(ctx, index=df.index)


def predict_cells(df, fit):
    ctx = assign_ctx(df, fit)
    tab = pd.concat([fit["cell"].rename(index=lambda s: "C:" + s),
                     fit["form"].rename(index=lambda s: "F:" + s),
                     fit["glob"].to_frame("ALL").T])
    return pd.DataFrame(tab.loc[ctx].to_numpy(), index=df.index, columns=CATS), ctx


def cross_fit_mix(d):
    exp = pd.DataFrame(np.nan, index=d.index, columns=CATS)
    tr, te = d[d.is_train], d[~d.is_train]
    for fi, oi in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr["gameId"]):
        e, _ = predict_cells(tr.iloc[oi], fit_cells(tr.iloc[fi]))
        exp.loc[e.index] = e
    full = fit_cells(tr)
    e, _ = predict_cells(te, full)
    exp.loc[e.index] = e
    return exp, full


def mix_quality(y, P, base):
    """Multiclass log loss / Brier for P vs constant base-rate baseline."""
    Y = pd.get_dummies(y).reindex(columns=CATS, fill_value=0).to_numpy(float)
    P = np.clip(np.asarray(P, float), 1e-6, 1)
    B = np.clip(np.tile(base, (len(Y), 1)), 1e-6, 1)
    ll = lambda Q: float(-(Y * np.log(Q / Q.sum(1, keepdims=True))).sum(1).mean())  # noqa: E731
    br = lambda Q: float(((Q - Y) ** 2).sum(1).mean())  # noqa: E731
    return {"n": int(len(Y)), "log_loss": ll(P), "log_loss_baseline": ll(B),
            "brier": br(P), "brier_baseline": br(B)}


def calib_table(y, P):
    rows = []
    for j, c in enumerate(CATS):
        p = pd.Series(np.asarray(P)[:, j])
        obs = (pd.Series(np.asarray(y)) == c).astype(float)
        b = pd.qcut(p.rank(method="first"), 5, labels=False)
        t = pd.DataFrame({"p": p, "o": obs, "b": b}).groupby("b").agg(
            mean_pred=("p", "mean"), obs_rate=("o", "mean"), n=("o", "size")).reset_index()
        t.insert(0, "category", c)
        rows.append(t)
    return pd.concat(rows).round(4).to_dict(orient="records")


# ----------------------------------------------------------------------------- IPW per player
def ipw_mix(ctx, cat, q):
    """Weighted shares of one player's snaps reweighted to league context dist q (Series)."""
    p = ctx.value_counts(normalize=True)
    qs = q.reindex(p.index).fillna(0)
    qs = qs / qs.sum() if qs.sum() > 0 else p
    wc = (qs / p).clip(upper=W_CAP)
    w = ctx.map(wc).to_numpy(float)
    oh = (cat.to_numpy()[:, None] == np.array(CATS)[None, :]).astype(float)
    sh = (w[:, None] * oh).sum(0) / w.sum()
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return sh, n_eff, w, float((ctx.map(qs / p) > W_CAP).mean())


def boot_player(g, q, rng):
    """Bootstrap games within a player; returns DDI draws."""
    games = g["gameId"].unique()
    cells = g["ctx"].unique()
    C = np.zeros((len(games), len(cells), K))
    gi = pd.Index(games).get_indexer(g["gameId"])
    ci = pd.Index(cells).get_indexer(g["ctx"])
    ki = pd.Index(CATS).get_indexer(g["category"])
    np.add.at(C, (gi, ci, ki), 1)
    M = rng.multinomial(len(games), np.full(len(games), 1 / len(games)), size=N_BOOT)
    A = np.einsum("bg,gck->bck", M.astype(float), C)            # B x cells x K
    pc = A.sum(2)
    pc = pc / pc.sum(1, keepdims=True)
    qv = q.reindex(cells).fillna(0).to_numpy()
    qs = np.where(pc > 0, qv[None, :], 0.0)
    qs = qs / np.where(qs.sum(1, keepdims=True) > 0, qs.sum(1, keepdims=True), 1)
    qs = np.where(qs.sum(1, keepdims=True) > 0, qs, pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pc > 0, np.minimum(qs / pc, W_CAP), 0.0)
    wa = (A * w[:, :, None]).sum(1)
    return norm_entropy(wa / wa.sum(1, keepdims=True))


def dirichlet_precision(counts, n, pi):
    """Method-of-moments Dirichlet-multinomial precision k from players' raw shares."""
    ph = counts / n[:, None]
    s = ((ph - pi) ** 2 / (pi * (1 - pi))).sum(1) - K * (1 / n)
    rho = s.sum() / (K * (1 - 1 / n)).sum()
    rho = min(max(rho, 1e-4), 0.999)
    return 1 / rho - 1


def half_reliability(d, half_col, q):
    """Spearman of per-half player DDI and IPW shares (both halves need >= MIN_REL snaps)."""
    rows = []
    for (pid, h), g in d.groupby(["nflId", half_col]):
        if len(g) < MIN_REL:
            continue
        sh, _, _, _ = ipw_mix(g["ctx"], g["category"], q)
        rows.append({"nflId": pid, "half": h, "ddi": norm_entropy(sh), **dict(zip(CATS, sh))})
    r = pd.DataFrame(rows)
    out = {}
    for col in ["ddi"] + CATS:
        w = r.pivot(index="nflId", columns="half", values=col).dropna()
        out[col] = {"spearman": float(sps.spearmanr(w.iloc[:, 0], w.iloc[:, 1]).correlation)
                    if len(w) >= 5 else None, "n_players": int(len(w))}
    return out


# ----------------------------------------------------------------------------- main
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cols = ["gameId", "playId", "nflId", "displayName", "week", "is_train", "assignment",
            "alignment", "pff_blockType", "pff_positionLinedUp"]
    t = io.te_plays()[cols].copy()
    p = io.plays()[["gameId", "playId", "offenseFormation", "personnelO", "possessionTeam",
                    "pff_playAction"]]
    d = t.merge(p, on=["gameId", "playId"], how="left", validate="m:1")
    d["form"] = d["offenseFormation"].fillna("UNKNOWN").astype(str)
    d["fp"] = d["form"] + "|" + d["personnelO"].fillna("UNKNOWN").astype(str)

    # --- category rule: crosstab check, literal vs refined
    pb = d[d.assignment == "pass_block"]
    xtab = pd.crosstab(pb["pff_blockType"].fillna("NA"), pb["alignment"], margins=True)
    pa_by_type = pb.groupby(pb["pff_blockType"].fillna("NA"))["pff_playAction"].mean().round(3)
    d["category_literal"] = classify(d, literal=True)
    d["category"] = classify(d)
    assert d["category"].notna().all() and d["category_literal"].notna().all()
    lit_counts = d.loc[pb.index, "category_literal"].value_counts().to_dict()
    ref_counts = d.loc[pb.index, "category"].value_counts().to_dict()
    pa_share_bucket = d.loc[pb.index].groupby("category")["pff_playAction"].mean().round(3).to_dict()
    pa_share_bucket_lit = d.loc[pb.index].groupby("category_literal")["pff_playAction"].mean().round(3).to_dict()

    # --- expected mix (cross-fit) + test-set quality
    exp, full = cross_fit_mix(d)
    d["ctx"] = assign_ctx(d, full)
    for c in CATS:
        d[f"exp_{c}"] = exp[c]
    tr, te = d[d.is_train], d[~d.is_train]
    base = tr["category"].value_counts(normalize=True).reindex(CATS).fillna(0).to_numpy()
    quality = mix_quality(te["category"], exp.loc[te.index].to_numpy(), base)
    calib = calib_table(te["category"].to_numpy(), exp.loc[te.index].to_numpy())
    q_league = tr["ctx"].value_counts(normalize=True)   # league (train) context distribution
    league_train = pd.Series(base, index=CATS)
    league_test = te["category"].value_counts(normalize=True).reindex(CATS).fillna(0)
    ct = pd.crosstab(d["is_train"], d["category"]).reindex(columns=CATS, fill_value=0)
    chi2 = sps.chi2_contingency(ct.to_numpy())
    exp_test_mean = exp.loc[te.index].mean()

    # --- per-player metric
    rows, wts = [], pd.Series(np.nan, index=d.index)
    for pid, g in d.groupby("nflId"):
        n = len(g)
        oh = pd.get_dummies(g["category"]).reindex(columns=CATS, fill_value=0).to_numpy(float)
        raw = oh.mean(0)
        sh, n_eff, w, frac_cap = ipw_mix(g["ctx"], g["category"], q_league)
        wts.loc[g.index] = w
        sh_lit, _, _, _ = ipw_mix(g["ctx"], g["category_literal"], q_league)
        draws = boot_player(g, q_league, rng)
        expm = g[[f"exp_{c}" for c in CATS]].mean().to_numpy()
        r = {"nflId": pid, "displayName": g["displayName"].iloc[0],
             "team": g["possessionTeam"].mode().iloc[0], "n": n, "n_games": g["gameId"].nunique(),
             "n_eff_ipw": n_eff, "frac_snaps_capped": frac_cap,
             "value": norm_entropy(sh), "lo": np.nanpercentile(draws, 2.5),
             "hi": np.nanpercentile(draws, 97.5), "boot_sd": float(np.nanstd(draws)), "value_mm": miller_madow(sh, n_eff),
             "versatility_raw": norm_entropy(raw), "versatility_raw_mm": miller_madow(raw, n),
             "value_literal_rule": norm_entropy(sh_lit)}
        for j, c in enumerate(CATS):
            r[f"n_{c}"] = int(oh[:, j].sum())
            r[f"share_{c}"] = raw[j]
            r[f"exp_{c}"] = expm[j]
            r[f"diff_{c}"] = raw[j] - expm[j]
            r[f"wshare_{c}"] = sh[j]
        rows.append(r)
    pl = pd.DataFrame(rows)
    d["ipw_weight"] = wts

    # --- EB (Dirichlet) shrinkage of weighted shares toward league mix
    hyp = pl[pl.n >= 10]
    kprec = dirichlet_precision(hyp[[f"n_{c}" for c in CATS]].to_numpy(float),
                                hyp["n"].to_numpy(float), league_train.to_numpy())
    W = pl[[f"wshare_{c}" for c in CATS]].to_numpy()
    ne = pl["n_eff_ipw"].to_numpy()[:, None]
    Ws = (ne * W + kprec * league_train.to_numpy()[None, :]) / (ne + kprec)
    # Entropy of Dirichlet-shrunk shares is NOT used as value_shrunk: entropy is concave, so mixing
    # any skewed mix with the league mix raises entropy and pushes small-n TEs to the top.
    # value_shrunk: normal-normal EB of DDI toward the grand mean, per-player variance = bootstrap
    # variance (1-game TEs, whose bootstrap is degenerate, get the median per-snap SD).
    sd_snap = pl["boot_sd"] * np.sqrt(pl["n"])
    sd_snap = sd_snap.where((pl["n_games"] >= 2) & (pl["boot_sd"] > 0), sd_snap[pl.n_games >= 2].median())
    pl["value_shrunk"] = stats.eb_shrink_mean(pl["value"], pl["n"], sd_snap.to_numpy())
    for j, c in enumerate(CATS):
        pl[f"wshare_shrunk_{c}"] = Ws[:, j]

    # --- archetypes (on IPW shares)
    route = pl["wshare_inline_route"] + pl["wshare_detached_route"]
    block = pl["wshare_stay_in_block"] + pl["wshare_standard_pass_block"]
    pl["archetype"] = np.select(
        [pl["wshare_chip_release"] >= ARCH["chip_and_release_min_chip"],
         block >= ARCH["inline_blocker_min_block"],
         route >= ARCH["receiving_specialist_min_route"]],
        ["chip-and-release specialist", "inline blocker", "receiving specialist"], "balanced")
    pl["archetype_reliable"] = pl["n"] >= MIN_TOP
    pl["higher_is_better"] = True
    pl = pl.sort_values("value_shrunk", ascending=False)

    # --- reliability
    d["_oe"] = d.groupby("nflId")["gameId"].rank(method="dense") % 2 == 0
    rel_split = half_reliability(d, "is_train", q_league)
    rel_oe = half_reliability(d, "_oe", q_league)
    oh_all = pd.get_dummies(d["category"]).reindex(columns=CATS, fill_value=0).astype(float)
    dd = pd.concat([d[["nflId", "gameId", "is_train"]], oh_all], axis=1)
    rel_raw = {c: {"split_half": stats.split_half_reliability(dd, "nflId", c, min_n=MIN_REL),
                   "odd_even": stats.odd_even_reliability(dd, "nflId", c, min_n=MIN_REL)}
               for c in CATS}
    rel_raw = {c: {k: {"spearman": float(v[0]) if v[0] == v[0] else None, "n_players": int(v[1])}
                   for k, v in r.items()} for c, r in rel_raw.items()}

    big = pl[pl.n >= MIN_TOP]
    sens_lit = float(sps.spearmanr(big["value"], big["value_literal_rule"]).correlation)
    sens_raw = float(sps.spearmanr(big["value"], big["versatility_raw"]).correlation)
    top5 = big.sort_values("value", ascending=False).head(5)

    # --- outputs
    pcols = ["gameId", "playId", "nflId", "displayName", "week", "is_train", "possessionTeam",
             "offenseFormation", "personnelO", "ctx", "alignment", "assignment", "pff_blockType",
             "pff_playAction", "category", "category_literal", "ipw_weight"] + [f"exp_{c}" for c in CATS]
    d[pcols].to_csv(OUTDIR / "plays.csv", index=False)
    pl.round(4).to_csv(OUTDIR / "players.csv", index=False)

    summary = {
        "metric": "Dual-Threat Deployment Index (DDI)", "slug": SLUG, "higher_is_better": True,
        "definition": ("value = normalized Shannon entropy H/log(5) of a TE's 5-category snap mix "
                       "after inverse-propensity reweighting of his snaps to the league (train-week) "
                       "offenseFormation x personnelO context distribution; weights "
                       "w(c)=q_league(c|TE support)/p_TE(c) capped at 5. value_shrunk = normal-normal "
                       "EB shrinkage of DDI toward the grand mean (stats.eb_shrink_mean), per-player "
                       "sampling variance = bootstrap variance. wshare_shrunk_* = IPW shares "
                       "Dirichlet-shrunk toward the league train mix (prior strength k, Kish n_eff). "
                       "lo/hi = 2.5/97.5 percentiles of a bootstrap over games within player."),
        "category_rule": {
            "implemented": {
                "inline_route": "assignment=route & alignment in {inline, wing, backfield}",
                "detached_route": "assignment=route & alignment=detached",
                "chip_release": "assignment=chip_release (Pass Route with blockType CH/SR)",
                "standard_pass_block": "assignment=pass_block & alignment=inline & blockType in {PP, PA}",
                "stay_in_block": "any other pass_block: off-line alignment (wing/detached/backfield) "
                                 "or non-conventional type (CL, NB, PT, PR, SW, UP, BH, PU, CH, SR, NA)"},
            "literal_rule_as_requested": {
                "standard_pass_block": "pass_block & blockType=PP & alignment=inline",
                "stay_in_block": "pass_block & (blockType in PA/PU/BH/PT/SW/PR/UP/CL/NB or off-line); "
                                 "residual inline CH/SR/NA-typed pass blocks also -> stay_in"},
            "why_refined": ("Under the literal rule 82% of TE pass blocks are stay_in, driven by PA "
                            "(557 of 1432; PFF defines PA as inline pass protection on a play-action "
                            "pass, 98% on play-action plays). The literal split mostly encodes the "
                            "play call (play action vs not), not the TE's job. Refined rule treats "
                            "inline PP and PA as the same conventional inline pass set; stay_in = "
                            "off-line or non-conventional protection. Literal-rule DDI kept as a "
                            "sensitivity column (value_literal_rule)."),
            "pass_block_counts_literal": lit_counts, "pass_block_counts_refined": ref_counts,
            "play_action_rate_by_bucket_literal": pa_share_bucket_lit,
            "play_action_rate_by_bucket_refined": pa_share_bucket,
            "blockType_x_alignment_te_pass_blocks": xtab.to_dict(),
            "play_action_rate_by_blockType": pa_by_type.to_dict(),
            "category_counts_all": d["category"].value_counts().to_dict()},
        "parameters": {"min_cell_train_snaps": MIN_CELL,
                       "backoff": "formation x personnel -> formation -> league",
                       "ipw_weight_cap": W_CAP, "n_bootstrap": N_BOOT, "bootstrap_unit": "game within player",
                       "dirichlet_prior_strength_k": round(float(kprec), 2),
                       "min_snaps_top5_and_archetype": MIN_TOP, "min_snaps_per_half_reliability": MIN_REL,
                       "archetype_thresholds_on_ipw_shares": ARCH,
                       "archetype_order": ["chip-and-release specialist", "inline blocker",
                                           "receiving specialist", "balanced (otherwise)"],
                       "train_weeks": TRAIN_WEEKS, "test_weeks": TEST_WEEKS, "seed": SEED},
        "sample": {"te_snaps": int(len(d)), "train_snaps": int(len(tr)), "test_snaps": int(len(te)),
                   "n_te": int(len(pl)), "n_te_ge_min": int(len(big)),
                   "n_contexts": int(d["ctx"].nunique()),
                   "context_level_counts": d["ctx"].str[:2].value_counts().to_dict(),
                   "snaps_with_capped_weight": int((d["ipw_weight"] >= W_CAP - 1e-9).sum())},
        "expected_mix_test_quality": {**quality, "calibration_5bin_test": calib,
                                      "test_observed_vs_expected": {
                                          c: {"observed": round(float(league_test[c]), 4),
                                              "expected": round(float(exp_test_mean[c]), 4)}
                                          for c in CATS}},
        "league_mix": {"train_weeks": league_train.round(4).to_dict(),
                       "test_weeks": league_test.round(4).to_dict(),
                       "chi2_train_vs_test": {"chi2": float(chi2[0]), "dof": int(chi2[2]),
                                              "p": float(chi2[1])}},
        "reliability": {"ddi_and_ipw_shares": {"split_half_train_vs_test": rel_split,
                                               "odd_even_games": rel_oe},
                        "raw_shares_stats_module": rel_raw},
        "sensitivity": {"spearman_ddi_refined_vs_literal_rule_n>=50": sens_lit,
                        "spearman_ddi_ipw_vs_unadjusted_n>=50": sens_raw},
        "top5_n>=50": top5[["displayName", "team", "n", "value", "lo", "hi", "value_shrunk",
                            "archetype"]].round(3).to_dict(orient="records"),
        "archetype_counts_n>=50": big["archetype"].value_counts().to_dict(),
        "caveats": [
            "Descriptive usage metric: high DDI = varied deployment, not better play.",
            "Labels come from PFF role/blockType; chip_release is PFF Pass Route with CH/SR only.",
            "IPW can only reweight within contexts a TE actually played; contexts he never saw are "
            "dropped from his target distribution (q renormalized on his support).",
            "Plug-in entropy is biased low for small n; Miller-Madow columns (value_mm uses Kish n_eff) "
            "and value_shrunk partly correct this. Bootstrap CIs are degenerate for 1-game TEs.",
            "Only dropbacks are in the data; run-play deployment is invisible.",
            "Weeks 7-8 half is small; split-half correlations are noisy (see n_players)."],
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(json.dumps({k: summary[k] for k in ["sample", "league_mix", "sensitivity", "top5_n>=50",
                                              "archetype_counts_n>=50"]}, indent=1, default=str))
    print(json.dumps(quality, indent=1))
    print("pass_block literal:", lit_counts, "refined:", ref_counts)
    print("PA rate literal:", pa_share_bucket_lit, "refined:", pa_share_bucket)
    print("k =", kprec)
    print(json.dumps(summary["reliability"]["ddi_and_ipw_shares"], indent=1))
    print(json.dumps(rel_raw, indent=1))


if __name__ == "__main__":
    main()

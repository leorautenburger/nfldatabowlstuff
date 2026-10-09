"""Red-Zone Conflict Score (RZCS): does the defense have to account for the TE as a route threat,
a run-fit player and a protector at the same time, in red-zone / goal-to-go snaps?

Population: TE snaps (te_plays) where plays.red_zone or plays.goal_to_go is True.

Components per TE
  1. Role ambiguity RA = H(assignment | alignment) / log(3)
       H = sum_l (n_l / N) * H_l,   H_l = -sum_a p(a|l) log p(a|l)
       p(a|l) = (n_la + K * q_la) / (n_l + K),  q_la = league RZ P(assignment=a | alignment=l), K = 5
       assignment in {route, chip_release, pass_block}; alignment in {inline, wing, detached, backfield}
  2. Route threat RT = mean( z(target share on RZ route snaps, beta-shrunk),
                              z(mean RZ coverage gravity on non-targeted route snaps, EB-shrunk) )
       gravity read (read-only) from output/coverage_gravity/plays.csv
  3. Protector PR = share of RZ snaps that are pass_block or chip_release (beta-shrunk)
  4. Run-fit proxy RF = mean( z(inline/wing share, beta-shrunk),
                               z(defendersInBox - league RZ train-week mean for offenseFormation, EB) )
       (no designed runs exist: this is a pre-snap proxy, not observed run-fit value)
  z() uses the mean / SD of the qualified TEs (>= 15 RZ snaps).

Score
  RZCS = 0.5 * RA + 0.5 * (pct(RT) * pct(PR) * pct(RF)) ** (1/3)
  pct(x) = (#ref < x + 0.5 * #ref == x + 0.5) / (N_ref + 1), ref = qualified TEs (in (0, 1)).
  value        = RZCS from raw (unshrunk / unsmoothed) components
  value_shrunk = RZCS from shrunk components (headline)
  lo / hi      = 95% percentile bootstrap (plays resampled within player; league params fixed)

Run: .venv/bin/python -m src.metrics.red_zone_conflict
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import time

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common import io, stats
from src.common.config import OUT, SEED

SLUG = "red_zone_conflict"
OUT_DIR = OUT / SLUG
GRAVITY_FILE = OUT / "coverage_gravity" / "plays.csv"

# ---------------------------------------------------------------- parameters
DIRICHLET_K = 5.0             # pseudo-snaps per alignment toward league RZ P(assignment|alignment)
MIN_SNAPS = 15                # qualified TE (percentile reference + top-5)
MIN_N_HYPER = 10              # players used to estimate shrinkage hyper-parameters
BETA_STRENGTH_CLIP = (2.0, 200.0)
MIN_FORM_PLAYS = 10           # formations with fewer train RZ plays fall back to overall RZ mean
N_BOOT = 1000
HALF_MIN_SNAPS = 8            # min RZ snaps in each half for score-level reliability
ASSIGN = ["route", "chip_release", "pass_block"]
ALIGN = ["inline", "wing", "detached", "backfield"]
W_RA = 0.5                    # weight of role ambiguity vs geometric mean of percentiles


# ---------------------------------------------------------------- helpers
def beta_prior(k, n, min_n=MIN_N_HYPER):
    """Method-of-moments beta-binomial prior (mean, strength) from players with n >= min_n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    m = k.sum() / n.sum()
    h = n >= min_n
    r = k[h] / n[h]
    tau2 = np.var(r) - np.mean(m * (1 - m) / n[h])
    strength = m * (1 - m) / tau2 - 1 if tau2 > 1e-9 else BETA_STRENGTH_CLIP[1]
    return float(m), float(np.clip(strength, *BETA_STRENGTH_CLIP))


def eb_params(means, ns, sd, min_n=MIN_N_HYPER):
    """Hyper-parameters exactly as in stats.eb_shrink_mean (so bootstrap can hold them fixed)."""
    means, ns = np.asarray(means, float), np.asarray(ns, float)
    ok = np.isfinite(means) & (ns > 0)
    se2 = np.where(ok, sd**2 / np.maximum(ns, 1), np.inf)
    h = ok & (ns >= min_n)
    if h.sum() < 3:
        h = ok
    grand = np.sum(means[h] * ns[h]) / np.sum(ns[h])
    tau2 = max(np.var(means[h]) - np.mean(se2[h]), 1e-6)
    return float(grand), float(tau2)


def eb_apply(mean, n, sd, grand, tau2):
    mean, n = np.asarray(mean, float), np.asarray(n, float)
    ok = np.isfinite(mean) & (n > 0)
    w = np.where(ok, tau2 / (tau2 + sd**2 / np.maximum(n, 1)), 0.0)
    return np.where(ok, w * np.nan_to_num(mean) + (1 - w) * grand, grand)


def pct(x, ref):
    """Mid-rank percentile of x against a sorted reference array, kept inside (0, 1)."""
    ref = np.sort(np.asarray(ref, float))
    x = np.asarray(x, float)
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return (lo + 0.5 * (hi - lo) + 0.5) / (len(ref) + 1)


def cond_entropy(counts, q, k):
    """counts: (..., L, A) assignment counts by alignment. Returns H(A|L)/log(A)."""
    n_l = counts.sum(-1, keepdims=True)
    p = (counts + k * q) / np.maximum(n_l + k, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_l = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)
    N = n_l.sum(-2)[..., 0]
    h = (n_l[..., 0] * h_l).sum(-1) / np.maximum(N, 1)
    return h / np.log(counts.shape[-1])


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return None, int(m.sum())
    return float(sps.spearmanr(a[m], b[m]).correlation), int(m.sum())


def r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


# ---------------------------------------------------------------- data
def load():
    p = io.plays()[["gameId", "playId", "week", "is_train", "red_zone", "goal_to_go",
                    "offenseFormation", "defendersInBox", "pff_playAction", "possessionTeam"]]
    t = io.te_plays()[["gameId", "playId", "nflId", "displayName", "alignment", "assignment",
                       "is_target"]]
    d = t.merge(p, on=["gameId", "playId"], how="inner")
    d["rz"] = d["red_zone"].astype(bool) | d["goal_to_go"].astype(bool)

    g = pd.read_csv(GRAVITY_FILE, usecols=["gameId", "playId", "nflId", "red_zone", "gravity",
                                           "is_target"])
    g = g.rename(columns={"red_zone": "g_red_zone", "is_target": "g_is_target"})
    d = d.merge(g, on=["gameId", "playId", "nflId"], how="left")

    d["is_route"] = d["assignment"].isin(["route", "chip_release"])
    d["is_target"] = d["is_target"].astype(bool)
    d["tgt_route"] = np.where(d["is_route"], d["is_target"].astype(float), np.nan)
    # gravity as in the source metric: non-targeted route snaps only
    d["grav"] = np.where(d["is_route"] & ~d["is_target"], d["gravity"], np.nan)
    d["protector"] = d["assignment"].isin(["pass_block", "chip_release"]).astype(float)
    d["inline"] = d["alignment"].isin(["inline", "wing"]).astype(float)
    d["pa"] = d["pff_playAction"].astype(float)
    d["a_code"] = d["assignment"].map({a: i for i, a in enumerate(ASSIGN)}).astype(int)
    d["l_code"] = d["alignment"].map({a: i for i, a in enumerate(ALIGN)}).astype(int)

    # defendersInBox expectation: league RZ mean by offenseFormation, TRAIN weeks only (play level)
    pr = p[(p["red_zone"].astype(bool) | p["goal_to_go"].astype(bool))].copy()
    tr = pr[pr["is_train"] & pr["defendersInBox"].notna()]
    overall = tr["defendersInBox"].mean()
    fs = tr.groupby("offenseFormation")["defendersInBox"].agg(["mean", "size"])
    form_mean = fs.loc[fs["size"] >= MIN_FORM_PLAYS, "mean"].to_dict()
    pr["box_exp"] = pr["offenseFormation"].map(form_mean).fillna(overall)
    d["box_exp"] = d["offenseFormation"].map(form_mean).fillna(overall)
    d["box_resid"] = d["defendersInBox"] - d["box_exp"]

    # test-set check of the box expectation vs naive train mean (play level, RZ)
    te_ = pr[~pr["is_train"] & pr["defendersInBox"].notna()]
    y = te_["defendersInBox"]
    box_model = {
        "n_train_plays": int(len(tr)), "n_test_plays": int(len(te_)),
        "formation_means_train": {k: round(v, 3) for k, v in form_mean.items()},
        "fallback_mean_train": round(overall, 3),
        "formation_mean": {"rmse": r(np.sqrt(mean_squared_error(y, te_["box_exp"]))),
                           "mae": r(mean_absolute_error(y, te_["box_exp"])),
                           "r2": r(r2_score(y, te_["box_exp"]))},
        "naive_train_mean": {"rmse": r(np.sqrt(mean_squared_error(y, np.full(len(y), overall)))),
                             "mae": r(mean_absolute_error(y, np.full(len(y), overall))),
                             "r2": r(r2_score(y, np.full(len(y), overall)))},
    }
    return d, form_mean, overall, box_model


# ---------------------------------------------------------------- league parameters (fixed)
def league_params(rz):
    P = {}
    c = np.zeros((len(ALIGN), len(ASSIGN)))
    np.add.at(c, (rz["l_code"].values, rz["a_code"].values), 1)
    P["q"] = np.where(c.sum(1, keepdims=True) > 0, c / np.maximum(c.sum(1, keepdims=True), 1),
                      1.0 / len(ASSIGN))
    g = rz.groupby("nflId")
    for col, k_col, n_col in [("tgt", "tgt_route", "tgt_route"), ("prot", "protector", "protector"),
                              ("inl", "inline", "inline"), ("pa", "pa", "pa")]:
        k = g[k_col].sum()
        n = g[n_col].count()
        keep = n > 0
        P[col] = beta_prior(k[keep], n[keep])
    for col, v in [("grav", "grav"), ("box", "box_resid")]:
        means, ns = g[v].mean(), g[v].count()
        sd = stats.pooled_within_sd(rz, "nflId", v)
        grand, tau2 = eb_params(means, ns, sd)
        # sanity: identical to the shared helper
        assert np.allclose(eb_apply(means, ns, sd, grand, tau2),
                           stats.eb_shrink_mean(means, ns, sd), equal_nan=True)
        P[col] = (sd, grand, tau2)
    return P


# ---------------------------------------------------------------- component engine
def raw_stats(a_code, l_code, tgt, grav, prot, inl, box, pa):
    """Sufficient statistics; inputs may be (n,) or (B, n) arrays."""
    sh = a_code.shape[:-1]
    counts = np.zeros(sh + (len(ALIGN), len(ASSIGN)))
    for li in range(len(ALIGN)):
        for ai in range(len(ASSIGN)):
            counts[..., li, ai] = ((l_code == li) & (a_code == ai)).sum(-1)

    def ksum(x):
        return np.nansum(x, -1), np.isfinite(x).sum(-1)

    return dict(counts=counts, tgt=ksum(tgt), grav=ksum(grav), prot=ksum(prot), inl=ksum(inl),
                box=ksum(box), pa=ksum(pa), n=a_code.shape[-1])


def components(S, P, shrink=True):
    """Return component dict (shrunk or raw) from sufficient statistics S and league params P."""
    out = {}
    out["role_ambiguity"] = cond_entropy(S["counts"], P["q"], DIRICHLET_K if shrink else 0.0)
    for c in ["tgt", "prot", "inl", "pa"]:
        k, n = S[c]
        m, s = P[c]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = stats.beta_shrink(k, n, m, s) if shrink else np.where(n > 0, k / n, np.nan)
    for c in ["grav", "box"]:
        tot, n = S[c]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
        sd, grand, tau2 = P[c]
        out[c] = eb_apply(mean, n, sd, grand, tau2) if shrink else mean
    return out


def combine(C, R, nan_to_ref_mean=True):
    """Route threat / run-fit z-composites, percentiles and score. R holds the fixed reference."""
    def z(x, key):
        mu, sd = R["z"][key]
        x = np.asarray(x, float)
        if nan_to_ref_mean:
            x = np.where(np.isfinite(x), x, mu)
        return (x - mu) / sd

    rt = 0.5 * z(C["tgt"], "tgt") + 0.5 * z(C["grav"], "grav")
    rf = 0.5 * z(C["inl"], "inl") + 0.5 * z(C["box"], "box")
    pr = np.asarray(C["prot"], float)
    p_rt, p_pr, p_rf = pct(rt, R["ref"]["rt"]), pct(pr, R["ref"]["pr"]), pct(rf, R["ref"]["rf"])
    gm = np.cbrt(p_rt * p_pr * p_rf)
    score = W_RA * C["role_ambiguity"] + (1 - W_RA) * gm
    return dict(route_threat=rt, run_fit=rf, protector=pr, pct_route_threat=p_rt,
                pct_protector=p_pr, pct_run_fit=p_rf, geo_mean_pct=gm, score=score)


def build_reference(C, qual):
    R = {"z": {}}
    for k in ["tgt", "grav", "inl", "box"]:
        x = np.asarray(C[k], float)[qual]
        R["z"][k] = (float(np.nanmean(x)), float(np.nanstd(x)) or 1.0)
    tmp = {"z": R["z"], "ref": {"rt": [0.0], "pr": [0.0], "rf": [0.0]}}
    K = combine(C, tmp)
    R["ref"] = {"rt": K["route_threat"][qual], "pr": K["protector"][qual], "rf": K["run_fit"][qual]}
    return R


PLAY_COLS = ["a_code", "l_code", "tgt_route", "grav", "protector", "inline", "box_resid", "pa"]


def player_arrays(df):
    return {pid: [g[c].to_numpy(float) if c not in ("a_code", "l_code") else g[c].to_numpy()
                  for c in PLAY_COLS] for pid, g in df.groupby("nflId", sort=True)}


def score_table(df, P, R=None, shrink=True):
    """Per-player components + score for any subset of RZ snaps (params P fixed; R built if None)."""
    arrs = player_arrays(df)
    ids = np.array(list(arrs.keys()))
    rows = [components(raw_stats(*a), P, shrink) for a in arrs.values()]
    C = {k: np.array([float(rw[k]) for rw in rows]) for k in rows[0]}
    n = np.array([len(a[0]) for a in arrs.values()])
    if R is None:
        R = build_reference(C, n >= MIN_SNAPS)
    K = combine(C, R)
    t = pd.DataFrame({"nflId": ids, "n": n, **C, **K})
    return t, R


# ---------------------------------------------------------------- league RZ vs non-RZ
def league_comparison(d, P):
    tr = d[d["is_train"] & d["defendersInBox"].notna()]
    form_all = tr.groupby("offenseFormation")["defendersInBox"].mean()
    d = d.assign(box_resid_all=d["defendersInBox"] - d["offenseFormation"].map(form_all)
                 .fillna(tr["defendersInBox"].mean()))
    out = {}
    for name, sub in [("red_zone", d[d["rz"]]), ("non_red_zone", d[~d["rz"]])]:
        c = np.zeros((len(ALIGN), len(ASSIGN)))
        np.add.at(c, (sub["l_code"].values, sub["a_code"].values), 1)
        row = {"n_te_snaps": int(len(sub)), "n_route_snaps": int(sub["is_route"].sum()),
               "role_ambiguity_pooled": r(cond_entropy(c, P["q"], 0.0))}
        for col, lab in [("tgt_route", "target_share_on_routes"), ("protector", "protector_share"),
                         ("inline", "inline_wing_share"), ("pa", "play_action_share")]:
            x = sub[col].dropna()
            row[lab] = r(x.mean())
        for col, lab in [("grav", "mean_gravity_nontarget_routes"),
                         ("defendersInBox", "mean_defenders_in_box"),
                         ("box_resid_all", "box_resid_vs_all_situation_formation_mean")]:
            mc = stats.mean_ci(sub[col])
            row[lab] = {"mean": r(mc["mean"]), "lo": r(mc["lo"]), "hi": r(mc["hi"]), "n": int(mc["n"])}
        out[name] = row
    # simple proportion differences (normal approx; TE snaps treated as independent)
    diffs = {}
    for col, lab in [("tgt_route", "target_share_on_routes"), ("protector", "protector_share"),
                     ("inline", "inline_wing_share"), ("pa", "play_action_share")]:
        a, b = d.loc[d["rz"], col].dropna(), d.loc[~d["rz"], col].dropna()
        diff = a.mean() - b.mean()
        se = np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
        diffs[lab] = {"rz_minus_nonrz": r(diff), "lo": r(diff - 1.96 * se), "hi": r(diff + 1.96 * se)}
    out["differences"] = diffs
    return out


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d, form_mean, overall_box, box_model = load()
    rz = d[d["rz"]].copy()
    P = league_params(rz)

    # point estimates
    tab, R = score_table(rz, P, shrink=True)
    raw, _ = score_table(rz, P, R=R, shrink=False)
    raw_score = raw.set_index("nflId")["score"]

    # bootstrap by play within player (league params P and reference R held fixed)
    rng = np.random.default_rng(SEED)
    lo, hi, se = {}, {}, {}
    for pid, a in player_arrays(rz).items():
        n = len(a[0])
        idx = rng.integers(0, n, size=(N_BOOT, n))
        S = raw_stats(*[x[idx] for x in a])
        sc = combine(components(S, P, True), R)["score"]
        lo[pid], hi[pid] = np.percentile(sc, [2.5, 97.5])
        se[pid] = sc.std(ddof=1)

    # reliability --------------------------------------------------------------
    rel = {"component_play_level": {}, "score_level": {}}
    for col in ["tgt_route", "grav", "protector", "inline", "box_resid", "pa"]:
        x = {}
        for mn in (10, 5):
            sh, nsh = stats.split_half_reliability(rz, "nflId", col, min_n=mn)
            oe, noe = stats.odd_even_reliability(rz, "nflId", col, min_n=mn)
            x[f"min_n_{mn}"] = {"split_half_rho": r(sh), "split_half_n_players": int(nsh),
                                "odd_even_rho": r(oe), "odd_even_n_players": int(noe)}
        rel["component_play_level"][col] = x
    rz["_oe"] = rz.groupby("nflId")["gameId"].rank(method="dense") % 2 == 0
    for lab, col in [("split_half_weeks1-6_vs_7-8", "is_train"), ("odd_even_games", "_oe")]:
        a, _ = score_table(rz[rz[col]], P, R=R)
        b, _ = score_table(rz[~rz[col]], P, R=R)
        m = a.merge(b, on="nflId", suffixes=("_a", "_b"))
        m = m[(m["n_a"] >= HALF_MIN_SNAPS) & (m["n_b"] >= HALF_MIN_SNAPS)]
        res = {}
        for c in ["score", "role_ambiguity", "route_threat", "protector", "run_fit"]:
            rho, k = spearman(m[f"{c}_a"].to_numpy(), m[f"{c}_b"].to_numpy())
            res[c] = r(rho)
        res["n_players"] = int(len(m))
        res["min_snaps_each_half"] = HALF_MIN_SNAPS
        rel["score_level"][lab] = res

    # outputs -------------------------------------------------------------------
    meta = (rz.groupby("nflId")
            .agg(displayName=("displayName", "first"),
                 team=("possessionTeam", lambda s: s.mode().iat[0]),
                 n_route=("is_route", "sum"), n_targets=("is_target", "sum"),
                 n_grav=("grav", "count"), n_box=("box_resid", "count")))
    out = tab.set_index("nflId").join(meta)
    out = out.rename(columns={"tgt": "tgt_share_shrunk", "grav": "gravity_shrunk",
                              "prot": "protector_share_shrunk", "inl": "inline_share_shrunk",
                              "box": "box_resid_shrunk", "pa": "pa_share_shrunk"})
    for c, lab in [("tgt", "tgt_share_raw"), ("grav", "gravity_raw"), ("prot", "protector_share_raw"),
                   ("inl", "inline_share_raw"), ("box", "box_resid_raw"), ("pa", "pa_share_raw"),
                   ("role_ambiguity", "role_ambiguity_raw")]:
        out[lab] = raw.set_index("nflId")[c]
    out["value"] = raw_score
    out["value_shrunk"] = out["score"]
    out["lo"], out["hi"], out["boot_se"] = pd.Series(lo), pd.Series(hi), pd.Series(se)
    out["qualified"] = out["n"] >= MIN_SNAPS
    out["higher_is_better"] = True
    out = out.drop(columns=["score", "protector"]).reset_index()
    first = ["nflId", "displayName", "team", "n", "value", "value_shrunk", "lo", "hi", "boot_se",
             "qualified", "role_ambiguity", "route_threat", "pct_route_threat", "protector_share_shrunk",
             "pct_protector", "run_fit", "pct_run_fit", "geo_mean_pct"]
    out = out[first + [c for c in out.columns if c not in first]]
    out = out.sort_values("value_shrunk", ascending=False)
    out.to_csv(OUT_DIR / "players.csv", index=False)

    play_cols = ["gameId", "playId", "nflId", "displayName", "week", "is_train", "red_zone",
                 "goal_to_go", "alignment", "assignment", "is_route", "is_target", "gravity", "grav",
                 "protector", "inline", "offenseFormation", "defendersInBox", "box_exp", "box_resid",
                 "pff_playAction"]
    rz[play_cols].rename(columns={"grav": "gravity_used"}).to_csv(OUT_DIR / "plays.csv", index=False)

    q = out[out["qualified"]]
    top5 = q.head(5)[["displayName", "team", "n", "value_shrunk", "lo", "hi", "role_ambiguity",
                      "pct_route_threat", "pct_protector", "pct_run_fit"]]
    grav_flag_mismatch = int(((rz["g_red_zone"].notna()) &
                              (rz["g_red_zone"].astype(float) != rz["red_zone"].astype(float))).sum())
    summary = {
        "metric": "Red-Zone Conflict Score", "slug": SLUG, "higher_is_better": True,
        "definition": __doc__.split("Run:")[0].strip(),
        "population": "te_plays rows where plays.red_zone OR plays.goal_to_go (all weeks)",
        "formulas": {
            "role_ambiguity": "RA = [sum_l (n_l/N) * -sum_a p(a|l) ln p(a|l)] / ln 3; "
                              "p(a|l) = (n_la + 5*q_la)/(n_l + 5); q = league RZ P(a|l)",
            "route_threat": "RT = 0.5*z(tgt_share_shrunk) + 0.5*z(gravity_shrunk)",
            "protector": "PR = beta-shrunk share of RZ snaps with assignment in {pass_block, chip_release}",
            "run_fit": "RF = 0.5*z(inline_wing_share_shrunk) + 0.5*z(box_resid_shrunk); "
                       "box_resid = defendersInBox - train-week league RZ mean for offenseFormation",
            "score": "RZCS = 0.5*RA + 0.5*cbrt(pct(RT)*pct(PR)*pct(RF))",
            "z_and_pct_reference": f"qualified TEs (n >= {MIN_SNAPS}); missing component -> reference mean",
            "percentile": "(#ref<x + 0.5*#ref==x + 0.5)/(N_ref+1)",
        },
        "parameters": {
            "dirichlet_prior_strength": DIRICHLET_K, "min_rz_snaps_qualified": MIN_SNAPS,
            "min_n_hyper": MIN_N_HYPER, "beta_strength_clip": BETA_STRENGTH_CLIP,
            "min_formation_train_plays": MIN_FORM_PLAYS, "n_boot": N_BOOT, "seed": SEED,
            "weight_role_ambiguity": W_RA, "half_min_snaps": HALF_MIN_SNAPS,
            "league_rz_p_assignment_given_alignment": {
                ALIGN[i]: {ASSIGN[j]: round(float(P["q"][i][j]), 4) for j in range(len(ASSIGN))}
                for i in range(len(ALIGN))},
            "beta_priors_mean_strength": {k: [round(P[k][0], 4), round(P[k][1], 2)]
                                          for k in ["tgt", "prot", "inl", "pa"]},
            "eb_priors_sd_grand_tau2": {k: [round(x, 5) for x in P[k]] for k in ["grav", "box"]},
            "z_reference_mean_sd": {k: [round(a, 5), round(b, 5)] for k, (a, b) in R["z"].items()},
        },
        "sources": {
            "gravity": f"{GRAVITY_FILE.relative_to(OUT.parent)} (read-only; Coverage Gravity metric) "
                       "columns gameId, playId, nflId, red_zone, gravity, is_target; mean over "
                       "non-targeted RZ route snaps, matching that metric's definition",
            "gravity_red_zone_flag_mismatches_vs_plays": grav_flag_mismatch,
        },
        "sample": {
            "rz_te_snaps": int(len(rz)), "rz_tes": int(rz["nflId"].nunique()),
            "qualified_tes": int(len(q)), "rz_train_snaps": int(rz["is_train"].sum()),
            "rz_test_snaps": int((~rz["is_train"]).sum()),
            "rz_snaps_goal_to_go_only": int((rz["goal_to_go"] & ~rz["red_zone"]).sum()),
            "rz_route_snaps": int(rz["is_route"].sum()), "rz_targets": int(rz["is_target"].sum()),
            "median_rz_snaps_per_te": float(tab["n"].median()),
        },
        "box_expectation_test_metrics": box_model,
        "league_rz_vs_non_rz": league_comparison(d, P),
        "reliability": rel,
        "raw_vs_shrunk_spearman_qualified": r(sps.spearmanr(q["value"], q["value_shrunk"]).correlation),
        "median_ci_width_qualified": r((q["hi"] - q["lo"]).median()),
        "top5_qualified": top5.round(3).to_dict(orient="records"),
        "caveats": [
            "No designed runs in the data (all dropbacks). Run-fit threat is a PRE-SNAP PROXY "
            "(inline/wing share + defendersInBox residual); it is not observed run-fit value. "
            "Play-action share is reported (pa_share_*) but not in the score.",
            "Box residual is confounded with personnel (multi-TE sets) and game state beyond formation.",
            "Small sample: median ~10 RZ snaps per TE, few RZ targets; shrinkage dominates for most TEs "
            "and bootstrap CIs are wide. Treat rankings as descriptive, not a stable trait.",
            "Dirichlet smoothing pulls low-snap TEs toward league-level ambiguity (inflates RA for small n).",
            "Percentiles and z-scores are relative to this 2021 wk1-8 qualified pool only.",
            "Bootstrap holds league priors and reference fixed; it ignores between-play correlation "
            "within games.",
        ],
        "runtime_s": None,
    }
    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    print(f"RZ TE snaps {len(rz)}, TEs {rz['nflId'].nunique()}, qualified {len(q)}")
    print("box model test:", box_model["formation_mean"], "naive:", box_model["naive_train_mean"])
    print("score reliability:", json.dumps(rel["score_level"]))
    print(top5.round(3).to_string(index=False))
    print(f"done in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()

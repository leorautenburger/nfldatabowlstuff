"""Combine every metric's players.csv into one TE table plus the optional composite.

Output: output/te_metrics_final.csv (one row per TE), output/te_metrics_final.json (metadata).
Run:    .venv/bin/python -m src.finalize
"""
import json

import numpy as np
import pandas as pd

from src.common import io
from src.common.config import OUT

# slug -> (column to use, higher_is_better, short name)
# value_shrunk is used where EB shrinkage is informative; raw value where tau^2 collapsed to 0
# (every TE shrunk to the same number), as reported in that metric's summary.json.
METRICS = {
    "route_release_tax":          ("value_shrunk", False, "release_tax_s"),
    "chip_to_separation_return":  ("value",        True,  "csr_yd"),
    "eligible_threat_rate":       ("value_shrunk", True,  "eligible_threat_rate"),
    "coverage_gravity":           ("value_shrunk", True,  "coverage_gravity"),
    "matchup_soe":                ("value_shrunk", True,  "soe_yd"),
    "middle_of_field_access":     ("value_shrunk", True,  "mof_access"),
    "yac_runway":                 ("value",        True,  "yac_runway_yd"),
    "protection_value_added":     ("value",        True,  "pva_per_snap"),
    "edge_seal_sustainability":   ("value",        True,  "edge_seal_oe"),
    "deployment_index":           ("value_shrunk", True,  "deployment_index"),
    "constraint_value":           ("value_shrunk", True,  "constraint_box"),
    "red_zone_conflict":          ("value_shrunk", True,  "rz_conflict"),
}

# Composite (doc "Optional composite"), each group = mean of z-scores of its members.
GROUPS = {
    "route_value": ["soe_yd", "eligible_threat_rate", "mof_access", "release_tax_s", "yac_runway_yd"],
    "protection_value": ["pva_per_snap", "edge_seal_oe"],
    "chip_value": ["csr_yd"],
    "gravity_value": ["coverage_gravity"],
    "versatility_value": ["deployment_index"],
}
WEIGHTS = {"route_value": 0.35, "protection_value": 0.25, "chip_value": 0.20,
           "gravity_value": 0.10, "versatility_value": 0.10}
MIN_SNAPS = 100   # TEs below this are listed but not z-scored / ranked


def main():
    te = io.te_plays()
    base = te.groupby("nflId").agg(displayName=("displayName", "first"), snaps=("playId", "size"),
                                   route_snaps=("assignment", lambda s: s.isin(["route", "chip_release"]).sum()),
                                   chip_snaps=("assignment", lambda s: (s == "chip_release").sum()),
                                   block_snaps=("assignment", lambda s: (s == "pass_block").sum()))
    team = io.player_plays().merge(io.plays()[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"])
    base["team"] = team[team.officialPosition == "TE"].groupby("nflId").possessionTeam.agg(lambda s: s.mode().iat[0])

    meta, missing = {}, []
    for slug, (col, hib, name) in METRICS.items():
        f = OUT / slug / "players.csv"
        if not f.exists():
            missing.append(slug)
            continue
        p = pd.read_csv(f).set_index("nflId")
        base[name] = p[col]
        base[f"{name}_n"] = p["n"]
        if "lo" in p and "hi" in p:
            base[f"{name}_lo"], base[f"{name}_hi"] = p["lo"], p["hi"]
        meta[name] = {"slug": slug, "column": col, "higher_is_better": hib}

    # z-scores among qualified TEs, sign-aligned so higher = better
    q = base.snaps >= MIN_SNAPS
    for name, m in meta.items():
        x = base.loc[q, name]
        z = (base[name] - x.mean()) / x.std()
        base[f"z_{name}"] = (z if m["higher_is_better"] else -z).where(q)

    for g, cols in GROUPS.items():
        zc = [f"z_{c}" for c in cols if f"z_{c}" in base]
        # missing component (e.g. no chip snaps) -> league average (0); count is reported
        base[g] = base[zc].mean(axis=1, skipna=True).fillna(0).where(q)
        base[f"{g}_n_components"] = base[zc].notna().sum(axis=1)
    base["te_dual_threat_value"] = sum(w * base[g] for g, w in WEIGHTS.items())
    base["rank"] = base.te_dual_threat_value.rank(ascending=False, method="min")
    base = base.sort_values("te_dual_threat_value", ascending=False)

    lead = ["displayName", "team", "snaps", "route_snaps", "chip_snaps", "block_snaps", "rank",
            "te_dual_threat_value"] + list(GROUPS)
    base = base[lead + [c for c in base.columns if c not in lead]]
    base.reset_index().to_csv(OUT / "te_metrics_final.csv", index=False, float_format="%.4f")
    json.dump({"metrics": meta, "missing_metrics": missing, "composite_groups": GROUPS,
               "composite_weights": WEIGHTS, "min_snaps_for_z_and_rank": MIN_SNAPS,
               "notes": ["z-scores computed among TEs with >= MIN_SNAPS snaps, sign-aligned (higher = better)",
                         "release_tax_s is lower-is-better, so its z is negated",
                         "missing group components are filled with 0 (league average)",
                         "composite is secondary; component metrics are primary (see tight-end-metrics.md)"]},
              open(OUT / "te_metrics_final.json", "w"), indent=2)
    print("missing:", missing)
    print(base[lead].head(15).round(3).to_string())


if __name__ == "__main__":
    main()

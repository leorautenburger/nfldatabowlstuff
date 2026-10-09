"""
Validate output/te_run_metrics.csv against the cached inputs.

Checks
  a) BDB 2023 contains no run plays (so it can only be dropback context).
  b) Output schema order, unique gsis_id, all run_snaps >= 40.
  c) Rates in [0, 1]; run_tilt == run_snap_rate - pass_snap_rate.
  d) Independent recomputation (direct substring search of offense_players)
     of run_snaps, rush_epa_on, rush_success_on, n_edge_runs for 3 TEs.
  e) run_snaps + pass_snaps <= PFR offense snaps (+5% slack).
  f) Dictionary covers every output column.
Prints PASS/FAIL per check; exits non-zero on any failure.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_run_metrics import BASE, NFLVERSE_DIR, OUT_DIR, SCHEMA  # noqa: E402

PASS_ROLES = {"Coverage", "Pass Block", "Pass Route", "Pass Rush", "Pass"}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    out = pd.read_csv(os.path.join(OUT_DIR, "te_run_metrics.csv"))
    dic = pd.read_csv(os.path.join(OUT_DIR, "te_run_metrics_dictionary.csv"))

    # a) BDB has no runs
    plays = pd.read_csv(os.path.join(BASE, "plays.csv"), usecols=["passResult"])
    roles = set(pd.read_csv(os.path.join(BASE, "pffScoutingData.csv"),
                            usecols=["pff_role"])["pff_role"].dropna().unique())
    check("a) BDB has no run plays",
          plays["passResult"].notna().all() and roles <= PASS_ROLES,
          f"({len(plays)} plays all with passResult; roles={sorted(roles)})")

    # b) schema
    check("b) schema order / unique ids / run_snaps>=40",
          list(out.columns) == SCHEMA and out["gsis_id"].is_unique
          and (out["run_snaps"] >= 40).all(), f"({len(out)} rows)")

    # c) rates + tilt identity
    rate_cols = ["run_snap_rate", "pass_snap_rate", "rush_success_on",
                 "edge_success_on", "short_yardage_conv", "heavy_share",
                 "bdb_pass_block_rate"]
    rates_ok = all(out[c].dropna().between(0, 1).all() for c in rate_cols)
    tilt_ok = ((out["run_snap_rate"] - out["pass_snap_rate"] - out["run_tilt"])
               .abs() <= 0.002).all()
    check("c) rates in [0,1] and run_tilt identity", rates_ok and tilt_ok)

    # d) independent recomputation
    pbp = pd.read_csv(os.path.join(NFLVERSE_DIR, "play_by_play_2021.csv.gz"),
                      usecols=["game_id", "play_id", "season_type", "week",
                               "play_type", "qb_scramble", "two_point_attempt",
                               "aborted_play", "epa", "success", "run_gap"],
                      low_memory=False)
    runs = pbp[(pbp["season_type"] == "REG") & (pbp["week"] <= 8)
               & (pbp["play_type"] == "run") & (pbp["qb_scramble"] == 0)
               & (pbp["two_point_attempt"] == 0) & (pbp["aborted_play"] == 0)]
    part = pd.read_csv(os.path.join(NFLVERSE_DIR, "pbp_participation_2021.csv"),
                       usecols=["nflverse_game_id", "play_id", "offense_players"])
    runs = runs.merge(part, left_on=["game_id", "play_id"],
                      right_on=["nflverse_game_id", "play_id"], how="left")
    s = out.sort_values("run_snaps").reset_index(drop=True)
    picks = [s.iloc[-1], s.iloc[len(s) // 2], s.iloc[0]]
    for p in picks:
        on = runs[runs["offense_players"].fillna("").str.contains(p["gsis_id"], regex=False)]
        ok = (len(on) == p["run_snaps"]
              and abs(on["epa"].mean() - p["rush_epa_on"]) <= 0.001
              and abs(on["success"].mean() - p["rush_success_on"]) <= 0.001
              and (on["run_gap"] == "end").sum() == p["n_edge_runs"])
        check(f"d) recompute {p['name']}", ok,
              f"(run_snaps {len(on)} vs {p['run_snaps']}, epa {on['epa'].mean():.3f} "
              f"vs {p['rush_epa_on']}, edge {(on['run_gap'] == 'end').sum()} "
              f"vs {p['n_edge_runs']})")

    # e) PFR snaps sanity
    m = out.dropna(subset=["pfr_offense_snaps"])
    ok = ((m["run_snaps"] + m["pass_snaps"]) <= 1.05 * m["pfr_offense_snaps"]).all()
    check("e) on-field snaps <= PFR offense snaps (+5%)", ok, f"(n={len(m)})")

    # f) dictionary
    check("f) dictionary covers all columns", set(dic["column"]) == set(SCHEMA))

    if not all(results):
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

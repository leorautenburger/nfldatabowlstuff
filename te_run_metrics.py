"""
Tight-end RUN-game metrics, 2021 regular season weeks 1-8.

Data
  nflverse (downloaded once, cached under data/nflverse/, never re-downloaded
  if the file is present):
    pbp/play_by_play_2021.csv.gz                 play-by-play (epa, success, gaps)
    pbp_participation/pbp_participation_2021.csv on-field gsis IDs, personnel, box
    weekly_rosters/roster_weekly_2021.csv        positions + ID crosswalk
    snap_counts/snap_counts_2021.csv.gz          PFR offensive snaps (cross-check)
  Big Data Bowl 2023 (local data/): pffScoutingData.csv. BDB 2023 contains
  dropbacks only (no run plays), so it is used only as context: each TE's
  pass-block vs route share on dropbacks. No tracking files are loaded.
  No extra dependencies (pandas + numpy + stdlib; CSV / csv.gz only).

Play definitions
  Designed run: season_type == 'REG', week <= 8, play_type == 'run',
    qb_scramble == 0, two_point_attempt == 0, aborted_play == 0.
    (play_type 'run' already excludes no_play penalties, kneels and spikes;
    QB designed runs such as sneaks/options stay in.)
  Dropback: same season/week filters, qb_dropback == 1,
    play_type in {pass, run}, two_point_attempt == 0, aborted_play == 0.
  TE: gsis_id whose modal roster_weekly position over REG weeks 1-8 is 'TE'
    (ties -> TE). On field = gsis_id in participation offense_players.
  A TE's games = distinct (game_id, posteam) where he is on the field for any
    designed run or dropback. Team baselines and "off" samples use only the
    team's plays in those games (handles injuries, inactives and trades).

The 5 metrics (per TE, over designed runs)
  1. run_tilt -- Run Usage Tilt = run_snap_rate - pass_snap_rate, where
     run_snap_rate = share of team designed runs he is on the field for and
     pass_snap_rate = share of team dropbacks. > 0 means the coaches deploy him
     preferentially when they run: a run-game (blocking) role.
  2. rush_epa_onoff -- Rush EPA On/Off = team rush EPA/play with the TE on the
     field minus with him off, each side shrunk toward the TE's team-games
     rush mean with prior weight K = 50 plays:
     shrunk(x, n) = (n*x + K*team_mean) / (n + K). Measures how much more
     efficient the team's run game is with him in the formation.
  3. rush_success_on -- Rush Success Rate On Field = mean pbp `success`
     (EPA > 0) on designed runs with him on the field. Captures how often the
     runs he blocks for keep the offense on schedule.
  4. edge_rush_epa_on -- Edge Run EPA = mean EPA on his on-field designed runs
     with run_gap == 'end' (outside runs, where an in-line TE sets/seals the
     edge). NaN if fewer than 10 such runs.
  5. short_yardage_conv -- Short-Yardage Conversion (FO "power success"
     style) = share of on-field designed runs on 3rd/4th down with <= 2 to go,
     or goal-to-go with <= 2 to go, that gain a first down or a TD. Captures
     his contribution in the most blocking-dependent run situations. NaN if
     fewer than 5 such runs.

Limitations
  These are on-field associations, not blocking grades: nflverse has no TE
  alignment, block assignment or block outcome for runs, and BDB covers
  dropbacks only. On/off confounds personnel/formation choice, game script,
  opponent and teammates (TE1 and TE2 share many snaps; TE1 off-samples are
  small, hence the shrinkage and n_off_runs). run_gap == 'end' is the ball
  carrier's gap, not proof the TE blocked that edge (side is not matched to
  his alignment). Eight weeks => wide uncertainty, especially short-yardage.
  pfr_advstats / ngs rushing / stats_player_week are rusher-week aggregates
  with no play-level key, so they cannot be attributed to TE on-field plays
  and are not used.

Outputs
  output/te_run_metrics.csv             one row per TE with >= 40 run snaps
  output/te_run_metrics_dictionary.csv  definition of every output column
"""

import os
import re
import urllib.request

import numpy as np
import pandas as pd

BASE = r"c:\Users\matth\OneDrive\Documents\UCL - BA Media\NFL Big Data Bowl\data"
OUT_DIR = r"c:\Users\matth\OneDrive\Documents\UCL - BA Media\NFL Big Data Bowl\output"
NFLVERSE_DIR = os.path.join(BASE, "nflverse")
NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(NFLVERSE_DIR, exist_ok=True)

WEEK_MAX = 8
MIN_RUN_SNAPS = 40
SHRINK_K = 50
MIN_EDGE = 10
MIN_SHORT = 5

PBP_COLS = [
    "game_id", "old_game_id", "play_id", "season_type", "week", "posteam",
    "defteam", "play_type", "qb_dropback", "qb_scramble", "two_point_attempt",
    "aborted_play", "epa", "success", "run_gap", "run_location", "down",
    "ydstogo", "goal_to_go", "first_down_rush", "rush_touchdown",
    "rusher_player_id", "rushing_yards",
]
PART_COLS = [
    "nflverse_game_id", "play_id", "offense_players", "offense_personnel",
    "defenders_in_box",
]

METRICS = [
    "run_tilt", "rush_epa_onoff", "rush_success_on", "edge_rush_epa_on",
    "short_yardage_conv",
]

SCHEMA = [
    "gsis_id", "name", "teams", "games", "run_snaps", "pass_snaps",
    "team_runs_in_games", "team_dropbacks_in_games", "run_snap_rate",
    "pass_snap_rate", "run_tilt", "rush_epa_on", "rush_epa_off", "n_off_runs",
    "rush_epa_onoff_raw", "rush_epa_onoff", "rush_success_on",
    "rush_success_onoff_raw", "edge_rush_epa_on", "edge_success_on",
    "n_edge_runs", "short_yardage_conv", "n_short_yardage",
    "short_yardage_conversions", "heavy_share", "avg_box_on", "te_carries",
    "te_rush_yards", "bdb_pass_block_rate", "bdb_dropback_snaps",
    "pfr_offense_snaps",
] + [f"rank_{m}" for m in METRICS]


def fetch(rel_path):
    """Return the local cached path of an nflverse release asset, downloading
    it (atomically via a .part file) only if it is not already cached."""
    local = os.path.join(NFLVERSE_DIR, os.path.basename(rel_path))
    if os.path.exists(local):
        print(f"  cache hit: {os.path.basename(local)}")
        return local
    url = NFLVERSE_URL + rel_path
    print(f"  downloading {url}")
    tmp = local + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, local)
    return local


def load_pbp():
    """Designed runs and dropbacks (REG wk 1-8) joined to participation."""
    pbp = pd.read_csv(fetch("pbp/play_by_play_2021.csv.gz"),
                      usecols=PBP_COLS, low_memory=False)
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["week"] <= WEEK_MAX)]
    clean = (pbp["two_point_attempt"] == 0) & (pbp["aborted_play"] == 0)
    runs = pbp[clean & (pbp["play_type"] == "run") & (pbp["qb_scramble"] == 0)].copy()
    dbs = pbp[clean & (pbp["qb_dropback"] == 1)
              & pbp["play_type"].isin(["pass", "run"])].copy()

    part = pd.read_csv(fetch("pbp_participation/pbp_participation_2021.csv"),
                       usecols=PART_COLS, low_memory=False)
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    runs = runs.merge(part, on=["game_id", "play_id"], how="left", validate="1:1")
    dbs = dbs.merge(part, on=["game_id", "play_id"], how="left", validate="1:1")
    return pbp, runs, dbs


def load_te_ids():
    """gsis_ids whose modal REG wk1-8 roster position is TE (tie -> TE)."""
    ros = pd.read_csv(fetch("weekly_rosters/roster_weekly_2021.csv"),
                      usecols=["gsis_id", "full_name", "position", "week",
                               "game_type", "pfr_id", "gsis_it_id"],
                      low_memory=False)
    ros = ros[(ros["game_type"] == "REG") & (ros["week"] <= WEEK_MAX)
              & ros["gsis_id"].notna()]
    counts = ros.groupby(["gsis_id", "position"]).size().unstack(fill_value=0)
    is_te = counts.get("TE", 0) >= counts.max(axis=1)
    te_gsis = counts.index[is_te & (counts.get("TE", 0) > 0)]
    ros = ros[ros["gsis_id"].isin(te_gsis)].sort_values("week")
    te = ros.groupby("gsis_id").agg(
        name=("full_name", "last"),
        pfr_id=("pfr_id", "last"),
        gsis_it_id=("gsis_it_id", "last"),
    ).reset_index()
    te["gsis_it_id"] = pd.to_numeric(te["gsis_it_id"], errors="coerce").astype("Int64")
    return te


def explode_on_field(plays, te_ids):
    """One row per (play, TE on the field)."""
    x = plays[["game_id", "play_id", "offense_players"]].copy()
    x["gsis_id"] = x["offense_players"].fillna("").str.split(";")
    x = x.explode("gsis_id")
    x = x[x["gsis_id"].isin(set(te_ids["gsis_id"]))]
    x = x.drop(columns="offense_players")
    return x.merge(plays.drop(columns="offense_players"),
                   on=["game_id", "play_id"], how="left")


def n_te(personnel):
    if not isinstance(personnel, str):
        return np.nan
    m = re.search(r"(\d+) TE", personnel)
    return int(m.group(1)) if m else 0


def flag_runs(runs):
    runs["is_edge"] = runs["run_gap"].eq("end")
    short = (runs["ydstogo"] <= 2) & (
        runs["down"].isin([3, 4]) | (runs["goal_to_go"] == 1))
    runs["is_short"] = short
    runs["converted"] = short & (
        (runs["first_down_rush"] == 1) | (runs["rush_touchdown"] == 1))
    n = runs["offense_personnel"].map(n_te)
    runs["heavy"] = np.where(n.isna(), np.nan, (n >= 2).astype(float))
    return runs


def shrunk(x, n, prior):
    return (n * x.fillna(0) + SHRINK_K * prior) / (n + SHRINK_K)


def compute_metrics(runs, dbs, on_runs, on_dbs, te_ids):
    # --- TE games: (gsis_id, game_id, posteam) on any run or dropback ---
    keys = ["gsis_id", "game_id", "posteam"]
    tg = pd.concat([on_runs[keys + ["week"]], on_dbs[keys + ["week"]]])
    tg = tg.drop_duplicates(keys).sort_values(["week", "game_id"])

    teams = (tg.drop_duplicates(["gsis_id", "posteam"])
             .groupby("gsis_id", sort=False)["posteam"].agg("/".join)
             .rename("teams"))

    # --- team baselines restricted to the TE's games ---
    team_r = runs.groupby(["game_id", "posteam"]).agg(
        t_runs=("play_id", "size"), t_epa=("epa", "sum"),
        t_succ=("success", "sum")).reset_index()
    team_d = dbs.groupby(["game_id", "posteam"]).agg(
        t_dbs=("play_id", "size")).reset_index()
    base = (tg.merge(team_r, on=["game_id", "posteam"], how="left")
              .merge(team_d, on=["game_id", "posteam"], how="left")
              .fillna({"t_runs": 0, "t_epa": 0.0, "t_succ": 0.0, "t_dbs": 0}))
    base = base.groupby("gsis_id").agg(
        games=("game_id", "size"), team_runs_in_games=("t_runs", "sum"),
        team_epa=("t_epa", "sum"), team_succ=("t_succ", "sum"),
        team_dropbacks_in_games=("t_dbs", "sum"))

    # --- on-field aggregates ---
    on_runs = on_runs.assign(
        edge_epa=on_runs["epa"].where(on_runs["is_edge"]),
        edge_succ=on_runs["success"].where(on_runs["is_edge"]),
        carry=on_runs["rusher_player_id"].eq(on_runs["gsis_id"]),
    )
    on_runs["carry_yds"] = on_runs["rushing_yards"].where(on_runs["carry"], 0)
    on = on_runs.groupby("gsis_id").agg(
        run_snaps=("play_id", "size"), on_epa=("epa", "sum"),
        rush_epa_on=("epa", "mean"), on_succ=("success", "sum"),
        rush_success_on=("success", "mean"),
        n_edge_runs=("is_edge", "sum"), edge_rush_epa_on=("edge_epa", "mean"),
        edge_success_on=("edge_succ", "mean"),
        n_short_yardage=("is_short", "sum"),
        short_yardage_conversions=("converted", "sum"),
        heavy_share=("heavy", "mean"), avg_box_on=("defenders_in_box", "mean"),
        te_carries=("carry", "sum"), te_rush_yards=("carry_yds", "sum"),
    )
    pass_snaps = on_dbs.groupby("gsis_id").size().rename("pass_snaps")

    df = (base.join(on, how="left").join(pass_snaps, how="left")
              .join(teams, how="left"))
    df = df.fillna({"run_snaps": 0, "pass_snaps": 0, "on_epa": 0.0,
                    "on_succ": 0.0, "n_edge_runs": 0, "n_short_yardage": 0,
                    "short_yardage_conversions": 0, "te_carries": 0,
                    "te_rush_yards": 0})
    for c in ["games", "run_snaps", "pass_snaps", "team_runs_in_games",
              "team_dropbacks_in_games", "n_edge_runs", "n_short_yardage",
              "short_yardage_conversions", "te_carries", "te_rush_yards"]:
        df[c] = df[c].astype(int)

    # 1. usage tilt
    df["run_snap_rate"] = df["run_snaps"] / df["team_runs_in_games"].replace(0, np.nan)
    df["pass_snap_rate"] = df["pass_snaps"] / df["team_dropbacks_in_games"].replace(0, np.nan)
    df["run_tilt"] = df["run_snap_rate"] - df["pass_snap_rate"]

    # 2. rush EPA on/off (raw + shrunk)
    df["n_off_runs"] = df["team_runs_in_games"] - df["run_snaps"]
    off_n = df["n_off_runs"].replace(0, np.nan)
    df["rush_epa_off"] = (df["team_epa"] - df["on_epa"]) / off_n
    df["rush_epa_onoff_raw"] = df["rush_epa_on"] - df["rush_epa_off"]
    team_mean = df["team_epa"] / df["team_runs_in_games"].replace(0, np.nan)
    df["rush_epa_onoff"] = (shrunk(df["rush_epa_on"], df["run_snaps"], team_mean)
                            - shrunk(df["rush_epa_off"], df["n_off_runs"], team_mean))

    # 3. success on (+ raw on-off)
    df["rush_success_onoff_raw"] = (df["rush_success_on"]
                                    - (df["team_succ"] - df["on_succ"]) / off_n)

    # 4. edge run EPA, sample floor
    low_edge = df["n_edge_runs"] < MIN_EDGE
    df.loc[low_edge, ["edge_rush_epa_on", "edge_success_on"]] = np.nan

    # 5. short-yardage conversion, sample floor
    df["short_yardage_conv"] = (df["short_yardage_conversions"]
                                / df["n_short_yardage"].replace(0, np.nan))
    df.loc[df["n_short_yardage"] < MIN_SHORT, "short_yardage_conv"] = np.nan

    df = df.reset_index().merge(te_ids, on="gsis_id", how="left")
    return df


def add_context(df):
    """BDB pass-block share on dropbacks + PFR offensive snap cross-check."""
    pff = pd.read_csv(os.path.join(BASE, "pffScoutingData.csv"),
                      usecols=["gameId", "playId", "nflId", "pff_role"])
    roles = (pff[pff["pff_role"].isin(["Pass Block", "Pass Route"])]
             .groupby(["nflId", "pff_role"]).size().unstack(fill_value=0))
    bdb = pd.DataFrame({
        "bdb_dropback_snaps": roles.sum(axis=1),
        "bdb_pass_block_rate": roles["Pass Block"] / roles.sum(axis=1),
    })
    bdb.index = bdb.index.astype("int64")
    df = df.merge(bdb, left_on="gsis_it_id", right_index=True, how="left")
    df["bdb_dropback_snaps"] = df["bdb_dropback_snaps"].astype("Int64")

    sc = pd.read_csv(fetch("snap_counts/snap_counts_2021.csv.gz"),
                     usecols=["pfr_player_id", "game_type", "week", "offense_snaps"])
    sc = sc[(sc["game_type"] == "REG") & (sc["week"] <= WEEK_MAX)]
    pfr = sc.groupby("pfr_player_id")["offense_snaps"].sum().rename("pfr_offense_snaps")
    df = df.merge(pfr, left_on="pfr_id", right_index=True, how="left")
    df["pfr_offense_snaps"] = df["pfr_offense_snaps"].astype("Int64")
    return df


def finalize(df):
    q = df[df["run_snaps"] >= MIN_RUN_SNAPS].copy()
    for m in METRICS:
        q[f"rank_{m}"] = q[m].rank(ascending=False, method="min").astype("Int64")
    float_cols = ["run_snap_rate", "pass_snap_rate", "run_tilt", "rush_epa_on",
                  "rush_epa_off", "rush_epa_onoff_raw", "rush_epa_onoff",
                  "rush_success_on", "rush_success_onoff_raw",
                  "edge_rush_epa_on", "edge_success_on", "short_yardage_conv",
                  "heavy_share", "avg_box_on", "bdb_pass_block_rate"]
    q[float_cols] = q[float_cols].round(3)
    q = q.sort_values(["rush_epa_onoff", "run_snaps"], ascending=[False, False])
    return q[SCHEMA].reset_index(drop=True)


LIMIT_ONOFF = ("On-field association, not a blocking grade; confounded by "
               "personnel, game script, opponent, teammates.")
DICTIONARY = [
    # column, metric_number, definition, formula, source_columns, limitations
    ("gsis_id", "", "NFL GSIS player id", "", "roster_weekly.gsis_id", ""),
    ("name", "", "Player name", "", "roster_weekly.full_name", ""),
    ("teams", "", "Teams played for, first-appearance order", "'/'-joined posteam",
     "pbp.posteam", ""),
    ("games", "", "Games with >=1 on-field designed run or dropback",
     "count distinct (game_id, posteam)", "participation.offense_players", ""),
    ("run_snaps", "", "Designed runs on field (R_on)", "|R_on|",
     "pbp play_type/qb_scramble/two_point_attempt/aborted_play; participation.offense_players", ""),
    ("pass_snaps", "", "Dropbacks on field (D_on)", "|D_on|",
     "pbp.qb_dropback; participation.offense_players", ""),
    ("team_runs_in_games", "", "Team designed runs in the TE's games", "|R_team|",
     "pbp", ""),
    ("team_dropbacks_in_games", "", "Team dropbacks in the TE's games", "|D_team|",
     "pbp.qb_dropback", ""),
    ("run_snap_rate", "1", "Share of team designed runs on field", "|R_on|/|R_team|",
     "participation.offense_players", ""),
    ("pass_snap_rate", "1", "Share of team dropbacks on field", "|D_on|/|D_team|",
     "participation.offense_players", ""),
    ("run_tilt", "1", "Run Usage Tilt: >0 = deployed preferentially on runs",
     "run_snap_rate - pass_snap_rate", "participation.offense_players; pbp",
     "Deployment signal, not effectiveness."),
    ("rush_epa_on", "2", "Team EPA/designed run with TE on field", "mean epa over R_on",
     "pbp.epa", LIMIT_ONOFF),
    ("rush_epa_off", "2", "Team EPA/designed run with TE off field (his games)",
     "mean epa over R_team - R_on", "pbp.epa", "NaN if n_off_runs == 0; noisy for TE1s."),
    ("n_off_runs", "2", "Team designed runs without TE in his games",
     "|R_team| - |R_on|", "pbp; participation", ""),
    ("rush_epa_onoff_raw", "2", "Unshrunk on minus off rush EPA",
     "rush_epa_on - rush_epa_off", "pbp.epa", "Noisy when n_off_runs small."),
    ("rush_epa_onoff", "2", "Rush EPA On/Off, shrunk (headline metric 2)",
     f"shrunk(on) - shrunk(off); shrunk(x,n)=(n*x+{SHRINK_K}*team_mean)/(n+{SHRINK_K})",
     "pbp.epa", LIMIT_ONOFF),
    ("rush_success_on", "3", "Rush Success Rate On Field", "mean success over R_on",
     "pbp.success", LIMIT_ONOFF),
    ("rush_success_onoff_raw", "3", "On minus off rush success rate (unshrunk)",
     "success(R_on) - success(R_off)", "pbp.success", "NaN if n_off_runs == 0."),
    ("edge_rush_epa_on", "4", "Edge Run EPA: EPA/run on outside (end) runs with TE on",
     f"mean epa over R_on with run_gap=='end'; NaN if n_edge_runs<{MIN_EDGE}",
     "pbp.run_gap, pbp.epa",
     "run_gap is ball-carrier gap; side not matched to TE alignment."),
    ("edge_success_on", "4", "Success rate on edge runs with TE on",
     f"mean success over edge R_on; NaN if n_edge_runs<{MIN_EDGE}",
     "pbp.run_gap, pbp.success", "As edge_rush_epa_on."),
    ("n_edge_runs", "4", "Edge designed runs with TE on", "count", "pbp.run_gap", ""),
    ("short_yardage_conv", "5", "Short-Yardage Conversion (power success)",
     f"conversions / n_short_yardage; NaN if n_short_yardage<{MIN_SHORT}",
     "pbp.down, ydstogo, goal_to_go, first_down_rush, rush_touchdown",
     "Very small samples over 8 weeks."),
    ("n_short_yardage", "5", "Short-yardage designed runs with TE on",
     "(down in 3,4 & ydstogo<=2) or (goal_to_go & ydstogo<=2)",
     "pbp.down, ydstogo, goal_to_go", ""),
    ("short_yardage_conversions", "5", "Short-yardage runs converted",
     "first_down_rush==1 or rush_touchdown==1", "pbp.first_down_rush, rush_touchdown", ""),
    ("heavy_share", "", "Share of R_on in 2+ TE personnel (context)",
     "mean(n_TE>=2) over R_on with personnel", "participation.offense_personnel", ""),
    ("avg_box_on", "", "Mean defenders in box on R_on (context)", "mean",
     "participation.defenders_in_box", ""),
    ("te_carries", "", "Designed-run carries by the TE himself", "rusher_player_id==gsis_id",
     "pbp.rusher_player_id", ""),
    ("te_rush_yards", "", "Rushing yards on those carries", "sum rushing_yards",
     "pbp.rushing_yards", ""),
    ("bdb_pass_block_rate", "", "BDB 2023 share of dropback snaps spent pass blocking",
     "Pass Block / (Pass Block + Pass Route)", "BDB pffScoutingData.pff_role",
     "Dropbacks only (BDB has no runs); joined via gsis_it_id."),
    ("bdb_dropback_snaps", "", "BDB dropback snaps (Pass Block + Pass Route)", "count",
     "BDB pffScoutingData", ""),
    ("pfr_offense_snaps", "", "PFR offensive snaps REG wk1-8 (cross-check)", "sum",
     "snap_counts.offense_snaps", "Includes penalties/2pt; joined via pfr_id."),
] + [
    (f"rank_{m}", str(i + 1), f"Rank of {m} among qualified TEs (1 = highest)",
     "rank(desc, method='min'); NaN metric -> NaN rank", m, "")
    for i, m in enumerate(METRICS)
]


def safe_write(frame, path):
    """Write CSV; if the target is locked (open in Excel), fall back to a
    timestamped filename so a completed run is never discarded."""
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        alt = f"{root}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{ext}"
        frame.to_csv(alt, index=False)
        print(f"  (target locked, wrote {alt} instead -- close the open "
              f"copy of {os.path.basename(path)})")
        return alt


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print("Loading data ...")
    pbp, runs, dbs = load_pbp()
    te_ids = load_te_ids()
    runs = flag_runs(runs)
    on_runs = explode_on_field(runs, te_ids)
    on_dbs = explode_on_field(dbs, te_ids)

    print(f"\nGames (REG wk1-{WEEK_MAX}): {pbp['game_id'].nunique()}")
    print(f"Designed runs: {len(runs)}  |  dropbacks: {len(dbs)}")
    print(f"Participation coverage: runs {runs['offense_players'].notna().mean():.3%}, "
          f"dropbacks {dbs['offense_players'].notna().mean():.3%}")
    print(f"TE ids (modal position TE): {len(te_ids)}  |  TEs on field for >=1 run: "
          f"{on_runs['gsis_id'].nunique()}")

    df = compute_metrics(runs, dbs, on_runs, on_dbs, te_ids)
    df = add_context(df)
    out = finalize(df)

    out_path = safe_write(out, os.path.join(OUT_DIR, "te_run_metrics.csv"))
    dic = pd.DataFrame(DICTIONARY, columns=["column", "metric_number", "definition",
                                            "formula", "source_columns", "limitations"])
    dic_path = safe_write(dic, os.path.join(OUT_DIR, "te_run_metrics_dictionary.csv"))
    print(f"\nWrote {len(out)} qualified TEs (run_snaps >= {MIN_RUN_SNAPS}) to {out_path}")
    print(f"Wrote dictionary ({len(dic)} columns) to {dic_path}")

    # ---------------- verification evidence ----------------
    print("\n=== VERIFICATION ===")
    print(f"Row count: {len(out)}  (TEs >= 1 run snap: {len(df)})")
    print(f"Columns ({len(out.columns)}): {', '.join(out.columns)}")
    assert list(out.columns) == SCHEMA
    assert out["gsis_id"].is_unique
    assert set(dic["column"]) == set(SCHEMA), "dictionary does not cover schema"
    for m in METRICS:
        nn = out[m].notna().sum()
        print(f"  {m:20s} non-null {nn:3d}/{len(out)}")
        assert nn > 0, f"{m} is all NaN"
    for c in ["run_snap_rate", "pass_snap_rate", "rush_success_on", "short_yardage_conv"]:
        assert out[c].dropna().between(0, 1).all(), c

    # Reconciliation 1: team designed-run totals from pbp vs. a fresh filter.
    team_tot = runs.groupby("posteam").size()
    raw = pbp[(pbp["play_type"] == "run") & (pbp["qb_scramble"] == 0)
              & (pbp["two_point_attempt"] == 0) & (pbp["aborted_play"] == 0)]
    assert team_tot.sum() == len(raw) and (team_tot == raw.groupby("posteam").size()).all()
    print(f"Recon 1: {len(team_tot)} teams, designed runs per team "
          f"{team_tot.min()}-{team_tot.max()}, sum {team_tot.sum()} == pbp filter {len(raw)}")
    # Reconciliation 2: on + off == team runs in his games, every TE.
    assert (df["run_snaps"] + df["n_off_runs"] == df["team_runs_in_games"]).all()
    print("Recon 2: run_snaps + n_off_runs == team_runs_in_games for all TEs: OK")
    # Reconciliation 3: TEs who appeared in every team game -> team season total.
    team_games = runs.groupby("posteam")["game_id"].nunique()
    full = out[~out["teams"].str.contains("/")].copy()
    full = full[full["games"] == full["teams"].map(team_games)]
    match = (full["team_runs_in_games"] == full["teams"].map(team_tot)).all()
    assert match
    print(f"Recon 3: {len(full)} single-team TEs who played every team game have "
          f"team_runs_in_games == pbp team total: OK")
    # Reconciliation 4: on-field detection vs PFR snap counts.
    chk = out.dropna(subset=["pfr_offense_snaps"])
    ratio = (chk["run_snaps"] + chk["pass_snaps"]) / chk["pfr_offense_snaps"].astype(float)
    print(f"Recon 4: (run+pass snaps)/PFR offense snaps median {ratio.median():.3f}, "
          f"range {ratio.min():.3f}-{ratio.max():.3f} (n={len(chk)})")

    print("\n=== Metric summary ===")
    print(out[METRICS].describe().round(3).to_string())

    show = ["name", "teams", "games", "run_snaps", "pass_snaps"] + METRICS
    print("\n=== Sanity rows ===")
    names = ["George Kittle", "Mark Andrews", "Travis Kelce", "Darren Waller",
             "Kyle Pitts", "Dawson Knox", "Marcedes Lewis", "Rob Gronkowski"]
    print(out[out["name"].isin(names)][show].to_string(index=False))

    print("\n=== Top 12 by rush_epa_onoff ===")
    print(out[show].head(12).to_string(index=False))
    for m in METRICS:
        top = out.dropna(subset=[m]).sort_values(m, ascending=False).head(10)
        print(f"\n--- Top 10 {m} ---")
        print(top[["name", "teams", "run_snaps", m]].to_string(index=False))


if __name__ == "__main__":
    main()

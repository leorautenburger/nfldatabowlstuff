"""Build the shared processed dataset used by every metric module.

Outputs (data/processed/):
  plays.parquet            play context + split + key frames + target + pressure
  player_plays.parquet     PFF scouting x players x alignment at snap
  tracking/<gameId>.parquet standardized tracking, snap -> last frame
  receiver_frames.parquet  per-frame features for every route runner
  routes.parquet           per route-runner-play summary (release, throw-frame separation)
  te_plays.parquet         per TE-play assignment table

Run:  .venv/bin/python -m src.prep
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.common.config import (FIELD_W, FPS, PROC, RAW, RELEASE_MIN_DISP, RELEASE_MIN_SPEED,
                               RELEASE_SUSTAIN, TEST_WEEKS, TRACK_DIR, TRAIN_WEEKS, RED_ZONE_YARDS)

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
THROW_EVENTS = {"pass_forward", "autoevent_passforward"}
END_EVENTS = {"qb_sack", "qb_strip_sack", "run", "autoevent_passinterrupted", "pass_shovel",
              "fumble", "out_of_bounds"}


# --------------------------------------------------------------------------- static tables
def load_static():
    games = pd.read_csv(RAW / "games.csv")
    plays = pd.read_csv(RAW / "plays.csv")
    players = pd.read_csv(RAW / "players.csv")
    pff = pd.read_csv(RAW / "pffScoutingData.csv")
    return games, plays, players, pff


_SUFFIX = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$")


def _last_name(display):
    return _SUFFIX.sub("", str(display)).split(" ", 1)[-1]


def parse_targets(plays, pp):
    """Infer targeted receiver from playDescription by matching offensive players on the play.

    Description format: '... pass short right to T.Kelce to KC 45 for 10 yards' or
    '... pass incomplete deep left intended for X.Name'. We take the earliest-matching
    offensive player name after the word 'pass'.
    """
    off = pp[pp.pff_role.isin(["Pass Route", "Pass Block"])][["gameId", "playId", "nflId", "displayName"]]
    by_play = {k: g for k, g in off.groupby(["gameId", "playId"])}
    out = {}
    for r in plays[["gameId", "playId", "playDescription", "passResult"]].itertuples(index=False):
        if r.passResult not in ("C", "I", "IN"):
            continue
        desc = str(r.playDescription)
        m = re.search(r"\bpass\b", desc)
        if not m:
            continue
        tail = desc[m.end():]
        best = None
        for p in by_play.get((r.gameId, r.playId), pd.DataFrame()).itertuples(index=False):
            first = str(p.displayName).split(" ")[0]
            last = _last_name(p.displayName)
            pat = re.compile(r"\b" + re.escape(first[0]) + r"[A-Za-z']*\.\s?" + re.escape(last) + r"\b")
            mm = pat.search(tail)
            if mm and (best is None or mm.start() < best[0]):
                best = (mm.start(), p.nflId)
        if best:
            out[(r.gameId, r.playId)] = best[1]
    return pd.Series(out, name="targetNflId")


# --------------------------------------------------------------------------- tracking
def _standardize(tr):
    left = tr.playDirection == "left"
    tr.loc[left, "x"] = 120 - tr.loc[left, "x"]
    tr.loc[left, "y"] = FIELD_W - tr.loc[left, "y"]
    for c in ("o", "dir"):
        tr.loc[left, c] = (tr.loc[left, c] + 180) % 360
    rad = np.deg2rad(tr["dir"].fillna(0))
    tr["vx"] = tr["s"] * np.sin(rad)   # +x = downfield for offense
    tr["vy"] = tr["s"] * np.cos(rad)
    return tr


def _key_frames(tr):
    ev = tr[["playId", "frameId", "event"]].drop_duplicates()
    ev = ev[ev.event.notna() & (ev.event != "None")]
    g = tr.groupby("playId").frameId.max().rename("last_frame").to_frame()
    g["snap_frame"] = ev[ev.event.isin(SNAP_EVENTS)].groupby("playId").frameId.min()
    g["throw_frame"] = ev[ev.event.isin(THROW_EVENTS)].groupby("playId").frameId.min()
    g["stop_frame"] = ev[ev.event.isin(END_EVENTS)].groupby("playId").frameId.min()
    g["play_action_frame"] = ev[ev.event == "play_action"].groupby("playId").frameId.min()
    # end of the dropback = throw, else sack/run/interrupt, else last tracked frame
    g["end_frame"] = g.throw_frame.fillna(g.stop_frame).fillna(g.last_frame)
    g["end_event"] = np.select(
        [g.throw_frame.notna(), g.stop_frame.notna()], ["throw", "other_stop"], "last_frame")
    return g.reset_index()


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB, plus projection parameter t in [0,1]."""
    abx, aby = bx - ax, by - ay
    L2 = abx**2 + aby**2
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / np.where(L2 == 0, 1, L2), 0, 1)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby)), t


def process_game(args):
    game_id, pp_game, plays_game = args
    tr = pd.read_csv(RAW / "tracking" / f"tracking_{game_id}.csv")
    tr = _standardize(tr)
    kf = _key_frames(tr)

    ball = tr[tr.team == "football"].merge(kf[["playId", "snap_frame"]], on="playId")
    los = ball[ball.frameId == ball.snap_frame].groupby("playId")[["x", "y"]].first()
    los.columns = ["los_x", "ball_y"]
    kf = kf.merge(los, left_on="playId", right_index=True, how="left")

    tr = tr.merge(kf[["playId", "snap_frame", "end_frame", "los_x", "ball_y"]], on="playId")
    tr = tr[tr.snap_frame.notna() & (tr.frameId >= tr.snap_frame)].copy()
    tr["t"] = ((tr.frameId - tr.snap_frame) / FPS).round(1)
    tr["x_rel"] = tr.x - tr.los_x
    tr["nflId"] = tr.nflId.astype("Int64")

    # side of ball per player from PFF roles
    roles = pp_game[["playId", "nflId", "pff_role", "officialPosition"]]
    tr = tr.merge(roles, on=["playId", "nflId"], how="left")
    off_team = plays_game.set_index("playId").possessionTeam
    tr["is_off"] = tr.team.values == tr.playId.map(off_team).values
    tr["is_def"] = (tr.team != "football") & ~tr.is_off

    keep = ["gameId", "playId", "nflId", "frameId", "t", "team", "is_off", "is_def", "pff_role",
            "officialPosition", "x", "y", "x_rel", "s", "a", "dis", "o", "dir", "vx", "vy", "event"]
    out = tr[keep].copy()
    for c in ["x", "y", "x_rel", "s", "a", "dis", "o", "dir", "vx", "vy", "t"]:
        out[c] = out[c].astype("float32")
    out.to_parquet(TRACK_DIR / f"{game_id}.parquet", index=False)

    # ---------------- receiver frame features (all route runners), snap -> end_frame
    upto = tr[tr.frameId <= tr.end_frame]
    rec = upto[upto.pff_role == "Pass Route"][["playId", "frameId", "nflId", "t", "x", "y", "x_rel",
                                               "s", "a", "vx", "vy", "dir", "o"]]
    dfn = upto[upto.is_def][["playId", "frameId", "nflId", "x", "y"]].rename(
        columns={"nflId": "defId", "x": "dx", "y": "dy"})
    qb = upto[upto.pff_role == "Pass"].groupby(["playId", "frameId"])[["x", "y"]].first()
    qb.columns = ["qb_x", "qb_y"]
    rec = rec.merge(qb, left_on=["playId", "frameId"], right_index=True, how="left")

    pairs = rec.merge(dfn, on=["playId", "frameId"])
    pairs["d"] = np.hypot(pairs.x - pairs.dx, pairs.y - pairs.dy)
    lane, tpar = _seg_dist(pairs.dx.values, pairs.dy.values, pairs.qb_x.values, pairs.qb_y.values,
                           pairs.x.values, pairs.y.values)
    # ignore defenders sitting on the QB end of the segment (pass rushers) for lane clearance
    pairs["lane"] = np.where(tpar > 0.15, lane, np.inf)
    key = ["playId", "frameId", "nflId"]
    pairs = pairs.sort_values(key + ["d"])
    pairs["w3"] = pairs.d <= 3
    pairs["w5"] = pairs.d <= 5
    pairs["rank"] = pairs.groupby(key, sort=False).cumcount()
    grp = pairs.groupby(key, sort=False)
    feat = grp.agg(sep=("d", "first"), n_def_3=("w3", "sum"), n_def_5=("w5", "sum"),
                   lane_clear=("lane", "min"), nearest_def_id=("defId", "first"),
                   nd_x=("dx", "first"), nd_y=("dy", "first"))
    feat["sep2"] = pairs[pairs["rank"] == 1].set_index(key).d
    rec = rec.merge(feat.reset_index(), on=key, how="left")
    rec["dist_qb"] = np.hypot(rec.x - rec.qb_x, rec.y - rec.qb_y)
    rec.insert(0, "gameId", game_id)

    # ---------------- route summaries
    rec = rec.sort_values(["playId", "nflId", "frameId"])
    snap_pos = rec.groupby(["playId", "nflId"])[["x_rel", "y"]].transform("first")
    disp = np.maximum(rec.x_rel - snap_pos.x_rel, 0)
    lat = (rec.y - snap_pos.y).abs()
    moved = (np.maximum(disp, lat) >= RELEASE_MIN_DISP)
    fast = rec.s >= RELEASE_MIN_SPEED
    # sustained speed: this frame and next RELEASE_SUSTAIN-1 frames fast
    sustained = fast.groupby([rec.playId, rec.nflId]).transform(
        lambda s: s[::-1].rolling(RELEASE_SUSTAIN, min_periods=RELEASE_SUSTAIN).min()[::-1].fillna(0).astype(bool))
    rec["released"] = (moved & sustained).groupby([rec.playId, rec.nflId]).cummax()
    rel = rec[rec.released].groupby(["playId", "nflId"]).agg(release_frame=("frameId", "first"),
                                                             release_t=("t", "first"))
    last = rec.groupby(["playId", "nflId"]).tail(1).set_index(["playId", "nflId"])
    first = rec.groupby(["playId", "nflId"]).head(1).set_index(["playId", "nflId"])
    rs = pd.DataFrame({
        "x_rel_snap": first.x_rel, "y_snap": first.y,
        "sep_snap": first.sep, "nd_dx_snap": first.nd_x - first.x, "nd_dy_snap": first.nd_y - first.y,
        "t_end": last.t, "x_rel_end": last.x_rel, "y_end": last.y, "sep_end": last.sep,
        "n_def_3_end": last.n_def_3, "n_def_5_end": last.n_def_5, "lane_end": last.lane_clear,
        "s_end": last.s, "vx_end": last.vx, "vy_end": last.vy,
        "max_depth": rec.groupby(["playId", "nflId"]).x_rel.max(),
        "min_sep": rec.groupby(["playId", "nflId"]).sep.min(),
        "max_sep": rec.groupby(["playId", "nflId"]).sep.max(),
    }).join(rel)
    rs = rs.reset_index()
    rs.insert(0, "gameId", game_id)
    kf.insert(0, "gameId", game_id)
    return rec, rs, kf


# --------------------------------------------------------------------------- main
def main(n_workers=6):
    PROC.mkdir(parents=True, exist_ok=True)
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    games, plays, players, pff = load_static()

    pp = pff.merge(players[["nflId", "officialPosition", "displayName", "height", "weight"]],
                   on="nflId", how="left")
    jobs = [(gid, pp[pp.gameId == gid], plays[plays.gameId == gid]) for gid in games.gameId]
    recs, rss, kfs = [], [], []
    with ProcessPoolExecutor(n_workers) as ex:
        for i, (rec, rs, kf) in enumerate(ex.map(process_game, jobs)):
            recs.append(rec); rss.append(rs); kfs.append(kf)
            print(f"\rgames {i + 1}/{len(jobs)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    rec = pd.concat(recs, ignore_index=True)
    rs = pd.concat(rss, ignore_index=True)
    kf = pd.concat(kfs, ignore_index=True)

    # ---------------- plays table
    p = plays.merge(games[["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"]], on="gameId")
    p["split"] = np.where(p.week.isin(TRAIN_WEEKS), "train", np.where(p.week.isin(TEST_WEEKS), "test", "other"))
    p["is_train"] = p.split == "train"
    p = p.merge(kf, on=["gameId", "playId"], how="left")
    p["time_to_throw"] = (p.throw_frame - p.snap_frame) / FPS
    p["time_to_end"] = (p.end_frame - p.snap_frame) / FPS
    p["yards_to_goal"] = 110 - p.los_x
    p["red_zone"] = p.yards_to_goal <= RED_ZONE_YARDS
    p["goal_to_go"] = p.yardsToGo >= p.yards_to_goal - 0.5
    p["targetNflId"] = p.set_index(["gameId", "playId"]).index.map(parse_targets(p, pp)).astype("Int64")

    d = pp[pp.pff_role == "Pass Rush"]
    pr = pp.assign(pressure=pp[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).max(axis=1))
    agg = pd.DataFrame({
        "pressure": pr.groupby(["gameId", "playId"]).pressure.max(),
        "n_rushers": d.groupby(["gameId", "playId"]).size(),
        "n_pass_blockers": pp[pp.pff_role == "Pass Block"].groupby(["gameId", "playId"]).size(),
        "n_routes": pp[pp.pff_role == "Pass Route"].groupby(["gameId", "playId"]).size(),
        "n_te_on_field": pp[pp.officialPosition == "TE"].groupby(["gameId", "playId"]).size(),
    }).reset_index()
    p = p.merge(agg, on=["gameId", "playId"], how="left")
    p[["n_rushers", "n_pass_blockers", "n_routes", "n_te_on_field"]] = \
        p[["n_rushers", "n_pass_blockers", "n_routes", "n_te_on_field"]].fillna(0).astype(int)
    p.to_parquet(PROC / "plays.parquet", index=False)

    # ---------------- player_plays with alignment at snap
    snap_pos = []
    for gid in games.gameId:
        t = pd.read_parquet(TRACK_DIR / f"{gid}.parquet", columns=["gameId", "playId", "nflId", "t", "x_rel", "y"])
        snap_pos.append(t[(t.t == 0) & t.nflId.notna()])
    snap_pos = pd.concat(snap_pos).drop(columns="t").rename(columns={"x_rel": "x_rel_snap", "y": "y_snap"})
    snap_pos["nflId"] = snap_pos.nflId.astype(int)
    ppl = pp.merge(snap_pos, on=["gameId", "playId", "nflId"], how="left")
    ppl = ppl.merge(p[["gameId", "playId", "ball_y", "split", "is_train", "week", "targetNflId",
                       "passResult"]], on=["gameId", "playId"], how="left")
    ppl["lat_from_ball"] = ppl.y_snap - ppl.ball_y
    ppl["alignment"] = _alignment(ppl)
    ppl["is_target"] = (ppl.nflId == ppl.targetNflId).fillna(False).astype(bool)
    ppl["is_catch"] = ppl.is_target & (ppl.passResult == "C")
    ppl.to_parquet(PROC / "player_plays.parquet", index=False)

    # ---------------- receiver frames / routes (attach context)
    ctx = ppl[["gameId", "playId", "nflId", "officialPosition", "pff_positionLinedUp", "pff_role",
               "pff_blockType", "alignment", "lat_from_ball", "is_target", "is_catch"]]
    rs = rs.merge(ctx, on=["gameId", "playId", "nflId"], how="left").merge(
        p[["gameId", "playId", "split", "is_train", "time_to_throw", "end_event"]], on=["gameId", "playId"])
    rs["released_flag"] = rs.release_t.notna()
    rs = _receiver_order(rs)
    rs.to_parquet(PROC / "routes.parquet", index=False)
    rec = rec.merge(p[["gameId", "playId", "split", "is_train"]], on=["gameId", "playId"])
    rec.to_parquet(PROC / "receiver_frames.parquet", index=False)

    # ---------------- TE plays
    te = ppl[ppl.officialPosition == "TE"].copy()
    te["assignment"] = np.select(
        [(te.pff_role == "Pass Route") & te.pff_blockType.isin(["CH", "SR"]),
         te.pff_role == "Pass Route",
         te.pff_role == "Pass Block"],
        ["chip_release", "route", "pass_block"], "other")
    te = te.merge(rs[["gameId", "playId", "nflId", "release_t", "release_frame", "max_depth",
                      "x_rel_end", "sep_end", "receiver_num", "n_rec_side"]],
                  on=["gameId", "playId", "nflId"], how="left")
    te.to_parquet(PROC / "te_plays.parquet", index=False)

    print("plays", len(p), "| targets parsed", p.targetNflId.notna().sum(), "of",
          p.passResult.isin(["C", "I", "IN"]).sum(), "| routes", len(rs), "| receiver frames", len(rec),
          "| TE plays", len(te))
    print(te.assignment.value_counts().to_string())


def _alignment(ppl):
    """inline / wing / detached / backfield from PFF lined-up position (+ geometry fallback)."""
    pos = ppl.pff_positionLinedUp.fillna("")
    a = np.select(
        [pos.isin(["TE-L", "TE-R", "TE-iL", "TE-iR"]),
         pos.isin(["TE-oL", "TE-oR"]),
         pos.str.contains("WR"),
         pos.str.startswith(("FB", "HB")) | (pos == "QB")],
        ["inline", "wing", "detached", "backfield"], "")
    geo = np.where(ppl.x_rel_snap < -2.5, "backfield", np.where(ppl.lat_from_ball.abs() > 8, "detached", "inline"))
    return np.where(a == "", geo, a)


def _receiver_order(rs):
    """#1/#2/#3 receiver to each side (1 = widest) and receivers on that side."""
    rs["side"] = np.sign(rs.lat_from_ball).fillna(0)
    rs["abs_lat"] = rs.lat_from_ball.abs()
    rs["receiver_num"] = rs.groupby(["gameId", "playId", "side"]).abs_lat.rank(ascending=False, method="first")
    rs["n_rec_side"] = rs.groupby(["gameId", "playId", "side"]).nflId.transform("size")
    return rs


if __name__ == "__main__":
    main()

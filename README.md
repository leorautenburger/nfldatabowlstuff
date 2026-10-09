# NFL Data Bowl Stuff

This repository contains reproducible exploratory metrics for tight ends in the
NFL Big Data Bowl regional-event dataset.

## Protection-to-Availability prototype

`pav_prototype.py` evaluates tight ends labeled as `CH` (chip block) and `Pass
Route` in PFF scouting data. It emits the inputs for a descriptive
**dual-threat window**:

```text
dualThreatWindowSeconds =
  max(protectionWindowSeconds - releaseTimeSeconds, 0) * availabilityScore
```

It measures the post-release time before the QB faces a nearby defender, scaled
by the tight end's separation when the ball is thrown, a QB threat occurs, or
the play ends.

This is not yet causal **Protection-to-Availability Value**. It does not
estimate the counterfactual extra time created by the tight end; that requires
an expected-pressure model. The output makes each component available for that
next stage.

### Setup

Download this dataset so the directory has this structure:

```text
data/
  players.csv
  pffScoutingData.csv
  tracking/
    tracking_<gameId>.csv
```

Install the dependency and run one game:

```bash
python3 -m pip install -r requirements.txt
python3 pav_prototype.py \
  --data-dir /path/to/data \
  --game-id 2021090900 \
  --output output/te_chip_release_pav.csv
```

Omit `--game-id` to process every tracking file represented by a tight-end
chip-and-release record.

### Output fields

| Field | Meaning |
| --- | --- |
| `releaseTimeSeconds` | Time from snap to sustained route movement. |
| `protectionWindowSeconds` | Time from snap to the first labeled pass rusher that enters a 2-yard, closing threat zone around the QB, or play end. |
| `nearestDefenderSeparationYards` | TE's nearest-defender separation at the evaluation frame. |
| `availabilityScore` | Separation scaled from 0 to 1, reaching 1 at 3 yards. |
| `dualThreatWindowSeconds` | Post-release protection window weighted by receiving availability. |

Thresholds are explicit constants at the top of the script. They should be
tested through sensitivity analysis before drawing player-level conclusions.

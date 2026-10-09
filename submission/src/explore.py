"""One-off exploration of the raw data to inform metric definitions."""
import pandas as pd

D = "data/repo/data"
games = pd.read_csv(f"{D}/games.csv")
plays = pd.read_csv(f"{D}/plays.csv")
players = pd.read_csv(f"{D}/players.csv")
pff = pd.read_csv(f"{D}/pffScoutingData.csv")

print(games.week.value_counts().sort_index())
print(plays.passResult.value_counts(dropna=False))
print(plays.dropBackType.value_counts(dropna=False))
print(plays.offenseFormation.value_counts(dropna=False))
print(plays.pff_passCoverage.value_counts(dropna=False))

te_ids = set(players.loc[players.officialPosition == "TE", "nflId"])
te = pff[pff.nflId.isin(te_ids)]
print("TE player-plays:", len(te), "unique TEs:", te.nflId.nunique())
print(pd.crosstab(te.pff_role, te.pff_blockType.fillna("NA")))
print(te.pff_positionLinedUp.value_counts())
print(pff.pff_positionLinedUp.value_counts().to_string())

tr = pd.read_csv(f"{D}/tracking/tracking_2021090900.csv")
print(tr.event.value_counts())
# frames after pass_forward per play
ev = tr[tr.team == "football"][["playId", "frameId", "event"]]
print(ev[ev.event != "None"].groupby("playId").event.apply(list).head(15).to_string())
print("frames per play", tr.groupby("playId").frameId.max().describe())

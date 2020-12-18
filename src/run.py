import datetime
import math

import numpy as np
import pandas as pd

from .config import PLAYERS


def main():
    df2016 = pd.read_csv(r"C:\Users\Michael.Copeland\Projects\nba\NBA Stats 2016.csv")
    df2017 = pd.read_csv(r"C:\Users\Michael.Copeland\Projects\nba\NBA Stats 2017.csv")
    df2018 = pd.read_csv(r"C:\Users\Michael.Copeland\Projects\nba\NBA Stats 2018.csv")
    df2019 = pd.read_csv(r"C:\Users\Michael.Copeland\Projects\nba\NBA Stats 2019.csv")

    df = pd.concat([df2016, df2017, df2018, df2019])

    dfg = df.groupby(["PLAYER"]).agg(
        {
            "YEAR": "count",
            "AGE": "max",
            "GP": "sum",
            "MIN": ["last", "sum"],
            "FGM": "sum",
            "FGA": "sum",
            "FTM": "sum",
            "FTA": "sum",
            "3PM": "sum",
            "3PA": "sum",
            "PTS": "sum",
            "OREB": "sum",
            "DREB": "sum",
            "AST": "sum",
            "STL": "sum",
            "BLK": "sum",
            "TOV": "sum",
        }
    )

    dfg.columns = ["_".join(x) if x[0] == "MIN" else x[0] for x in dfg.columns.ravel()]
    dfg.rename(columns={'MIN_sum': 'MIN'}, inplace=True)

    dfg = dfg[
        ((dfg["MIN"] > 1800.0) & (dfg["YEAR"] > 1)) | ((dfg["MIN"] > 1000.0) & (dfg["YEAR"] == 1))
    ]

    dfg["FG%"] = dfg["FGM"] / dfg["FGA"]
    dfg["FT%"] = dfg["FTM"] / dfg["FTA"]
    dfg["3PM_per_MIN"] = dfg["3PM"] / dfg["MIN"]
    dfg["PTS_per_MIN"] = dfg["PTS"] / dfg["MIN"]
    dfg["REB_per_MIN"] = (dfg["OREB"] + dfg["DREB"]) / dfg["MIN"]
    dfg["AST_per_MIN"] = dfg["AST"] / dfg["MIN"]
    dfg["STL_per_MIN"] = dfg["STL"] / dfg["MIN"]
    dfg["BLK_per_MIN"] = dfg["BLK"] / dfg["MIN"]
    dfg["MIN_per_TOV"] = dfg["MIN"] / dfg["TOV"]


    dfg["FG%_Z"] = dfg["FG%"].apply(
        lambda x: (x - dfg["FG%"].mean()) / dfg["FG%"].std()
    )
    dfg["FT%_Z"] = dfg["FT%"].apply(
        lambda x: (x - dfg["FT%"].mean()) / dfg["FT%"].std()
    )
    dfg["3PM_Z"] = dfg["3PM_per_MIN"].apply(
        lambda x: (x - dfg["3PM_per_MIN"].mean()) / dfg["3PM_per_MIN"].std()
    )
    dfg["PTS_Z"] = dfg["PTS_per_MIN"].apply(
        lambda x: (x - dfg["PTS_per_MIN"].mean()) / dfg["PTS_per_MIN"].std()
    )
    dfg["REB_Z"] = dfg["REB_per_MIN"].apply(
        lambda x: (x - dfg["REB_per_MIN"].mean()) / dfg["REB_per_MIN"].std()
    )
    dfg["AST_Z"] = dfg["AST_per_MIN"].apply(
        lambda x: (x - dfg["AST_per_MIN"].mean()) / dfg["AST_per_MIN"].std()
    )
    dfg["STL_Z"] = dfg["STL_per_MIN"].apply(
        lambda x: (x - dfg["STL_per_MIN"].mean()) / dfg["STL_per_MIN"].std()
    )
    dfg["BLK_Z"] = dfg["BLK_per_MIN"].apply(
        lambda x: (x - dfg["BLK_per_MIN"].mean()) / dfg["BLK_per_MIN"].std()
    )
    dfg["MPT_Z"] = dfg["MIN_per_TOV"].apply(
        lambda x: (x - dfg["MIN_per_TOV"].mean()) / dfg["MIN_per_TOV"].std()
    )

    dfg["Z"] = (
        dfg["FG%_Z"]
        + dfg["FT%_Z"]
        + dfg["3PM_Z"]
        + dfg["PTS_Z"]
        + dfg["REB_Z"]
        + dfg["AST_Z"]
        + dfg["STL_Z"]
        + dfg["BLK_Z"]
        + dfg["MPT_Z"]
    )

    dfg["Z_ADJ"] = np.where(
        dfg["AGE"] <= 24,
        dfg["Z"] * 1.04,
        np.where(dfg["AGE"] >= 28, dfg["Z"] * 0.96, dfg["Z"]),
    )

    player_lookup = {}
    for manager, player_list in iter(PLAYERS.items()):
        for player_id in player_list:
            player_lookup[player_id] = manager

    dfg.reset_index(inplace=True)

    dfg['Manager'] = dfg['PLAYER'].apply(
        lambda x: player_lookup.get(x)
    )

    dfg = dfg.sort_values(by=["Z_ADJ"], ascending=False)

    dfg.to_csv(r"C:\Users\Michael.Copeland\Projects\nba\NBA 2016-19 (3YR).csv")

    dfm = dfg.groupby(['Manager']).agg({
        'FG%_Z': 'mean',
        'FT%_Z': 'mean',
        '3PM_Z': 'mean',
        'PTS_Z': 'mean',
        'REB_Z': 'mean',
        'AST_Z': 'mean',
        'STL_Z': 'mean',
        'BLK_Z': 'mean',
        'MPT_Z': 'mean',
        'Z': 'mean',
        'Z_ADJ': 'mean'
    })

    dfm = dfm.sort_values(by=["Z_ADJ"], ascending=False)

    dfm.to_csv(r"C:\Users\Michael.Copeland\Projects\nba\Manager Analysis.csv")




if __name__ == "__main__":
    main()

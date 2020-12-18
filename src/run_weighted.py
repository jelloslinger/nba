import datetime
import math
import os

import numpy as np
import pandas as pd

from src.config import HOME_PATH, PLAYERS


def fib(n):
    if n < 2:
        return n
    return fib(n-2) + fib(n-1)


def main():
    q = 0.5
    df2016 = pd.read_csv(os.path.sep.join([HOME_PATH, "NBA Stats 2016.csv"]))
    df2016 = df2016[df2016["MIN"] > df2016["MIN"].quantile(q)]
    df2017 = pd.read_csv(os.path.sep.join([HOME_PATH, "NBA Stats 2017.csv"]))
    df2017 = df2017[df2017["MIN"] > df2017["MIN"].quantile(q)]
    df2018 = pd.read_csv(os.path.sep.join([HOME_PATH, "NBA Stats 2018.csv"]))
    df2018 = df2018[df2018["MIN"] > df2018["MIN"].quantile(q)]
    df2019 = pd.read_csv(os.path.sep.join([HOME_PATH, "NBA Stats 2019.csv"]))
    df2019 = df2019[df2019["MIN"] > df2019["MIN"].quantile(q)]

    df = pd.concat([df2016, df2017, df2018, df2019])
    df["YEAR_RANK"] = df.groupby(["PLAYER"])["YEAR"].rank(method='dense').astype(int)

    df_year_count = df.groupby(["PLAYER"]).agg({"YEAR": "count"}).astype(int).rename(columns={"YEAR": "YEAR_COUNT"})

    df = df.join(df_year_count, on="PLAYER")

    df["WGT"] = df.apply(
        lambda row: fib(row["YEAR_RANK"] + 1) / sum(map(fib, range(2, row["YEAR_COUNT"] + 2))),
        axis=1
    )

    df["FG%"] = df.apply(
        lambda row: 
            df["FGM"].mean() / df["FGA"].mean()
            if row["FGM"] / row["FGA"] > df["FGM"].mean() / df["FGA"].mean() and row["MIN"] <= df["MIN"].mean()
            else row["FGM"] / row["FGA"],
        axis=1
    )
    df["FT%"] = df.apply(
        lambda row: 
            df["FTM"].mean() / df["FTA"].mean()
            if row["FTM"] / row["FTA"] > df["FTM"].mean() / df["FTA"].mean() and row["MIN"] <= df["MIN"].mean()
            else row["FTM"] / row["FTA"],
        axis=1
    )
    df["3P%"] = df.apply(
        lambda row: 
            0 if row["3PA"] == 0 else
                df["3PM"].mean() / df["3PA"].mean()
                if row["3PM"] / row["3PA"] > df["3PM"].mean() / df["3PA"].mean() and row["MIN"] <= df["MIN"].mean()
                else row["3PM"] / row["3PA"],
        axis=1
    )
 
    df["3PM_per_MIN"] = df["3PM"] / df["MIN"]
    df["PTS_per_MIN"] = df["PTS"] / df["MIN"]
    df["OREB_per_MIN"] = df["OREB"] / df["MIN"]
    df["DREB_per_MIN"] = df["DREB"] / df["MIN"]
    df["AST_per_MIN"] = df["AST"] / df["MIN"]
    df["STL_per_MIN"] = df["STL"] / df["MIN"]
    df["BLK_per_MIN"] = df["BLK"] / df["MIN"]

    df["MIN_per_TOV"] = df.apply(
        lambda row: 
            df["MIN"].mean() / df["TOV"].mean()
            if row["MIN"] / row["TOV"] > df["MIN"].mean() / df["TOV"].mean() and row["MIN"] <= df["MIN"].mean()
            else row["MIN"] / row["TOV"],
        axis=1
    )


    df["MIN_per_TOV"] = df.apply(
        lambda row: df["MIN"].mean() / df["TOV"].mean() if row["MIN"] <= df["MIN"].mean() else row["MIN"] / row["TOV"],
        axis=1
    )

    df["FG%_Z"] = df.apply(
        lambda row: (row["FG%"] - df[df["YEAR"] == row["YEAR"]]["FG%"].mean()) / df[df["YEAR"] == row["YEAR"]]["FG%"].std(),
        axis=1
    )
    df["FT%_Z"] = df.apply(
        lambda row: (row["FT%"] - df[df["YEAR"] == row["YEAR"]]["FT%"].mean()) / df[df["YEAR"] == row["YEAR"]]["FT%"].std(),
        axis=1
    )
    df["3P%_Z"] = df.apply(
        lambda row: (row["3P%"] - df[df["YEAR"] == row["YEAR"]]["3P%"].mean()) / df[df["YEAR"] == row["YEAR"]]["3P%"].std(),
        axis=1
    )
    df["3PM_Z"] = df.apply(
        lambda row: (row["3PM_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["3PM_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["3PM_per_MIN"].std(),
        axis=1
    )
    df["PTS_Z"] = df.apply(
        lambda row: (row["PTS_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["PTS_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["PTS_per_MIN"].std(),
        axis=1
    )
    df["OREB_Z"] = df.apply(
        lambda row: (row["OREB_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["OREB_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["OREB_per_MIN"].std(),
        axis=1
    )
    df["DREB_Z"] = df.apply(
        lambda row: (row["DREB_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["DREB_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["DREB_per_MIN"].std(),
        axis=1
    )
    df["AST_Z"] = df.apply(
        lambda row: (row["AST_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["AST_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["AST_per_MIN"].std(),
        axis=1
    )
    df["STL_Z"] = df.apply(
        lambda row: (row["STL_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["STL_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["STL_per_MIN"].std(),
        axis=1
    )
    df["BLK_Z"] = df.apply(
        lambda row: (row["BLK_per_MIN"] - df[df["YEAR"] == row["YEAR"]]["BLK_per_MIN"].mean()) / df[df["YEAR"] == row["YEAR"]]["BLK_per_MIN"].std(),
        axis=1
    )
    df["MPT_Z"] = df.apply(
        lambda row: (row["MIN_per_TOV"] - df[df["YEAR"] == row["YEAR"]]["MIN_per_TOV"].mean()) / df[df["YEAR"] == row["YEAR"]]["MIN_per_TOV"].std(),
        axis=1
    )

    df["WGT_Z"] = df["WGT"] * (
        df["FG%_Z"]
        + df["FT%_Z"]
        + df["3P%_Z"]
        + df["3PM_Z"]
        + df["PTS_Z"]
        + df["OREB_Z"]
        + df["DREB_Z"]
        + df["AST_Z"]
        + df["STL_Z"]
        + df["BLK_Z"]
        + df["MPT_Z"]
    )

    dfg = df.groupby(["PLAYER"]).agg(
        {
            "YEAR": "count",
            "AGE": "max",
            "GP": "sum",
            "MIN": ["last", "sum"],
            "FG%_Z": "sum",
            "FT%_Z": "sum",
            "3P%_Z": "sum",
            "3PM_Z": "sum",
            "PTS_Z": "sum",
            "OREB_Z": "sum",
            "DREB_Z": "sum",
            "AST_Z": "sum",
            "BLK_Z": "sum",
            "STL_Z": "sum",
            "MPT_Z": "sum",
            "WGT_Z": "sum",
        }
    )

    dfg.columns = ["_".join(x) if x[0] == "MIN" else x[0] for x in dfg.columns.ravel()]
    dfg.rename(columns={'MIN_sum': 'MIN'}, inplace=True)

    dfg["MIN_per_GP"] = dfg["MIN"] / dfg["GP"]
    dfg["max_MIN_per_GP"] = dfg["MIN_per_GP"].max()


    dfg["WGT_MIN"] = 2 * dfg["MIN_per_GP"] / (dfg["max_MIN_per_GP"] + dfg["MIN_per_GP"])

    dfg["SCORE"] = dfg["WGT_MIN"] * dfg["WGT_Z"]

    # dfg["WGT2_Z"] = dfg.apply(
    #     lambda row: row["WGT_Z"] * row["MIN_per_GP"] / max_min_per_gp
    # )

    player_lookup = {}
    for manager, player_list in iter(PLAYERS.items()):
        for player_id in player_list:
            player_lookup[player_id] = manager

    dfg.reset_index(inplace=True)

    dfg['Manager'] = dfg['PLAYER'].apply(
        lambda x: player_lookup.get(x)
    )

    dfg = dfg.sort_values(by=["SCORE"], ascending=False)

    dfg.to_csv(os.path.sep.join([HOME_PATH, "NBA 2016-19 (4YR weighted).csv"]))

    dfm = dfg.groupby(['Manager']).agg({
        'FG%_Z': 'mean',
        'FT%_Z': 'mean',
        '3P%_Z': 'mean',
        '3PM_Z': 'mean',
        'PTS_Z': 'mean',
        'OREB_Z': 'mean',
        'DREB_Z': 'mean',
        'AST_Z': 'mean',
        'STL_Z': 'mean',
        'BLK_Z': 'mean',
        'MPT_Z': 'mean',
        'WGT_Z': 'mean',
    })

    dfm = dfm.sort_values(by=["WGT_Z"], ascending=False)

    dfm.to_csv(os.path.sep.join([HOME_PATH, "Manager Analysis (4YR weighted).csv"]))


if __name__ == "__main__":
    main()

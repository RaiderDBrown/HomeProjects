# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 00:49:04 2023

@author: BrownPlanning
"""
import os
import pandas as pd
import numpy as np
import random
from statistics import mean
from time import perf_counter as now
from draft_simulation import get_extremes

def pandas_search(player_pool, position_types):
    def search(player_pool, position_type):
        col = 'Rank' if position_type == 'FLEX' else 'PosRank'
        ranks = player_pool[col].fillna(np.inf).values
        player = player_pool.iloc[np.argmin(ranks)]['Player']
        return player

    drafted = set()
    drafted_players = []
    for position_type in position_types:
        eligible = player_pool[player_pool['POS'] == position_type]
        available = set(eligible['Player']) - drafted
        undrafted = eligible[eligible['Player'].isin(available)]

        player = search(undrafted, position_type)
        drafted.add(player)
        drafted_players.append(player)
    return drafted_players

def dict_search(player_pool, position_types):
    def search(player_pool, position_type):
        col = 'Rank' if position_type == 'FLEX' else 'PosRank'
        player = min(player_pool, key=lambda x: x[col])['Player']
        return player

    drafted = set()
    drafted_players = []
    for position_type in position_types:
        eligible = set(row['Player'] for row in player_pool if row['POS'] == position_type)
        available = eligible - drafted
        undrafted = [row for row in player_pool if row['Player'] in available]

        player = search(undrafted, position_type)
        drafted.add(player)
        drafted_players.append(player)
    return drafted_players

def get_draft_df(player_list, pick_order, players_df, num):
    draft_df = players_df[players_df['Player'].isin(player_list)].reset_index(drop=True)
    draft_df['DraftNum'] = num
    draft_df['DraftPos'] = random.randint(1, 5)
    draft_df['PickOrder'] = str(tuple(pick_order))
    draft_df['PickNum'] = range(1, len(draft_df) + 1)
    return draft_df

def pandas_score(draft_df):
    draft_df['ADP_Score'] = draft_df['PickNum'] - draft_df['ADP']
    draft_df['Rank_Score'] = draft_df['PosRank_Max'] - draft_df['PosRank']

    group = ['DraftNum', 'PickOrder', 'DraftPos']
    score_df = draft_df.groupby(group).agg({
        'ADP_Score': 'sum'
        , 'FPTS': 'sum'
        , 'Rank_Score': 'sum'
    }).reset_index()
    return score_df

def dict_score(draft_df):
    draft_df['ADP_Score'] = draft_df['PickNum'] - draft_df['ADP']
    draft_df['Rank_Score'] = draft_df['PosRank_Max'] - draft_df['PosRank']

    group = ['DraftNum', 'PickOrder', 'DraftPos']
    metrics = ['ADP_Score', 'FPTS', 'Rank_Score']
    data = draft_df.to_dict()

    score_data = {}
    for i in range(len(draft_df)):
        key = tuple(data[col][i] for col in group)
        if key not in score_data:
            score_data[key] = {metric: 0 for metric in metrics}
        for metric in metrics:
            score_data[key][metric] += data[metric][i]

    result = []
    for key, values in score_data.items():
        values.update(dict(zip(group, key)))  # add key back in to result
        result.append(values)
    return pd.DataFrame(result)

pdir = os.path.dirname(os.path.abspath(__file__))
players = pd.read_csv(f'{pdir}\\PlayerStats.csv')
players = players.merge(get_extremes(players), on='POS')

pos_types = ['WR', 'RB', 'TE', 'QB', 'DST']

keys = ['Player', 'Rank', 'POS', 'PosRank']
players_dict = [{k:x[k] for k in keys} for i, x in players.iterrows()]

def is_nan(x):
    return str(x).lower() == 'nan'
for row in players_dict:
    row['Rank'] = float('inf') if is_nan(row['Rank']) else row['Rank']
    row['PosRank'] = float('inf') if is_nan(row['PosRank']) else row['PosRank']

times = {'dict': [], 'pandas': []}
dict_df = pd.DataFrame()
pandas_df = pd.DataFrame()
for draft_num in range(100):

    dict_start = now()
    draft_list = [random.choice(pos_types) for _ in range(40)]
    draft_class = dict_search(players_dict, draft_list)
    dict_df = dict_df.append(
        get_draft_df(draft_class, draft_list, players, draft_num)
        , ignore_index=True)
    score_df_dict = dict_score(dict_df)
    times['dict'].append(now() - dict_start)

    panda_start = now()
    draft_list = [random.choice(pos_types) for _ in range(4)]
    draft_class = pandas_search(players, draft_list)
    pandas_df = pandas_df.append(
        get_draft_df(draft_class, draft_list, players, draft_num)
        , ignore_index=True)
    score_df_pandas = pandas_score(pandas_df)
    times['pandas'].append(now() - panda_start)

print(f'Mean: pandas {mean(times["pandas"])} dictionary {mean(times["dict"])}')
print(f'Total: pandas {sum(times["pandas"])} dictionary {sum(times["dict"])}')

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:28:56 2023

@author: BrownPlanning
"""
import os
import pandas as pd
from statistics import mean
from time import perf_counter as now
from draft_simulation import DraftStrategyManager, DraftSimulator, DraftStats
from draft_simulation import get_extremes

def old_score(draft_df):
    draft_df['ADP_Score'] = draft_df['PickNum'] - draft_df['ADP']
    draft_df['Rank_Score'] = draft_df['PosRank_Max'] - draft_df['PosRank']

    group = ['DraftNum', 'PickOrder', 'DraftPos']
    score_df = draft_df.groupby(group).agg({
        'ADP_Score': 'sum'
        , 'FPTS': 'sum'
        , 'Rank_Score': 'sum'
    }).reset_index()
    return score_df

def score(draft_df):
    scores = {}

    for _, row in draft_df.iterrows():
        key = (row['DraftNum'], row['PickOrder'], row['DraftPos'])

        if key not in scores:
            scores[key] = {
                'ADP_Score': 0,
                'FPTS': 0,
                'Rank_Score': 0
            }

        scores[key]['ADP_Score'] += row['PickNum'] - row['ADP']
        scores[key]['FPTS'] += row['FPTS']
        scores[key]['Rank_Score'] += row['PosRank_Max'] - row['PosRank']

    score_list = [
        {'DraftNum': key[0], 'PickOrder': key[1], 'DraftPos': key[2], **value}
        for key, value in scores.items()
    ]

    score_df = pd.DataFrame(score_list)

    return score_df

pd.options.display.max_columns = None
n_teams, n_rounds = 5, 8

pdir = os.path.dirname(os.path.abspath(__file__))
players = pd.read_csv(f'{pdir}\\PlayerStats.csv')
players = players.merge(get_extremes(players), on='POS')

rules = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
tracker = DraftStrategyManager(rules)

history = pd.DataFrame()
sim = DraftSimulator(n_teams, n_rounds, players)
max_case = sim.generate_draft(players, tracker, record=False)
stats = DraftStats(max_case)

times = {'new': [], 'old': []}
for draft_num in range(10000):
    draft_roster = sim.generate_draft(players, tracker)
    start = now()
    draft_summ = old_score(draft_roster)
    times['old'].append(now() - start)

    start = now()
    df = score(draft_roster)
    times['new'].append(now() - start)

print(f'Mean: current {mean(times["old"])} new {mean(times["new"])}')
print(f'Total: current {sum(times["old"])} new {sum(times["new"])}')

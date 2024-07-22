# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:06:17 2023

@author: BrownPlanning
"""
import pandas as pd
from statistics import mean
from time import perf_counter as now

def build_draft_order_list(fmgrs, num_rounds):
    num_fmgrs = len(fmgrs)
    draft_order = []
    for round_num in range(1, num_rounds + 1):
        # Determine the order of teams for this round
        if round_num % 2 == 1:
            round_order = list(range(1, num_fmgrs + 1))
        else:
            round_order = list(range(num_fmgrs, 0, - 1))
        draft_order.extend([fmgrs[i - 1] for i in round_order])
    return draft_order

def get_draft_positions_list(draft_order):
    draft_pos = []
    seen_items = set()
    for item in draft_order:
        if item not in seen_items:
            draft_pos.append(item)
            seen_items.add(item)
    return draft_pos

def get_strategies_list(plans, draft_order):
    draft_pos = get_draft_positions_list(draft_order)
    num_fmgrs = len(plans)
    features = ['PickNum', 'Round', 'FantasyManager', 'DraftPos', 'POS'
                , 'PickOrder']
    data = {key: [] for key in features}
    for idx, fmgr in enumerate(draft_order):
        rnd = idx // num_fmgrs
        dpos = draft_pos.index(fmgr) + 1
        values = [idx + 1, rnd + 1, fmgr, dpos, plans[fmgr][rnd], plans[fmgr]]
        for feature, value in zip(features, values):
            data[feature].append(value)
    return pd.DataFrame(data)

def build_draft_order(fmgrs, num_rounds):
    num_fmgrs = len(fmgrs)
    pick_order = []
    for round_num in range(1, num_rounds + 1):
        # Determine the order of teams for this round
        if round_num % 2 == 1:
            round_order = list(range(1, num_fmgrs + 1))
        else:
            round_order = list(range(num_fmgrs, 0, - 1))
        pick_order.extend([fmgrs[i - 1] for i in round_order])

    num_range = range(len(pick_order))
    draft_order = pd.DataFrame({
        'FantasyManager': pick_order
        , 'PickNum': [i + 1 for i in num_range]
        , 'Round': [(i // num_fmgrs) + 1 for i in num_range]
        , 'DraftPos': [pick_order.index(fmgr) + 1 for fmgr in pick_order]
        })
    return draft_order

def get_strategies(plans):
    strategies = pd.DataFrame(plans)
    strategies['Round'] = [i + 1 for i in range(len(strategies))]
    strategies = strategies.melt(
        id_vars='Round', var_name='FantasyManager', value_name='POS'
        )
    fmgrs = strategies['FantasyManager']
    strategies['PickOrder'] = [plans[fmgr] for fmgr in fmgrs]
    return strategies

fmgrs = ('SilverWolves', 'SilverVultures', 'BlueHitSquad'
         , 'CapitalTornados', 'EmeraldPanthers'
         )

strategies = {
    'SilverWolves': ('RB', 'WR', 'TE', 'RB', 'FLEX', 'WR', 'WR', 'QB'),
    'SilverVultures': ('TE', 'FLEX', 'WR', 'RB', 'WR', 'RB', 'WR', 'QB'),
    'BlueHitSquad': ('RB', 'TE', 'FLEX', 'WR', 'WR', 'WR', 'QB', 'RB'),
    'CapitalTornados': ('RB', 'RB', 'WR', 'TE', 'WR', 'FLEX', 'WR', 'QB'),
    'EmeraldPanthers': ('WR', 'QB', 'RB', 'RB', 'TE', 'FLEX', 'WR', 'WR')
    }

times = {'new': [], 'old': []}
for draft_num in range(100):
    # list approach
    start = now()
    picks = build_draft_order_list(fmgrs, 8)
    df = get_strategies_list(strategies, picks)
    times['new'].append(now()-start)

    # current approach
    start = now()
    draft_df = build_draft_order(fmgrs, 8)
    strat_df = get_strategies(strategies)
    key = ['Round', 'FantasyManager']
    draft_df = draft_df.merge(strat_df, on=key)
    times['old'].append(now()-start)

print(f'Mean: current {mean(times["old"])} new {mean(times["new"])}')
print(f'Total: current {sum(times["old"])} new {sum(times["new"])}')

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:58:58 2023

@author: BrownPlanning
"""
import cProfile
import pstats
import os
import pandas as pd
from draft_simulation import DraftStrategyManager, DraftSimulator, DraftStats
from draft_simulation import get_extremes, print_now

# Profile your code
cprofiler = cProfile.Profile()
cprofiler.enable()

n_teams, n_rounds = 5, 8

pdir = os.path.dirname(os.path.abspath(__file__))
players = pd.read_csv(f'{pdir}\\PlayerStats.csv')
players = players.merge(get_extremes(players), on='POS')

rules = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
tracker = DraftStrategyManager(reps=100, rules_branch=rules, flex_last=True)

history = pd.DataFrame()
sim = DraftSimulator(n_teams, n_rounds, players)
dstats = DraftStats()

while tracker.processed < 1:
    draft_roster = sim.generate_draft(players, tracker)
    draft_summ = dstats.score(draft_roster, tracker.processed)
    history = pd.concat([history, draft_summ], ignore_index=True)
    print_now(tracker, sim, dstats)

features = ['MaxVal', 'DraftPos']
analysis_df = dstats.add_analytics(history)

cprofiler.disable()
# Create a pstats.Stats object from the cProfile results
stats = pstats.Stats(cprofiler).strip_dirs()
# Sort the statistics by cumulative time
stats.sort_stats('cumulative').print_stats(50, 'draft_simulation.py')

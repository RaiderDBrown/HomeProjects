"""
Fantasy Football Draft Simulator and Analysis

This module contains classes and functions to simulate fantasy football
drafts and analyze draft statistics. It includes the following key components:

- DraftStrategyManager: Manages draft strategy generation and tracking.
- DraftSimulator: Simulates the draft process for fantasy football.
- DraftStats: Analyzes draft statistics and scores drafted players.
- Other utility functions for data generation and analysis.

To run this module, define the desired number of teams, draft rounds,
and player data in the "__main__" section at the bottom of the file.
"""
import copy
import itertools
import random
import os

import numpy as np
import pandas as pd
from itertools import product as iprod
from time import perf_counter as now

class DraftStrategyManager:
    """Tracks the draft strategies for fantasy football."""
    def __init__(self, rules_branch, num_rounds, flex_last=True):
        """
        Parameters:
            pos_types (list): List of position types for drafting.
            num_rounds (int): Number of draft rounds.
            flex_last (bool, optional): Whether to place 'FLEX' positions
                at the end of the draft order. Defaults to False.
            spec_freq (tuple, optional): Specific frequency of position types.
                Defaults to None.
        """
        self.positions = tuple(pos for pos in rules_branch)
        self.branches = self.generate_branches(rules_branch, num_rounds)
        self.strategies = self.generate_strategies(flex_last)
        self.iteration = 0
        self.processed = 0
        self.count = 0
        self.last_print = 0

    def generate_branches(self, rules_branch, num_rounds):
        """
        Generate draft branches based on position types and rounds.

        Parameters:
            pos_types (list): List of position types for drafting.
            num_rounds (int): Number of draft rounds.
            specific_frequency (tuple): Specific frequency of position types.

        Returns:
            list: List of draft branches.
        """
        mins = tuple(rules_branch[position] for position in rules_branch)
        gen = iprod(range(1, num_rounds + 1), repeat=len(rules_branch))

        def is_valid(perm, mins, num_rounds):
            cond1 = all(perm[i] >= mins[i] for i in range(len(perm)))
            cond2 = sum(perm) == num_rounds
            return cond1 and cond2

        freqs = (perm for perm in gen if is_valid(perm, mins, num_rounds))

        def get_branch(key):
            title = dict(zip(self.positions, key))
            return {'branch': title, 'count': 0, 'size': 0, 'iteration': 0}

        return {key: get_branch(key) for key in freqs}

    def generate_strategies(self, flex_last):
        """
        Generate draft strategies based on branch data.

        Parameters:
            flex_last (bool): Whether to place 'FLEX' positions at the end
            of the draft order.

        Returns:
            dict: A dictionary of draft strategies.
        """
        def create_strategies(key, flex_last):
            branch = self.branches[key]['branch']
            base = [pos for pos, cnt in branch.items() for _ in range(cnt)]
            flex_cnt = branch.get('FLEX', 0)

            def move_flex(strategy):
                if flex_last:
                    strategy = [pos for pos in strategy if 'FLEX' not in pos]
                    strategy += ['FLEX'] * flex_cnt
                return tuple(strategy)

            strategies = itertools.permutations(base, len(base))
            return (move_flex(strategy) for strategy in strategies)

        return {strategy: {'key': key, 'count': 0} for key in self.branches
                for strategy in create_strategies(key, flex_last)
                }

    def select_strategy(self, record=True):
        """generated_strategies is global to act as a cache for the function"""
        def get_available():
            store = self.strategies
            min_val = min(store[strategy]['count'] for strategy in store)
            def not_selected(strategy):
                return store[strategy]['count'] == min_val
            return [strategy for strategy in store if not_selected(strategy)]

        strategy = random.choice(get_available())
        strategy_data = copy.deepcopy(self.strategies[strategy])
        if record:
            strategy_data['count'] += 1
            self.strategies[strategy] = strategy_data

            self.calc_progress()
        return strategy

    def calc_progress(self):
        """Calculate the number of times that all strategies in all branches
        have been attempted."""

        strat_data = self.strategies.values()
        for key, data in self.branches.items():
            data['size'] = sum(1 for vals in strat_data if vals['key'] == key)
            data['count'] = sum(vals['count'] for vals in strat_data
                                if vals['key'] == key
                                )
            data['iteration'] = data['count'] // data['size']
            self.branches[key] = data

        branch_data = self.branches.values()
        self.iteration = min(data['iteration'] for data in branch_data)
        self.count = sum(data['count'] for data in branch_data)

        total = sum(data['size'] for data in branch_data)*(self.iteration + 1)
        self.processed = self.count/total

    def print_status(self):
        """
        Print the current status of the draft strategy manager.

        Returns:
            str: Status message.
        """
        self.last_print = self.iteration
        return f'{self.processed * 100:4.0f}% of rep {self.iteration:4}'

class DraftSimulator:
    """Simulates the draft process for fantasy football."""
    def __init__(self, num_teams, num_rounds):
        """
        Parameters:
            num_teams (int): Number of competing teams.
            num_rounds (int): Number of draft rounds.
        """
        self.rounds = num_rounds
        self.num_teams = num_teams
        self.competitors = self.generate_competitors(num_teams)
        self.draft_order = self.generate_draft_order()
        self.players_selected = set()
        self.draft_num = 0

    def draft_player(self, players_df, position_type):
        """
        Simulate drafting a player based on the given position type.

        Parameters:
            players_df (pd.DataFrame): DataFrame containing player statistics.
            position_type (str): The position type to draft.

        Returns:
            str: The name of the drafted player.
        """
        filter_selected = ~players_df['Player'].isin(self.players_selected)
        player_pool = players_df[filter_selected]
        if position_type.startswith('FLEX'):
            pos_selection = ['WR', 'RB', 'TE']
            col = 'Rank'
        elif isinstance(position_type, str):
            pos_selection = [position_type]
            col = 'PosRank'

        available = player_pool[player_pool['POS'].isin(pos_selection)]
        lowest_rank_idx = available[col].fillna(float('inf')).idxmin()
        player = available.loc[lowest_rank_idx]['Player']
        self.players_selected.add(player)
        return player

    def generate_draft_order(self):
        """
        Generate the order of teams for each round in the draft based
        on the snake draft concept.

        Returns:
            list: List of team indices for each round.
        """
        num_teams = len(self.competitors)
        teams = list(self.competitors)

        draft_order = []
        for round_num in range(1, self.rounds + 1):
            # Determine the order of teams for this round
            if round_num % 2 == 1:
                round_order = list(range(1, num_teams + 1))
            else:
                round_order = list(range(num_teams, 0, - 1))
            draft_order.extend([teams[i - 1] for i in round_order])
        return draft_order

    def generate_competitors(self, n_competitors):
        """
        Generate competitor team names.

        Parameters:
            n_competitors (int): Number of competitor teams.

        Returns:
            list: List of competitor team names.
        """
        adj = [
            'Red', 'Blue', 'Rabid', 'Yellow', 'Orange', 'Purple', 'Silver'
            , 'Golden', 'Sapphire', 'Emerald', 'Jade', 'Black', 'Gray'
            , 'Bronze', 'Capital', 'Killer', 'Crazy', 'Thunder'
            ]
        nouns = [
            'Dragons', 'Tigers', 'Lions', 'Eagles', 'Wolves', 'Bears'
            , 'Sharks', 'Hawks', 'Panthers', 'Cobras', 'Vultures'
            , 'Grizzlies', 'Techies', 'HitSquad', 'Dogs', 'Hunters'
            , 'Crazies', 'Tornados', 'Volcanoes', 'Cats'
            ]
        competitors = set()
        while len(competitors) < n_competitors:
            competitor = random.choice(adj) + random.choice(nouns)
            competitors.add(competitor)
        return list(competitors)

    def generate_draft(self, players_df, dsm):
        """
        Generate a draft using the provided player data and draft strategy
        manager.

        Parameters:
            players_df (pd.DataFrame): DataFrame containing player statistics.
            dsm (DraftStrategyManager): The draft strategy manager.

        Returns:
            pd.DataFrame: DataFrame representing the drafted players.
        """
        tactics = [dsm.select_strategy() for _ in range(self.num_teams)]
        strategies = dict(zip(self.competitors, tactics))
        self.players_selected = set()
        self.draft_num += 1

        draft_data = []  # Collect draft data in a list of dictionaries
        for j, competitor in enumerate(self.draft_order):
            draft_round = j // len(self.competitors)
            pos_selection = strategies[competitor][draft_round]
            player = self.draft_player(players_df, pos_selection)
            row = {
                'Round': draft_round + 1, 'FantasyManager': competitor
                , 'PickOrder': str(strategies[competitor])
                , 'DraftPos': self.competitors.index(competitor) + 1
                , 'Player': player, 'PickNum': j + 1
            }
            draft_data.append(row)
        draft_df = pd.DataFrame(draft_data)
        draft_df['DraftNum'] = self.draft_num
        draft_df = draft_df.merge(players_df, on='Player', how='left')
        return draft_df

    def print_status(self):
        """
        Print the current status of the draft simulator.

        Returns:
            str: Status message.
        """
        return f'{self.draft_num:7,} drafts'

class DraftStats:
    """Analyzes draft statistics for fantasy football."""
    def __init__(self, strategy, players_df):
        """
        Parameters:
            strategy_store (dict): A dictionary of draft strategies.
            players_df (pd.DataFrame): DataFrame containing player statistics.
        """
        self.position_maxs = self.get_position_maxs(players_df)
        self.metrics = ['ADP_Score', 'FPTS_Score', 'Rank_Score']
        self.mins = [0, 0, 0]
        self.maxs = self.set_edge(strategy, players_df)
        self.best = 0
        self.prev_best = 0
        self.best_row = None

    def get_position_maxs(self, players_df, get_max=True):
        """
        Get extreme values by position.

        Parameters:
            players_df (pd.DataFrame): DataFrame containing player statistics.
            get_max (bool, optional): Whether to get maximum values.

        Returns:
            pd.DataFrame: DataFrame containing extreme values by position.
        """
        func = 'max' if get_max else 'min'
        sfx = 'Max' if get_max else 'Min'
        cols = ['PosRank', 'ADP', 'FPTS']
        params = {'index': 'POS', 'values': cols, 'aggfunc': func}
        pivot_df = players_df.pivot_table(**params).reset_index()
        pivot_df.columns = ['POS', f'PosRank_{sfx}', f'ADP_{sfx}', f'FPTS_{sfx}']
        return pivot_df

    def score_players(self, draft_data):
        """
        Score players based on draft data.

        Parameters:
            draft_data (pd.DataFrame): DataFrame containing draft data.

        Returns:
            pd.DataFrame: DataFrame with player scores.
        """
        maxs = self.position_maxs
        score_df = draft_data.merge(maxs, on='POS', suffixes=('', ''))
        score_df['ADP_Score'] = score_df['PickNum'] - score_df['ADP']
        score_df['FPTS_Score'] = score_df['FPTS']
        score_df['Rank_Score'] = score_df['PosRank_Max'] - score_df['PosRank']
        return score_df

    def set_edge(self, strategy, players_df, maxs=True):
        """
        Set edge values for draft statistics.

        Parameters:
            strategy_store (dict): A dictionary of draft strategies.
            players_df (pd.DataFrame): DataFrame containing player statistics.
            maxs (bool, optional): Whether to calculate maximum values.

        Returns:
            float: Edge value for draft statistics.
        """
        def pick(players_df, position_type):
            if position_type.startswith('FLEX'):
                pos_selection = ['WR', 'RB', 'TE']
                col = 'Rank'
            elif isinstance(position_type, str):
                pos_selection = [position_type]
                col = 'PosRank'

            available = players_df[players_df['POS'].isin(pos_selection)]
            func = 'idxmin' if maxs else 'idxmax'
            rank_idx = available[col].fillna(float('inf')).agg(func)
            player = available.loc[rank_idx]['Player']
            return player

        def set_draft(strategy, players_df):
            draft_data = []  # Collect draft data in a list of dictionaries
            for draft_round, position_type in enumerate(strategy):
                row = {  'PickOrder': strategy
                       , 'PickNum': draft_round + 1
                       , 'Player': pick(players_df, position_type)
                       }
                draft_data.append(row)

            draft_df = pd.DataFrame(draft_data)
            draft_df = draft_df.merge(players_df, on='Player', how='left')
            draft_df = self.score_players(draft_df)
            return draft_df

        edge = set_draft(strategy, players_df)
        edge = edge.groupby('PickOrder')[self.metrics].sum().reset_index()
        return edge[self.metrics].max()

    def euclidean(self, metrics):
        """
        Calculate the Euclidean distance for metrics.

        Parameters:
            metrics (list): List of metric values.

        Returns:
            numpy.ndarray: Array of Euclidean distances.
        """
        # w = [0.25, 0.55, 0.2]
        # x = np.multiply(w, metrics)
        # mins, maxs = np.multiply(w, self.mins), np.multiply(w, self.maxs)

        # diff_to_min = mins - x
        diff_to_min = self.mins - metrics
        dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=1))

        # diff_to_max = maxs - x
        diff_to_max = self.maxs - metrics
        dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=1))
        return dist_to_min / (dist_to_max + dist_to_min)

    def score_draft(self, draft_data):
        """
        Score a draft based on draft data.

        Parameters:
            draft_data (pd.DataFrame): DataFrame containing draft data.

        Returns:
            pd.DataFrame: DataFrame with draft scores.
        """
        score_df = self.score_players(draft_data)
        group = ['DraftNum', 'PickOrder', 'DraftPos']
        df_grouped = score_df.groupby(group)[self.metrics].sum().reset_index()
        self.mins = np.minimum(self.mins, df_grouped[self.metrics].min())
        self.maxs = np.maximum(self.maxs, df_grouped[self.metrics].max())

        df_grouped['DraftScore'] = self.euclidean(df_grouped[self.metrics])
        best = df_grouped['DraftScore'].max()
        if best > self.best:
            self.best = best
            row_filter = df_grouped['DraftScore'] == self.best
            self.best_row = df_grouped[row_filter].iloc[0]
        return df_grouped

    def add_analytics(self, draft_data):
        """
        Add analytics to draft data.

        Parameters:
            draft_data (pd.DataFrame): DataFrame containing draft data.

        Returns:
            pd.DataFrame: DataFrame with added analytics.
        """
        group = ['PickOrder', 'DraftPos']
        pick_scores = draft_data.drop(columns='DraftNum')
        pick_scores = pick_scores.groupby(group).mean().reset_index()

        funcs = ['min', 'max', 'std', 'count']
        agg_scores = draft_data.groupby(group)['DraftScore']
        agg_scores = agg_scores.agg(funcs).reset_index()
        titles = {'min': 'MinVal', 'max': 'MaxVal', 'std': 'StdDev'}
        agg_scores.rename(columns=titles, inplace=True)

        score_df = pick_scores.merge(agg_scores, on=group)
        return score_df

    def print_status(self):
        """
        Print the current status of draft statistics.

        Returns:
            str: Status message.
        """
        self.prev_best = self.best
        items = self.best_row.to_dict().items()
        best_metrics = {k: v for k, v in items if k in self.metrics}
        msg = ' '.join([f'{k} {v:4,.0f}' for k, v in best_metrics.items()])
        return f'Best {self.best:0.5f} {msg}'

if __name__ == "__main__":
    start = now()
    n_teams, n_rounds = 5, 8

    pdir = os.path.dirname(os.path.abspath(__file__))
    players = pd.read_csv(f'{pdir}\\PlayerStats.csv')

    rules = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
    tracker = DraftStrategyManager(rules, n_rounds, flex_last=False)

    history = pd.DataFrame()
    stats = DraftStats(tracker.select_strategy(record=False), players)
    sim = DraftSimulator(n_teams, n_rounds)

    while stats.best < 0.99 and tracker.iteration < 100:
        draft_roster = sim.generate_draft(players, tracker)
        history = pd.concat([history, draft_roster], ignore_index=True)
        summ_df = stats.score_draft(history)

        improved = stats.best - stats.prev_best > 0.001
        time_to_rpt = tracker.iteration - tracker.last_print >= 20
        show_status = time_to_rpt and (tracker.processed > 0.995)

        if improved or show_status:
            txt = [f'{x.print_status()}' for x in [sim, tracker, stats]]
            print(' '.join(txt))

    features = ['MaxVal', 'DraftPos']
    analysis_df = stats.add_analytics(summ_df)
    analysis_df.sort_values(by=features, inplace=True, ascending=[False, True])
    analysis_df.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
    print(stats.best_row.transpose(), '\n')

    best_vals = analysis_df.groupby('DraftPos')['MaxVal'].max().reset_index()
    best_picks = analysis_df.merge(best_vals, on=features)
    best_picks.sort_values(by='DraftPos', inplace=True)
    best_picks.to_csv(f'{pdir}\\BestPicks.csv', index=False)
    end = now()
    elapsed = (end - start)/60
    print(f'Total Time for {tracker.iteration} iterations {elapsed:.2f} minutes')

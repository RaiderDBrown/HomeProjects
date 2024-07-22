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
from time import perf_counter as now

class DraftStrategyManager:
    """Tracks the draft strategies for fantasy football."""
    def __init__(self, pos_types, num_rounds, flex_last=False, spec_freq=None):
        """
        Parameters:
            pos_types (list): List of position types for drafting.
            num_rounds (int): Number of draft rounds.
            flex_last (bool, optional): Whether to place 'FLEX' positions
                at the end of the draft order. Defaults to False.
            spec_freq (tuple, optional): Specific frequency of position types.
                Defaults to None.
        """
        self.branches = self.generate_branches(pos_types, num_rounds, spec_freq)
        self.strategy_store = self.generate_strategies(flex_last)
        self.branch_store = self.set_branch_store(self.strategy_store)
        self.iteration = 0
        self.processed = 0
        self.count = 0
        self.last_print = 0

    def generate_strategies(self, flex_last):
        """
        Generate draft strategies based on branch data.

        Parameters:
            flex_last (bool): Whether to place 'FLEX' positions at the end
            of the draft order.

        Returns:
            dict: A dictionary of draft strategies.
        """
        def create_strategies(branch):
            base = [pick for pick, frq in branch.items() for _ in range(frq)]
            strategies = itertools.permutations(base, len(base))
            return {strategy: {'count': 0} for branch_perm in strategies
                    for strategy in self.add_flex(branch_perm, flex_last)}

        the_strategies = {}
        for branch in self.branches:
            the_strategies[branch] = create_strategies(dict(branch))

        return the_strategies

    def generate_branches(self, pos_types, num_rounds, specific_frequency):
        """
        Generate draft branches based on position types and rounds.

        Parameters:
            pos_types (list): List of position types for drafting.
            num_rounds (int): Number of draft rounds.
            specific_frequency (tuple): Specific frequency of position types.

        Returns:
            list: List of draft branches.
        """
        draft_rounds = range(1, num_rounds + 1)
        freqs = itertools.product(draft_rounds, repeat=len(pos_types))
        def get_branch(frq):
            return frozenset(dict(zip(pos_types, frq)).items())
        if specific_frequency is not None:
            return [get_branch(specific_frequency)]
        return [get_branch(frq) for frq in freqs if sum(frq) == num_rounds]

    def add_flex(self, base_strategy, at_end):
        """
        Add 'FLEX' positions to the base strategy.

        Parameters:
            base_strategy (tuple): Base strategy without 'FLEX' positions.
            at_end (bool): Whether to place 'FLEX' positions at the end of
                the draft order.

        Returns:
            list: List of strategies with 'FLEX' positions added.
        """
        flex_pos = ['WR', 'RB', 'TE']
        strategies = []
        indices = [i for i, pos in enumerate(base_strategy) if pos == 'FLEX']

        def flex_at_end(indices, base_strategy):
            first_flex_index = next(iter(indices), -1)
            if first_flex_index > -1:
                # Check that all subsequent elements are also 'FLEX'
                for i in range(first_flex_index, len(base_strategy)):
                    if 'FLEX' not in base_strategy[i]:
                        return False
            else:
                return True  # if no flex elements
            return True  # if only one Flex element

        for flex in flex_pos:
            strategy = list(base_strategy)
            for perm in itertools.product([flex], repeat=len(indices)):
                for i, index in enumerate(indices):
                    strategy[index] = f'FLEX_{perm[i]}'
                if at_end == flex_at_end(indices, base_strategy):
                    strategies.append(tuple(strategy))
        return strategies

    def calc_iteration(self):
        """Calculate the number of times that all strategies in all branches
        have been attempted."""
        repetitions = []
        progress, total, num = 0, 0, 0
        for branch, data_dict in self.branch_store.items():
            data_dict['repetition'] = data_dict['count']//data_dict['size']

            adj_size = data_dict['repetition']*data_dict['size']
            progress += (data_dict['count'] - adj_size)

            total += data_dict['size']
            num += data_dict['count']
            repetitions.append(data_dict['repetition'])

            self.branch_store[branch] = data_dict

        self.iteration = min(repetitions)
        self.processed = progress/total
        self.count = num

    def set_branch_store(self, strategy_store):
        """
        Set the initial branch store based on the provided strategy store.

        Parameters:
            strategy_store (dict): A dictionary of draft strategies.

        Returns:
            dict: A dictionary representing the branch store.
        """
        branch_store = {}
        for branch in self.branches:
            size = len(strategy_store[branch])
            branch_store[branch] = {'count': 0, 'size': size, 'repetition': 0}
        return branch_store

    def get_available(self, store_type, branch=None):
        """
        Get the available items in a specified store type.

        Parameters:
            store_type (str): The type of store to query ('branch' or
                'strategy').
            branch (frozenset, optional): The branch for which to
                retrieve available strategies. Defaults to None.

        Returns:
            list: List of available items in the specified store.
        """
        if store_type == 'branch':
            store = self.branch_store
        elif store_type == 'strategy':
            store = self.strategy_store[branch]

        min_val = min(store[x]['count'] for x in store)
        return [x for x in store if store[x]['count'] == min_val]

    def select_strategy(self):
        """generated_strategies is global to act as a cache for the function"""
        branch = random.choice(self.get_available('branch'))
        branch_data = copy.deepcopy(self.branch_store[branch])
        branch_data['count'] += 1
        self.branch_store[branch] = branch_data

        strategy = random.choice(self.get_available('strategy', branch))
        strategy_data = copy.deepcopy(self.strategy_store[branch][strategy])
        strategy_data['count'] += 1
        self.strategy_store[branch][strategy] = strategy_data

        self.calc_iteration()
        return strategy

    def print_status(self):
        """
        Print the current status of the draft strategy manager.

        Returns:
            str: Status message.
        """
        self.last_print = self.iteration
        return f'{self.processed*100:4.0f}% of rep {self.iteration:4}'

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
    def __init__(self, strategy_store, players_df):
        """
        Parameters:
            strategy_store (dict): A dictionary of draft strategies.
            players_df (pd.DataFrame): DataFrame containing player statistics.
        """
        self.pos_maxs = self.get_extremes_by_position(players_df)
        self.metrics = ['ADP_Score', 'FPTS_Score', 'Rank_Score']
        self.mins = [0, 0, 0]
        self.maxs = self.set_edge(strategy_store, players_df)
        self.best = 0
        self.prev_best = 0
        self.best_row = None

    def get_extremes_by_position(self, players_df, get_max=True):
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
        score_df = draft_data.merge(self.pos_maxs, on='POS', suffixes=('', ''))
        score_df['ADP_Score'] = score_df['PickNum'] - score_df['ADP']
        score_df['FPTS_Score'] = score_df['FPTS']
        score_df['Rank_Score'] = score_df['PosRank_Max'] - score_df['PosRank']
        return score_df

    def set_edge(self, strategy_store, players_df, maxs=True):
        """
        Set edge values for draft statistics.

        Parameters:
            strategy_store (dict): A dictionary of draft strategies.
            players_df (pd.DataFrame): DataFrame containing player statistics.
            maxs (bool, optional): Whether to calculate maximum values.

        Returns:
            float: Edge value for draft statistics.
        """
        def pick(players_df, pos_type):
            func = 'idxmin' if maxs else 'idxmax'
            if pos_type.startswith('FLEX'):
                pos_type = random.choice(['WR', 'TE', 'RB'])
            available = players_df[players_df['POS'] == pos_type]
            row = available.loc[getattr(available['PosRank'], func)()]
            return row['Player']

        def set_draft(strategy_store, players_df):
            draft_data = []  # Collect draft data in a list of dictionaries
            for branch_tuple in strategy_store:
                branch = dict(branch_tuple)
                draft_order = [k for k, v in branch.items() for _ in range(v)]
                for draft_round, pos_type in enumerate(draft_order):
                    row = {'Branch': str(branch)
                           , 'PickOrder': str(draft_order)
                           , 'PickNum': draft_round + 1
                           , 'Player': pick(players_df, pos_type)
                           }
                    draft_data.append(row)

            draft_df = pd.DataFrame(draft_data)
            draft_df = draft_df.merge(players_df, on='Player', how='left')
            draft_df = self.score_players(draft_df)
            return draft_df

        best_draft = set_draft(strategy_store, players_df)
        group = ['Branch', 'PickOrder']
        best_draft = best_draft.groupby(group)[self.metrics].sum()
        best_draft = best_draft.reset_index()
        return best_draft[self.metrics].sum()

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
        self.best = score_df['MaxVal'].max()
        self.best_row = score_df[score_df['MaxVal'] == self.best].iloc[0]
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
    positions = ['QB', 'RB', 'WR', 'TE', 'FLEX']

    pdir = os.path.dirname(os.path.abspath(__file__))
    players = pd.read_csv(f'{pdir}\\PlayerStats.csv')

    # rules_branch {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
    rules_branch = dict(zip(positions, tuple([1, 2, 3, 1, 1])))
    kwargs = {'flex_last': True, 'spec_freq': tuple([1, 2, 3, 1, 1])}
    tracker = DraftStrategyManager(positions, n_rounds, **kwargs)

    history = pd.DataFrame()
    stats = DraftStats(tracker.strategy_store, players)
    sim = DraftSimulator(n_teams, n_rounds)

    while stats.best < 0.99 and tracker.iteration < 20:
        draft_roster = sim.generate_draft(players, tracker)
        history = history.append(draft_roster, ignore_index=True)
        summ_df = stats.score_draft(history)
        analysis_df = stats.add_analytics(summ_df)

        if stats.best - stats.prev_best > 0.0001:
            txt = [f'{x.print_status()}' for x in [sim, tracker, stats]]
            print(' '.join(txt))
        elif tracker.iteration - tracker.last_print == 1:
            if tracker.processed > 0.995:
                txt = [f'{x.print_status()}' for x in [sim, tracker, stats]]
                print(' '.join(txt))

    # features = ['MaxVal', 'DraftPos']
    # analysis_df.sort_values(by=features, inplace=True, ascending=[False, True])
    # analysis_df.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
    # print(stats.best_row.transpose(), '\n')

    # best_vals = analysis_df.groupby('DraftPos')['MaxVal'].max().reset_index()
    # best_picks = analysis_df.merge(best_vals, on=features)
    # best_picks.sort_values(by='DraftPos', inplace=True)
    # best_picks.to_csv(f'{pdir}\\BestPicks.csv', index=False)
    end = now()
    elapsed = (end - start)/60
    print(f'Total Time for {tracker.iteration} iterations {elapsed:.2f} minutes')

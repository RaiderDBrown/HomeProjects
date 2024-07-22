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
import itertools
import random
import os

import numpy as np
import pandas as pd
from time import perf_counter as now

class RulesBranchCreator:
    def __init__(self, rules_branch):
        """
        Initializes a RulesBranchCreator object.

        Parameters:
            rules_branch (dict): Rules for drafting based on position types.
        """
        self.positions = [position for position in rules_branch]
        self.branches = self.generate_rules_branches(rules_branch)

    def generate_rules_branches(self, rules_branch):
        """
        Generate draft branches based on predefined rules for different position types.

        Parameters:
            rules_branch (dict): Rules for drafting based on position types.

        Returns:
            dict: Dictionary containing draft branches with associated statistics.
        """
        key = tuple(rules_branch.values())
        branch_stats = {metric: 0 for metric in ['count', 'size', 'iteration']}
        return {key: {'branch': rules_branch, **branch_stats}}

class PermutationsBranchCreator:
    positions = ['RB', 'QB', 'TE', 'DST', 'WR', 'K', 'FLEX']

    def __init__(self, num_rounds, positions=None):
        """
        Initializes a PermutationsBranchCreator object.

        Parameters:
            num_rounds (int): Number of draft rounds.
            positions (list, optional): List of position types for drafting. Defaults to None.
        """
        self.branches = self.generate_nrounds_branches(positions, num_rounds)

    def generate_nrounds_branches(self, positions, num_rounds):
        """
        Generate draft branches based on position types and rounds using permutations.

        Parameters:
            positions (list): List of position types for drafting.
            num_rounds (int): Number of draft rounds.

        Returns:
            dict: Dictionary containing draft branches with associated statistics.
        """
        if positions is not None:
            self.positions = positions

        draft_rounds = range(1, num_rounds + 1)
        num_positions = len(self.positions)
        allfrqs = itertools.product(draft_rounds, repeat=num_positions)

        def is_valid(perm, num_rounds):
            cond1 = all(perm[i] > 0 for i in range(len(perm)))
            cond2 = sum(perm) == num_rounds
            return cond1 and cond2

        freqs = (perm for perm in allfrqs if is_valid(perm, num_rounds))

        def get_branch(key):
            branch = dict(zip(self.positions, key))
            return {'branch': branch, 'count': 0, 'size': 0, 'iteration': 0}

        return {key: get_branch(key) for key in freqs}

class DraftStrategyManager(RulesBranchCreator, PermutationsBranchCreator):
    """Tracks the draft strategies for fantasy football."""
    def __init__(self, rules_branch=None, positions=None, num_rounds=None):
        """
        Initializes a DraftStrategyManager object.

        Parameters:
            rules_branch (dict, optional): Rules for drafting based on position types. Defaults to None.
            positions (list, optional): List of position types for drafting. Defaults to None.
            num_rounds (int, optional): Number of draft rounds. Defaults to None.
        """
        if rules_branch is not None:
            RulesBranchCreator.__init__(self, rules_branch)
        if num_rounds is not None:
            PermutationsBranchCreator.__init__(self, positions, num_rounds)
        self.strategies = self.generate_strategies()
        self.strategy_iterator = self.reset_available(self.strategies)

        self.iteration = 0
        self.processed = 0
        self.last_print = 0

    def generate_strategies(self):
        """
        Generate draft strategies based on branch data.
        Returns:
            dict: A dictionary of draft strategies.
        """
        def create_strategies(key):
            branch = self.branches[key]['branch']
            base = [pos for pos, cnt in branch.items() for _ in range(cnt)]
            strategies = itertools.permutations(base, len(base))
            return (strategy for strategy in strategies)

        strategies = {}
        for key in self.branches:
            for strategy in create_strategies(key):
                strategies[strategy] = {'key': key, 'count': 0}
        return strategies

    def reset_available(self, strategies_dict):
        """Set up the random generator of strategy keys"""
        available = list(strategy for strategy in strategies_dict)
        random.shuffle(available)
        return iter(available)

    def select_strategy(self, record=True):
        """Randomly selecte a strategy from the strategy dictionary keys."""
        try:
            strategy = next(self.strategy_iterator)
        except StopIteration:
            # If the end of the shuffled list is reached, start over
            self.strategy_iterator = self.reset_available(self.strategies)
            strategy = next(self.strategy_iterator)
        if record:
            self.strategies[strategy]['count'] += 1
            self.calc_progress()
        return strategy

    def calc_branch_stats(self):
        # Create a dictionary to store branch-related data temporarily
        branch_data = {}
        # Initialize branch_data with zeros
        for key in self.branches:
            branch_data[key] = {'size': 0, 'count': 0}

        # Calculate branch sizes and counts
        for strategy, strategy_data in self.strategies.items():
            key = strategy_data['key']
            branch_data[key]['size'] += 1
            branch_data[key]['count'] += strategy_data['count']

        # Update the branches with the calculated data
        for key, branch in self.branches.items():
            branch_info = branch_data[key]
            branch['size'] = branch_info['size']
            branch['count'] = branch_info['count']
            iteration = branch_info['count'] // branch_info['size']
            branch['iteration'] = iteration

    def calc_total_size(self, iteration):
        size = sum(self.branches[key]['size'] for key in self.branches)
        return size * (iteration + 1)

    def calc_progress(self):
        """Calculate the number of times that all strategies in all branches
        have been attempted."""
        self.calc_branch_stats()

        store = self.branches
        self.iteration = min(store[key]['iteration'] for key in store)
        count = sum(store[key]['count'] for key in store)
        self.processed = count/self.calc_total_size(self.iteration)

    def print_status(self):
        """
        Print the current status of the draft strategy manager.

        Returns:
            str: Status message.
        """
        self.last_print = self.iteration
        status = f'{self.processed * 100:4.0f}% of rep {self.iteration:4}'
        return status

class DraftSimulator:
    """Simulates the draft process for fantasy football."""
    def __init__(self, num_teams, num_rounds, players_df):
        """
        Parameters:
            num_teams (int): Number of competing teams.
            num_rounds (int): Number of draft rounds.
        """
        self.rounds = num_rounds
        self.num_teams = num_teams
        self.draft_num = 0
        self.fmgrs = tuple()
        self.drafted = set()
        self.eligible = self.create_position_subsets(players_df)

    def generate_fantasy_mgrs(self, n_competitors):
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
        return tuple(competitors)

    def create_position_subsets(self, players_df):
        """ Create a dictionary that maps position types to pre-filtered
        player subsets"""
        def get_eligible(position):
            def is_nan(item):
                if str(item).lower() == 'nan':
                    return float('inf')
                return item
            flex_pos = ['WR', 'RB', 'TE']
            pos_types = flex_pos if position == 'FLEX' else [position]
            cols = ['Rank', 'PosRank']
            eligible = {row['Player']: {col: is_nan(row[col]) for col in cols}
                            for i, row in players_df.iterrows()
                            if row['POS'] in pos_types}
            return eligible

        positions = ['WR', 'RB', 'TE', 'FLEX', 'QB', 'DST', 'K']
        subsets = {position: get_eligible(position) for position in positions}
        return subsets

    def search_player(self, player_pool, position_type, record):
        """Search criteria for player by position."""
        col = 'Rank' if position_type == 'FLEX' else 'PosRank'
        draftee = min(player_pool, key=lambda player: player_pool[player][col])
        if record:
            self.drafted.add(draftee)
        return draftee

    def draft_player(self, position_types, record=True):
        """
        Simulate drafting players based on the given position types.

        Parameters:
            players_df (pd.DataFrame): DataFrame containing player statistics.
            position_types (list or str): List of position types to draft.
            record (bool): Whether to record drafted players.

        Returns:
            list: The names of the drafted players.
        """
        drafted_players = []
        for position_type in position_types:
            eligible = self.eligible[position_type]
            available = set(eligible) - self.drafted
            undrafted = {player: eligible[player] for player in eligible
                         if player in available}

            player = self.search_player(undrafted, position_type, record)
            drafted_players.append(player)

        return drafted_players

    def build_draft_order(self, record):
        """
        Generate the order of teams for each round in the draft based
        on the snake draft concept.

        Returns:
            list: List of team indices for each round.
        """
        self.fmgrs = self.generate_fantasy_mgrs(self.num_teams)
        if record:
            self.draft_num += 1

        pick_order = []
        for round_num in range(1, self.rounds + 1):
            # Determine the order of teams for this round
            if round_num % 2 == 1:
                round_order = list(range(1, self.num_teams + 1))
            else:
                round_order = list(range(self.num_teams, 0, - 1))
            pick_order.extend([self.fmgrs[i - 1] for i in round_order])

        num_range = range(len(pick_order))
        draft_order = pd.DataFrame({
            'FantasyManager': pick_order
            , 'PickNum': [i + 1 for i in num_range]
            , 'Round': [(i // self.num_teams) + 1 for i in num_range]
            , 'DraftPos': [pick_order.index(fmgr) + 1 for fmgr in pick_order]
            , 'DraftNum': [self.draft_num for _ in pick_order]
            })
        return draft_order

    def get_strategies(self, dsm, record):
        plans = {fmgr: dsm.select_strategy(record) for fmgr in self.fmgrs}
        strategies = pd.DataFrame(plans)
        strategies['Round'] = [i + 1 for i in range(len(strategies))]
        strategies = strategies.melt(
            id_vars='Round', var_name='FantasyManager', value_name='POS'
            )
        fmgrs = strategies['FantasyManager']
        strategies['PickOrder'] = [plans[fmgr] for fmgr in fmgrs]
        return strategies

    def generate_draft(self, players_df, dsm, record=True):
        """
        Generate a draft using the provided player data and draft strategy
        manager.

        Parameters:
            players_df (pd.DataFrame): DataFrame containing player statistics.
            dsm (DraftStrategyManager): The draft strategy manager.

        Returns:
            pd.DataFrame: DataFrame representing the drafted players.
        """
        draft_df = self.build_draft_order(record)
        strategies = self.get_strategies(dsm, record)
        key = ['Round', 'FantasyManager']
        draft_df = draft_df.merge(strategies, on=key)

        self.drafted = set()
        draft_df['Player'] = self.draft_player(draft_df['POS'], record)
        draft_df = draft_df.merge(players_df, on='Player')
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
    metrics = ['ADP_Score', 'FPTS', 'Rank_Score']
    response = 'DraftScore'
    def __init__(self, edge_case):
        """
        Parameters:
            strategy_store (dict): A dictionary of draft strategies.
            players_df (pd.DataFrame): DataFrame containing player statistics.
        """
        self.initial = True
        self.mins = [0, 0, 0]
        self.maxs = self.set_extreme(edge_case)
        self.best = 0
        self.prev_best = -1
        self.best_row = None

    def update_range(self, score_df):
        metrics = score_df[self.metrics]
        self.mins = np.minimum(self.mins, metrics.min())
        self.maxs = np.maximum(self.maxs, metrics.max())

        best = score_df[self.response].max()
        if best > self.best:
            self.best = best
            row_filter = score_df[self.response] == self.best
            self.best_row = score_df[row_filter].iloc[0]

    def euclidean(self, metrics):
        """
        Calculate the Euclidean distance for metrics.

        Parameters:
            metrics (list): List of metric values.

        Returns:
            numpy.ndarray: Array of Euclidean distances.
        """
        diff_to_min = self.mins - metrics
        dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=1))

        diff_to_max = self.maxs - metrics
        dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=1))

        result = dist_to_min / (dist_to_max + dist_to_min)
        return result

    def score(self, draft_df):
        """
        Score a draft based on draft data.

        Parameters:
            draft_data (pd.DataFrame): DataFrame containing draft data.

        Returns:
            pd.DataFrame: DataFrame with draft scores.
        """
        draft_df['ADP_Score'] = draft_df['PickNum'] - draft_df['ADP']
        draft_df['Rank_Score'] = draft_df['PosRank_Max'] - draft_df['PosRank']

        group = ['DraftNum', 'PickOrder', 'DraftPos']
        score_df = draft_df.groupby(group).agg({
            'ADP_Score': 'sum'
            , 'FPTS': 'sum'
            , 'Rank_Score': 'sum'
        }).reset_index()

        score_df[self.response] = self.euclidean(score_df[self.metrics])
        if not self.initial:
            self.update_range(score_df)
        return score_df

    def set_extreme(self, edge_case, get_max=True):
        self.maxs = [0, 0, 0]
        score_df = self.score(edge_case)
        func = 'max' if get_max else 'min'
        best = score_df[self.response].agg(func)
        best_row = score_df[score_df[self.response] == best].iloc[0]
        self.initial = False
        return best_row[self.metrics]

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
        if self.best > self.prev_best:
            self.prev_best = self.best
        items = self.best_row.to_dict().items()
        best_metrics = {k: v for k, v in items if k in self.metrics}
        msg = ' '.join([f'{k} {v:4,.0f}' for k, v in best_metrics.items()])
        return f'Best {self.best:0.5f} {msg}'

def get_extremes(players_df, get_max=True):
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
    cols = ['PosRank', 'ADP', 'FPTS', 'Rank']

    pos_metrics = players_df.groupby('POS')[cols].agg(func).reset_index()
    pos_metrics.columns = ['POS'] + [f'{col}_{sfx}' for col in cols]
    return pos_metrics

if __name__ == "__main__":
    start = now()
    n_teams, n_rounds = 5, 8

    pdir = os.path.dirname(os.path.abspath(__file__))
    players = pd.read_csv(f'{pdir}\\PlayerStats.csv')
    players = players.merge(get_extremes(players), on='POS')

    rules = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
    tracker = DraftStrategyManager(rules, n_rounds)

    history = pd.DataFrame()
    sim = DraftSimulator(n_teams, n_rounds, players)
    max_case = sim.generate_draft(players, tracker, record=False)
    dstats = DraftStats(max_case)

    while dstats.best < 0.99 and tracker.iteration < 750:
        draft_roster = sim.generate_draft(players, tracker)
        draft_summ = dstats.score(draft_roster)
        history = pd.concat([history, draft_summ], ignore_index=True)

        improved = dstats.best - dstats.prev_best > 0.001
        time_to_rpt = tracker.iteration - tracker.last_print >= 50
        show_status = time_to_rpt and (tracker.processed > 0.995)

        if improved or show_status:
            txt = [f'{x.print_status()}' for x in [sim, tracker, dstats]]
            print(' '.join(txt))

    features = ['MaxVal', 'DraftPos']
    analysis_df = dstats.add_analytics(history)
    analysis_df.sort_values(by=features, inplace=True, ascending=[False, True])
    analysis_df.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
    print(dstats.best_row.transpose(), '\n')

    best_vals = analysis_df.groupby('DraftPos')['MaxVal'].max().reset_index()
    best_picks = analysis_df.merge(best_vals, on=features)
    best_picks.sort_values(by='DraftPos', inplace=True)
    best_picks.to_csv(f'{pdir}\\BestPicks.csv', index=False)
    end = now()
    elapsed = (end - start)/60
    print(f'Total Time for {tracker.iteration} iterations {elapsed:.2f} minutes')

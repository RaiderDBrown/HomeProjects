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
from time import perf_counter as now
import numpy as np
import pandas as pd

class BranchCreator:
    branch_stats = {metric: 0 for metric in ['count', 'size', 'iteration']}

    def __init__(self, rules_branch=None, num_rounds=None):
        """
        Initializes a BranchCreator object. Branches are the basis for
        multiple strategies used by Fantasy Managers. Each branch has
        permutations, i.e. strategies.

        Parameters:
            rules_branch (dict or list): Rules for drafting
                based on position types.
            num_rounds (int): Number of draft rounds.
        """
        self.positions = self.get_positions(rules_branch)
        self.branches = self.generate_branches(rules_branch, num_rounds)

    def get_positions(self, rules_branch):
        """Establish the positions used in drafts."""
        if isinstance(rules_branch, list):
            return list(
                set(position for rule in rules_branch
                    for position in rule.keys()
                    )
                )
        if rules_branch is not None:
            return list(rules_branch)
        return ['RB', 'QB', 'TE', 'DST', 'WR', 'K', 'FLEX']

    def get_branch(self, key):
        """
        The key is a tuple of the number of players to be selected
        at each position type, e.g. tuple(2, 1, 3) 2 RBs, 1 QB, and 3 TEs.
        """
        branch = dict(zip(self.positions, key))
        return {'branch': branch, **self.branch_stats}

    def generate_rules_branch(self, rules_branch):
        """
        Generate draft branches based on predefined rules for position types.

        Parameters:
            rules_branch (dict): Rules for drafting based on position types.

        Returns:
            dict: Draft branches with associated statistics.
        """
        key = tuple(val for val in rules_branch.values())
        return {key: self.get_branch(key)}

    def is_valid_branch(self, position_freqs, num_rounds):
        """
        The number of players selected by position should be at least
        one and the total number of players should match the number of
        rounds in the draft.
        """
        at_least_one = min(position_freqs) > 0
        match_num_rounds = sum(position_freqs) == num_rounds
        no_more_than = max(position_freqs) < 6
        return all([at_least_one, match_num_rounds, no_more_than])

    def generate_nrounds_branches(self, num_rounds):
        """
        Generate draft branches based on position types and rounds using
        permutations.

        Parameters:
            num_rounds (int): Number of draft rounds.

        Returns:
            dict: Draft branches with associated statistics.
        """
        draft_rounds = range(1, num_rounds + 1)
        frqs = len(self.positions)

        keys = (
            perm for perm in itertools.product(draft_rounds, repeat=frqs)
            if self.is_valid_branch(perm, num_rounds)
            )
        return {key: self.get_branch(key) for key in keys}

    def generate_branches(self, rules_branch, num_rounds):
        if isinstance(rules_branch, list):
            rules = [self.generate_rules_branch(rule) for rule in rules_branch]
            return {key: val for rule in rules for key, val in rule.items()}
        if num_rounds is not None:
            return self.generate_nrounds_branches(num_rounds)
        if rules_branch is not None:
            return self.generate_rules_branch(rules_branch)

class StrategyCreator:
    def __init__(self, branches, flex_last=False):
        """
        Initializes StrategyCreator object.

        Parameters:
            branches (dict): Rules for drafting based on position type.
            flex_last (boolean): Whether the FLEX position has to be last
                in the draft order.
        """
        self.branches = branches
        self.size = 0
        self.strategies = self.generate_strategies(flex_last)

    def move_flex_to_end(self, strategy):
        """Move the FLEX position to the end of the draft."""
        count = strategy.count('FLEX')
        new_strategy = [pos for pos in strategy if 'FLEX' not in pos]
        new_strategy += ['FLEX'] * count
        return tuple(new_strategy)

    def get_branch_strategies(self, branch_key, flex_last):
        """Generate draft strategies for a specific branch."""
        branch = self.branches[branch_key]['branch']
        base = [pos for pos, cnt in branch.items() for _ in range(cnt)]
        strategies = set(itertools.permutations(base, len(base)))
        if flex_last:
            return set(self.move_flex_to_end(strat) for strat in strategies)
        return strategies

    def generate_strategies(self, flex_last):
        """
        Generate draft strategies based on branch data.
        Returns:
            generator: A generator of draft strategies.
        """
        func = self.get_branch_strategies
        strategies = []
        for branch_key in self.branches:
            branch_strategies = func(branch_key, flex_last)
            self.branches[branch_key]['size'] = len(branch_strategies)
            self.size += len(branch_strategies)
            strategies.extend(branch_strategies)
        return strategies

class DraftStrategyManager(BranchCreator, StrategyCreator):
    """Tracks the draft strategies for fantasy football."""
    def __init__(self, reps=1, rules_branch=None, num_rounds=None
                 , flex_last=False):
        """
        Initializes a DraftStrategyManager object.

        Parameters:
            rules_branch (dict, optional): Rules based on position types.
            num_rounds (int, optional): Number of draft rounds.
        """
        BranchCreator.__init__(self, rules_branch, num_rounds)
        StrategyCreator.__init__(self, self.branches, flex_last)
        self.iteration = 0
        self.count = 0
        self.processed = 0
        self.last_print = 0
        self.planned_size = self.size * reps
        self.selected = set()

    def select_strategy(self):
        """Randomly selecte a strategy from the strategy dictionary keys."""
        if self.size == len(self.selected):
            self.selected = set()
        indices = set(range(self.size))
        strategy_index = random.choice(list(indices - self.selected))
        strategy = self.strategies[strategy_index]

        self.selected.add(strategy_index)
        self.calc_progress(strategy)
        return strategy

    def calc_progress(self, strategy):
        """Calculate the number of times that all strategies in all branches
        have been attempted."""
        def find_branch(strategy_selected):
            key = tuple(strategy_selected.count(pos) for pos in self.positions)
            return key

        branch_key = find_branch(strategy)
        self.branches[branch_key]['count'] += 1
        self.count += 1

        bcount = self.branches[branch_key]['count']
        bsize = self.branches[branch_key]['size']
        self.branches[branch_key]['iteration'] = bcount // bsize

        self.processed = self.count / self.planned_size
        self.iteration = self.count // self.size

    def time_to_rpt(self, num_reports=15):
        iterations = self.planned_size // self.size
        threshold = max(iterations // num_reports, 1)
        if self.iteration - self.last_print >= threshold:
            return True
        return False

    def print_status(self):
        """
        Print the current status of the draft strategy manager.

        Returns:
            str: Status message.
        """
        self.last_print = self.iteration
        remainder = self.count % self.size
        count =  remainder if remainder > 0 else self.size
        perc = round(count / self.size  * 100)
        overall = self.processed * 100
        status = f'{perc:3.0f}% of rep {self.iteration:3} '
        status += f'{overall:3.0f}% overall'
        return status

class DraftSimulator:
    """Simulates the draft process for fantasy football."""
    def __init__(self, num_fmgrs, num_rounds, players_df):
        """
        Parameters:
            num_teams (int): Number of competing teams.
            num_rounds (int): Number of draft rounds.
        """
        self.num_rounds = num_rounds
        self.num_fmgrs = num_fmgrs
        self.draft_num = 0
        self.fmgrs = tuple()
        self.drafted = set()
        self.eligible = self.create_position_subsets(players_df)

    def generate_fantasy_mgrs(self, num_fmgrs):
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
        while len(competitors) < num_fmgrs:
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

    def search_player(self, player_pool, position_type):
        """Search criteria for player by position."""
        col = 'Rank' if position_type == 'FLEX' else 'PosRank'
        draftee = min(player_pool, key=lambda player: player_pool[player][col])

        self.drafted.add(draftee)
        return draftee

    def draft_player(self, position_types):
        """
        Simulate drafting players based on the given position types.

        Parameters:
            players_df (pd.DataFrame): DataFrame containing player statistics.
            position_types (list or str): List of position types to draft.

        Returns:
            list: The names of the drafted players.
        """
        drafted_players = []
        for position_type in position_types:
            eligible = self.eligible[position_type]
            available = set(eligible) - self.drafted
            undrafted = {player: eligible[player] for player in eligible
                         if player in available}

            player = self.search_player(undrafted, position_type)
            drafted_players.append(player)

        return drafted_players

    def build_draft_order(self, fmgrs):
        """
        Generate the order of teams for each round in the draft based
        on the snake draft concept.

        Returns:
            list: List of team indices for each round.
        """
        draft_order = []
        for round_num in range(1, self.num_rounds + 1):
            # Determine the order of teams for this round
            if round_num % 2 == 1:
                round_order = list(range(1, self.num_fmgrs + 1))
            else:
                round_order = list(range(self.num_fmgrs, 0, - 1))
            draft_order.extend([fmgrs[i - 1] for i in round_order])
        return draft_order

    def get_draft_positions(self, draft_order):
        """Find the draft position for each Fantasy Manager"""
        draft_positions = []
        seen_items = set()
        for item in draft_order:
            if item not in seen_items:
                draft_positions.append(item)
                seen_items.add(item)
        return draft_positions

    def build_draft_plan(self, fmgr_strategies, draft_order):
        """Build the dataframe of selection types for each Fantasy Manager.
        Dataframe specifies the position that each Fantasy Manager chooses
        for each round of the draft.
        """
        draft_pos = self.get_draft_positions(draft_order)
        df_features = ['PickNum', 'Round', 'FantasyManager', 'DraftPos'
                       , 'POS', 'PickOrder']
        draft_plan = {key: [] for key in df_features}
        for idx, fmgr in enumerate(draft_order):
            rnd = idx // self.num_fmgrs
            dpos = draft_pos.index(fmgr) + 1
            strategy = fmgr_strategies[fmgr]
            values = [idx + 1, rnd + 1, fmgr, dpos, strategy[rnd], strategy]
            for feature, value in zip(df_features, values):
                draft_plan[feature].append(value)
        return pd.DataFrame(draft_plan)

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
        fmgrs = self.generate_fantasy_mgrs(self.num_fmgrs)
        draft_order = self.build_draft_order(fmgrs)
        fmgr_strategies = {fmgr: dsm.select_strategy() for fmgr in fmgrs}
        draft_plan = self.build_draft_plan(fmgr_strategies, draft_order)

        self.draft_num += 1
        draft_plan['DraftNum'] = self.draft_num

        self.drafted = set()
        draft_plan['Player'] = self.draft_player(draft_plan['POS'])
        draft = draft_plan.merge(players_df, on=['Player'])
        draft['POS'] = draft['POS_x']  # POS_x has FLEX positions
        draft.drop(['POS_x', 'POS_y'], axis=1, inplace=True)
        return draft

    def print_status(self):
        """
        Print the current status of the draft simulator.

        Returns:
            str: Status message.
        """
        return f'{self.draft_num:7,} drafts'

class BestStats:
    """Tracker for the best drafts"""
    metrics = ['ADP_Score', 'FPTS', 'Rank_Score']
    response = 'DraftScore'
    def __init__(self):
        """Results of best drafts stored in a dataframe for lookup."""
        self.best_df = None

    def add_row(self, row):
        """Add a new row only if it is unique."""
        if self.best_df is None:
            return row

        all_vals = self.best_df[self.metrics].values.tolist()
        row_vals = row[self.metrics].values.tolist()[0]
        if row_vals not in all_vals:
            return self.best_df.append(row)
        return self.best_df  # else return existing dataframe

    def get_best_row(self, df=None, score_count=-1):
        if df is None:
            df = self.best_df

        best = df[self.response].max()
        best_row = df[df[self.response] == best]
        best_row = best_row.iloc[0].to_frame().T

        if score_count > -1:
            best_row['ScoreIndex'] = score_count
        return best_row

    def has_improved(self, score_count=None):
        if self.best_df is None:
            return False

        best = self.best_df[self.response].max()
        if score_count is None:
            score_count = self.best_df['ScoreIndex'].max()
        row_cond = self.best_df['ScoreIndex'] < score_count
        prev_best = self.best_df[row_cond][self.response].max()
        if best - prev_best > 0.001:
            return True

        return False

    def print_status(self):
        """
        Print the current status of draft statistics.

        Returns:
            str: Status message.
        """
        if self.ready_to_print():
            best_row = self.get_best_row()
            items = best_row.to_dict(orient='records')[0].items()
            best_metrics = {k: v for k, v in items if k in self.metrics}
            msg = ' '.join([f'{k} {v:4,.0f}' for k, v in best_metrics.items()])
            best = best_row[self.response].max()
            return f'Best {best:0.5f} {msg}'

    def ready_to_print(self):
        if self.best_df is None:
            return False
        if len(self.best_df) > 0:
            return True
        return False

class DraftStats(BestStats):
    """Analyzes draft statistics for fantasy football."""
    def __init__(self):
        """
        Parameters:
            strategy_store (dict): A dictionary of draft strategies.
            players_df (pd.DataFrame): DataFrame containing player statistics.
        """
        BestStats.__init__(self)
        self.mins = [0, 0, 0]
        self.maxs = [0, 0, 0]
        self.draft_cache = None
        self.count = 0

    def call_score_method(self, draft_df, progress):
        """Controls how often score method is called. Due to rounding, the
        function will run at a multiple (2-3 times) of the number of steps.
        """
        if self.draft_cache is None:
            self.draft_cache = draft_df
        else:
            rows = [self.draft_cache, draft_df]
            self.draft_cache = pd.concat(rows, ignore_index=True)

        if len(self.draft_cache) > 25000 or progress == 1:
            return True
        return False

    def rank_scores(self, score_df):
        """
        Rank the outcomes for the scoring metrics (self.metrics) of each
        draft in the score dataframe.
        """
        self.mins = np.minimum(self.mins, score_df[self.metrics].min())
        self.maxs = np.maximum(self.maxs, score_df[self.metrics].max())

        score_df = score_df.append(self.best_df)
        score_df[self.response] = self.euclidean(score_df[self.metrics])
        best_row = self.get_best_row(score_df, self.count)
        self.best_df = self.add_row(best_row)
        new_ranks = self.euclidean(self.best_df[self.metrics])
        self.best_df[self.response] = new_ranks
        return score_df

    def agg_players(self, draft_df):
        """Sum all players across each metric."""
        group = ['DraftNum', 'PickOrder', 'DraftPos']
        sums = {metric: 'sum' for metric in self.metrics}
        score_df = draft_df.groupby(group).agg(sums).reset_index()
        return score_df

    def score(self, draft_df, progress=0):
        """
        Score a draft based on draft data.

        Parameters:
            draft_data (pd.DataFrame): DataFrame containing draft data.

        Returns:
            pd.DataFrame: DataFrame with draft scores.
        """
        if self.call_score_method(draft_df, progress):
            cache = self.draft_cache
            cache['ADP_Score'] = cache['PickNum'] - cache['ADP']
            cache['Rank_Score'] = cache['PosRank_Max'] - cache['PosRank']

            score_df = self.agg_players(cache)
            score_df = self.rank_scores(score_df)

            self.count += 1
            self.draft_cache = None
            return score_df

    def add_analytics(self, draft_data):
        """
        Add analytics to draft data.

        Parameters:
            draft_data (pd.DataFrame): DataFrame containing draft data.

        Returns:
            pd.DataFrame: DataFrame with added analytics.
        """
        group = ['PickOrder', 'DraftPos']
        pick_scores = draft_data.drop(columns=['DraftNum', 'ScoreIndex'])
        pick_scores = pick_scores.groupby(group).mean().reset_index()

        funcs = ['min', 'max', 'std', 'count']
        agg_scores = draft_data.groupby(group)['DraftScore']
        agg_scores = agg_scores.agg(funcs).reset_index()
        titles = {'min': 'MinVal', 'max': 'MaxVal', 'std': 'StdDev'}
        agg_scores.rename(columns=titles, inplace=True)

        score_df = pick_scores.merge(agg_scores, on=group)
        return score_df

    def euclidean(self, metrics):
        """
        Calculate the Euclidean distance for metrics.

        Parameters:
            metrics (list): List of metric values.

        Returns:
            numpy.ndarray: Array of Euclidean distances.
        """
        if isinstance(metrics, pd.Series):
            metrics = metrics.to_numpy()
            func_axis = 0  # For Series, use axis=0 for a one-dimensional array
        elif isinstance(metrics, pd.DataFrame):
            func_axis = 1  # For DataFrames, use axis=1 to sum along columns

        diff_to_min = self.mins - metrics
        dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=func_axis))

        diff_to_max = self.maxs - metrics
        dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=func_axis))

        result = dist_to_min / (dist_to_max + dist_to_min)
        return result

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

def print_now(dsm, dsim, dsts):
    if dsts.has_improved(dsts.count) or dsm.time_to_rpt():
        if dsts.count > 0:
            txt = [f'{x.print_status()}' for x in [dsim, dsm, dsts]]
            print(' '.join(txt))

if __name__ == "__main__":
    N = 2000
    rule = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
    n_teams, n_rounds = 5, 8

    start = now()
    pdir = os.path.dirname(os.path.abspath(__file__))
    players = pd.read_csv(f'{pdir}\\PlayerStats.csv')
    players = players.merge(get_extremes(players), on='POS')

    tracker = DraftStrategyManager(reps=N, rules_branch=rule, flex_last=True)
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
    analysis_df.sort_values(by=features, inplace=True, ascending=[False, True])
    analysis_df.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
    print(dstats.get_best_row(), '\n')

    best_vals = analysis_df.groupby('DraftPos')['MaxVal'].max().reset_index()
    best_picks = analysis_df.merge(best_vals, on=features)
    best_picks.sort_values(by='DraftPos', inplace=True)
    best_picks.to_csv(f'{pdir}\\BestPicks.csv', index=False)
    end = now()
    elapsed = (end - start)/60
    print(f'Total Time for {tracker.iteration} iterations {elapsed:.2f} minutes')

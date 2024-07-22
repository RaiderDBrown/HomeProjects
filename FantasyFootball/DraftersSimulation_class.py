import copy
import itertools
import random
import os

import numpy as np
import pandas as pd

class DraftStrategyManager:
    def __init__(self, pos_types, num_rounds, flex_last=False, spec_freq=None):
        self.branches = self.generate_branches(pos_types, num_rounds, spec_freq)
        self.strategy_store = self.generate_strategies(flex_last)
        self.branch_store = self.set_branch_store(self.strategy_store)
        self.iteration = 0
        self.processed = 0
        self.count = 0

    def generate_strategies(self, flex_last):
        def create_strategies(branch):
            base = [pick for pick, frq in branch.items() for _ in range(frq)]
            strategies = itertools.permutations(base, len(base))
            cnt = {'count': 0}
            func = self.add_flex
            return {s: cnt for b in strategies for s in func(b, flex_last)}

        the_strategies = {}
        for b in self.branches:
            the_strategies[b] = create_strategies(dict(b))

        return the_strategies

    def generate_branches(self, pos_types, num_rounds, specific_frequency):
        draft_rounds = range(1, num_rounds + 1)
        freqs = itertools.product(draft_rounds, repeat=len(pos_types))
        def get_branch(frq):
            return frozenset(dict(zip(pos_types, frq)).items())
        if specific_frequency is not None:
            return [get_branch(specific_frequency)]
        return [get_branch(frq) for frq in freqs if sum(frq) == num_rounds]

    def add_flex(self, base_strategy, at_end):
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
        for branch in self.branch_store:
            count = self.branch_store[branch]['count']
            size = self.branch_store[branch]['size']
            self.branch_store[branch]['repetition'] = count//size
            progress += count - (count//size)*size
            total += size
            num += count
            repetitions.append(count//size)

        self.iteration = min(repetitions)
        self.processed = progress/total
        self.count = num

    def set_branch_store(self, strategy_store):
        branch_store = {}
        for b in self.branches:
            size = len(strategy_store[b])
            branch_store[b] = {'count': 0, 'size': size, 'repetition': 0}
        return branch_store

    def get_available(self, store_type, branch=None):
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
        txt0 = f'{self.count:7,} runs {self.processed*100:3.0f}%'
        status = f'{txt0} of iteration {self.iteration:2}'
        return status

class DraftSimulator:
    def __init__(self, num_teams, num_rounds, strategies, draft_num=0):
        self.rounds = num_rounds
        self.competitors = self.generate_competitors(num_teams)
        self.strategies = dict(zip(self.competitors, strategies))
        self.draft_order = self.generate_draft_order()
        self.players_selected = set()
        self.i = draft_num

    def draft_player(self, players_df, position_type):
        filter_selected = ~players_df['Player'].isin(self.players_selected)
        player_pool = players_df[filter_selected]

        if position_type.startswith('FLEX_'):
            position_type = position_type.replace('FLEX_', '')
        available = player_pool[player_pool['POS'] == position_type]
        # Filter rows where 'Rank' is not NaN
        valid_rows = available[~available['Rank'].isna()]
        if not valid_rows.empty:
            # Select the row with the lowest 'Rank'
            lowest_rank_row = valid_rows.loc[valid_rows['PosRank'].idxmin()]
            player = lowest_rank_row['Player']
        elif valid_rows.empty:
            try:
                # If 'Rank' contains only NaN values, select a row at random
                random_row = available.sample(n=1).iloc[0]
                player = random_row['Player']
            except ValueError:
                player = 'Trash Man'

        self.players_selected.add(player)
        return player

    def generate_draft_order(self):
        """Generate team indices for the draft in each round
            based on the snake draft concept."""
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
        return competitors

    def generate_draft(self, players_df):
        draft_data = []  # Collect draft data in a list of dictionaries
        for j, competitor in enumerate(self.draft_order):
            draft_round = j // len(self.competitors) + 1
            pos_selection = self.strategies[competitor][draft_round - 1]
            player = self.draft_player(players_df, pos_selection)
            row = {
                'Round': draft_round, 'FantasyManager': competitor
                , 'PickOrder': str(self.strategies[competitor])
                , 'Player': player, 'PickNum': j + 1
            }
            draft_data.append(row)
        df = pd.DataFrame(draft_data)
        df['DraftIndex'] = self.i
        df = df.merge(players_df, on='Player', how='left')
        return df

class DraftStats:
    def __init__(self, strategy_store, players_df):
        self.pos_maxs = self.get_extremes_by_position(players_df)
        self.metrics = ['ADP_Score', 'FPTS_Score', 'Rank_Score']
        self.mins = self.set_edge(strategy_store, players_df, False)
        self.maxs = self.set_edge(strategy_store, players_df, True)
        self.best = 0
        self.best_row = None

    def get_extremes_by_position(self, players_df, get_max=True):
        func = 'max' if get_max else 'min'
        sfx = 'Max' if get_max else 'Min'
        cols = ['PosRank', 'ADP', 'FPTS']
        df = players_df.pivot_table(index='POS', values=cols, aggfunc=func)
        df = df.reset_index()
        df.columns = ['POS', f'PosRank_{sfx}', f'ADP_{sfx}', f'FPTS_{sfx}']
        return df

    def score_players(self, draft_df):
        df = draft_df.merge(self.pos_maxs, on='POS', suffixes=('', ''))
        df['ADP_Score'] = df['PickNum'] - df['ADP']
        df['FPTS_Score'] = df['FPTS']
        df['Rank_Score'] = df['PosRank_Max'] - df['PosRank']
        return df

    def set_edge(self, strategy_store, players_df, maxs=True):

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
                for r, pos_type in enumerate(draft_order):
                    row = {'Branch': str(branch)
                           , 'PickOrder': str(draft_order), 'PickNum': r + 1
                           , 'Player': pick(players_df, pos_type)
                           }
                    draft_data.append(row)

            df = pd.DataFrame(draft_data)
            df = df.merge(players_df, on='Player', how='left')
            df = self.score_players(df)
            return df

        best_draft = set_draft(strategy_store, players_df)
        group = ['Branch', 'PickOrder']
        best_draft = best_draft.groupby(group)[self.metrics].sum()
        best_draft = best_draft.reset_index()
        return best_draft[self.metrics].sum()

    def euclidean(self, metrics):
        w = [0.25, 0.55, 0.2]
        x = np.multiply(w, metrics)
        mins, maxs = np.multiply(w, self.mins), np.multiply(w, self.maxs)

        diff_to_min = mins - x
        # diff_to_min = self.mins - metrics
        dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=1))

        diff_to_max = maxs - x
        # diff_to_max = self.maxs - metrics
        dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=1))
        return dist_to_min / (dist_to_max + dist_to_min)

    def score_draft(self, draft_df):
        df = self.score_players(draft_df)
        group = ['FantasyManager', 'DraftIndex', 'PickOrder']
        df_grouped = df.groupby(group)[self.metrics].sum().reset_index()
        self.mins = np.minimum(self.mins, df_grouped[self.metrics].min())
        self.maxs = np.maximum(self.maxs, df_grouped[self.metrics].max())

        df_grouped['DraftScore'] = self.euclidean(df_grouped[self.metrics])
        return df_grouped

    def add_analytics(self, draft_df):
        pick_scores = draft_df.drop(columns=['FantasyManager', 'DraftIndex'])
        pick_scores = pick_scores.groupby('PickOrder').mean().reset_index()

        funcs = ['min', 'max', 'std', 'count']
        agg_scores = draft_df.groupby('PickOrder')['DraftScore']
        agg_scores = agg_scores.agg(funcs).reset_index()
        titles = {'min': 'MinVal', 'max': 'MaxVal', 'std': 'StdDev'}
        agg_scores.rename(columns=titles, inplace=True)

        df = pick_scores.merge(agg_scores, on='PickOrder')

        self.best = df['MaxVal'].max()
        self.best_row = df[df['MaxVal'] == self.best].iloc[0]
        return df

    def print_status(self):
        items = self.best_row.to_dict().items()
        best_metrics = {k: v for k, v in items if k in self.metrics}
        txt = ' '.join([f'{k} {v:4,.0f}' for k, v in best_metrics.items()])
        status = f'Best {self.best:0.5f} {txt}'
        return status

if __name__ == "__main__":
    n_teams, n_rounds = 5, 8
    positions = ['QB', 'RB', 'WR', 'TE', 'FLEX']

    pdir = os.path.dirname(os.path.abspath(__file__))
    players = pd.read_csv(f'{pdir}\\PlayerStats.csv')

    trash_team = []
    for position in [p for p in positions if not p.startswith('FL')]:
        trash_man = {'Player': 'Trash Man', 'Team': 'BAD', 'POS': position}
        trash_team.append(trash_man)

    players = players.append(trash_team, ignore_index=True)

    # rules_branch {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1}
    rules_branch = dict(zip(positions, tuple([1, 2, 3, 1, 1])))
    kwargs = {'flex_last': True, 'spec_freq': tuple([1, 2, 3, 1, 1])}
    tracker = DraftStrategyManager(positions, n_rounds, **kwargs)

    player_data = pd.DataFrame()
    stats = DraftStats(tracker.strategy_store, players)
    i = 0

    while stats.best < 0.99 and tracker.iteration < 3:
        best = stats.best
        tactics = [tracker.select_strategy() for _ in range(n_teams)]
        draft_num = tracker.count // n_teams
        simulator = DraftSimulator(n_teams, n_rounds, tactics, draft_num)

        draft = simulator.generate_draft(players)
        player_data = player_data.append(draft, ignore_index=True)
        all_drafts = stats.score_draft(player_data)
        analysis = stats.add_analytics(all_drafts)

        if stats.best - best >= 0.0001:
            i = tracker.iteration
            best = stats.best
            print(f'{tracker.print_status()}  {stats.print_status()}')
        elif tracker.iteration - i == 2 and tracker.processed > 0.99:
            i = tracker.iteration
            print(f'{tracker.print_status()}  {stats.print_status()}')

    features = ['MaxVal', 'DraftScore']
    analysis.sort_values(by=features, inplace=True, ascending=[False, True])
    analysis.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
    print(stats.best_row.transpose(), '\n')

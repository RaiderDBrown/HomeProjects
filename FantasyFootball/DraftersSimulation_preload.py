import random, itertools, math
import pandas as pd
import numpy as np

global generated_strategies
generated_strategies = {}

def generate_team_name():
    # List of adjectives and nouns for team names
    adjectives = ['Red', 'Blue', 'Rabid', 'Yellow', 'Orange'
                  , 'Purple', 'Silver', 'Golden', 'Sapphire'
                  , 'Emerald', 'Jade', 'Black', 'Gray', 'Bronze'
                  , 'Capital', 'Killer', 'Crazy', 'Thunder']
    nouns = ['Dragons', 'Tigers', 'Lions', 'Eagles', 'Wolves', 'Bears'
             , 'Sharks', 'Hawks', 'Panthers', 'Cobras', 'Vultures'
             , 'Grizzlies', 'Techies', 'HitSquad', 'Dogs', 'Hunters'
             , 'Crazies', 'Tornados', 'Volcanoes', 'Cats']

    # Generate a random team name by combining an adjective and a noun
    team_name = random.choice(adjectives) + random.choice(nouns)
    return team_name

def generate_competitors(n_competitors):
    competitors = set()
    while len(competitors) < n_competitors:
        competitor = generate_team_name()
        competitors.add(competitor)
    return competitors

def calculate_num_strategies(draft_branch):
    """Calculate all of the permuations by position group for the branch"""
    total_permutations = math.factorial(sum(draft_branch.values()))
    for count in draft_branch.values():  # by position group
        total_permutations //= math.factorial(count)
    return total_permutations

def count_num_processed(draft_branch):
    global generated_strategies
    draft_branch_key = frozenset(draft_branch.items())
    strategy_dict = generated_strategies[draft_branch_key]
    min_val = min(strategy_dict.values())
    count = sum(1 for v in strategy_dict.values() if v > min_val)
    return count

def calc_iteration():
    """Calculate the number of times that all strategies in all branches
    have been attempted."""
    global generated_strategies
    freqs = []
    for branch in generated_strategies:
        freqs.append(min(generated_strategies[branch].values()))
    return min(freqs)

def add_flex_position(base_strategy, flex_positions=['WR', 'RB', 'TE']):
    strategies = []
    indices = [i for i, pos in enumerate(base_strategy) if pos == 'FLEX']
    
    for flex_position in flex_positions:
        strategy = list(base_strategy)
        for perm in itertools.product(flex_positions, repeat=len(indices)):
            for i, index in enumerate(indices):
                strategy[index] = f'FLX_{perm[i]}'
            strategies.append(tuple(strategy))
    return strategies

def flex_at_end(strategy):
    # Find the index of the first occurrence of 'FLX' (if any)
    flex_indices = (i for i, pos in enumerate(strategy) if 'FLX' in pos)
    first_flex_index = next((flex_indices), -1)
    if first_flex_index > -1:
        # Check that all subsequent elements are also 'FLX'
        for i in range(first_flex_index, len(strategy)):
            if 'FLX' not in strategy[i]:
                return False
    else:
        return True  # if no flex elements
    return True  # if only one Flex element

def initiate_generated_strategies(pos_types, num_rounds):
    """Setup the generated_strategies object"""
    global generated_strategies
    mixes = itertools.product(range(1, num_rounds + 1), repeat=len(pos_types))
    
    for freqs in mixes:
        draft_branch = dict(zip(pos_types, freqs))
        key = frozenset(draft_branch.items())
        picks = [p for p, f in draft_branch.items() for _ in range(f)]
        
        if sum(freqs) == num_rounds:
            generated_strategies[key] = {}
            all_strategies = itertools.permutations(picks, len(picks))
            for base_strategy in all_strategies:
                strategies = add_flex_position(base_strategy)
                strategy_count = {s: 0 for s in strategies if flex_at_end(s)}
                generated_strategies[key].update(strategy_count)

def select_branch():
    global generated_strategies
    n_branches = len(generated_strategies)
    random_index = random.randint(0, n_branches - 1)
    random_branch_key = list(generated_strategies)[random_index]
    return dict(random_branch_key)

def generate_strategy(draft_branch):
    """generated_strategies is global to act as a cache for the function"""
    global generated_strategies
    draft_branch_key = frozenset(draft_branch.items())
    strategy_dict = generated_strategies[draft_branch_key]
    min_val = min(list(strategy_dict.values()))

    strategies = [k for k, v in strategy_dict.items() if v == min_val]
    selection = random.choice(strategies)
    generated_strategies[draft_branch_key][selection] += 1
    return selection

def draft_player(players_df, position_type, prev_selected):
    filter_selected = ~players_df['Player'].isin(prev_selected)
    player_pool = players_df[filter_selected]

    if position_type.startswith('FLX'):
        available = player_pool[player_pool['POS'] == position_type[4:]]
    else:
        available = player_pool[player_pool['POS'] == position_type]
    # Filter rows where 'Rank' is not NaN
    valid_rows = available[~available['Rank'].isna()]
    if not valid_rows.empty:
        # Select the row with the lowest 'Rank'
        lowest_rank_row = valid_rows.loc[valid_rows['PosRank'].idxmin()]
        return lowest_rank_row['Player']

    try:
        # If 'Rank' contains only NaN values, select a row at random
        random_row = available.sample(n=1).iloc[0]
        player = random_row['Player']
    except ValueError:
        player = 'Trash Man'
    return player

def snake_draft(num_teams, num_rounds):
    """Generate team indices for the draft in each round."""
    draft_sort = []
    for round_num in range(1, num_rounds + 1):
        # Determine the order of teams for this round
        if round_num % 2 == 1:
            order = list(range(1, num_teams + 1))
        else:
            order = list(range(num_teams, 0, -1))
        draft_sort.extend(order)
    return draft_sort

def generate_draft_order(teams, num_rounds):
    num_teams = len(teams)
    draft_sort = snake_draft(num_teams, num_rounds)
    return [list(teams)[i - 1] for i in draft_sort]

def get_pos_maxs(players_df):
    ranked_players = players_df.dropna(subset=['PosRank'])
    pos_maxs = ranked_players.groupby('POS')['PosRank'].max().to_dict()
    pos_maxs.update({'ADP': ranked_players['ADP'].max()})
    return pos_maxs

def calculate_adp_score(player, players_df, pick_num):
    """Higher pick_num indicates higher than expected value (avg draft pos)"""
    player_row = players_df.loc[players_df['Player'] == player].iloc[0]
    max_adp = get_pos_maxs(players_df)['ADP']
    adp_player = max_adp if pd.isna(player_row['ADP']) else player_row['ADP']
    return pick_num - adp_player

def calculate_fantasy_points_score(player, players_df):
    """Project performance of the player for the season"""
    player_row = players_df.loc[players_df['Player'] == player].iloc[0]
    return 0 if pd.isna(player_row['FPTS']) else player_row['FPTS']

def calculate_rank_score(player, players_df):
    """Max position rank minus position rank"""
    player_row = players_df.loc[players_df['Player'] == player].iloc[0]
    max_rank = get_pos_maxs(players_df)[player_row['POS']]
    pos_value = player_row['PosRank']
    return 0 if pd.isna(pos_value) else max_rank - pos_value

def score_player(player, players_df, pick_num):
    player_row = players_df.loc[players_df['Player'] == player].iloc[0]
    adp = calculate_adp_score(player, players_df, pick_num)
    fpts = calculate_fantasy_points_score(player, players_df)
    rank = calculate_rank_score(player, players_df)
    player_score = {
        'Pick': player, 'POS': player_row['POS']
        , 'ADP_Score': adp, 'FPTS_Score': fpts, 'Rank_Score': rank
        }
    return player_score

def pivot_draft_history(df):
    # Create a new column 'Round_str' with leading zeros
    df['Round_str'] = df['Round'].apply(lambda x: f'Round{int(x):02d}')
    round_history = df.pivot(index='Team', columns='Round_str', values='POS')
    # Reset the index to make 'Team' a regular column
    round_history = round_history.reset_index()
    return round_history

def summarize_draft(df, metric_cols):
    team_sums = df.groupby('Team')[metric_cols].sum()
    # Reset the index to make 'Team' a regular column
    team_sums = team_sums.reset_index()

    round_history = pivot_draft_history(df)
    # Merge the two DataFrames on the 'Team' column
    draft_stats = pd.merge(team_sums, round_history, on='Team')
    branch_history = df[['Team', 'Branch', 'PickOrder']].drop_duplicates()
    draft_stats = draft_stats.merge(branch_history, on='Team')
    return draft_stats

def score_draft_strategies(team_sums, metric_cols, mn, mx):
    min_values = pd.Series(dict(zip(metric_cols, mn)))
    diff_to_min = team_sums[metric_cols] - min_values[metric_cols]
    dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=1))

    max_values = pd.Series(dict(zip(metric_cols, mx)))
    diff_to_max = team_sums[metric_cols] - max_values[metric_cols]
    dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=1))

    team_sums['DraftScore'] = dist_to_min / (dist_to_max + dist_to_min)
    return team_sums

def generate_draft(players_df, num_teams, mode='mock'):
    teams = generate_competitors(num_teams)
    branches = {k: select_branch() for k in teams}
    strategies = {k: generate_strategy(b) for k, b in branches.items()}
    rounds = min(len(v) for k, v in strategies.items())

    draft_data = []  # Collect draft data in a list of dictionaries
    selected = set()
    for j, competitor in enumerate(generate_draft_order(teams, rounds)):
        draft_round = j // num_teams + 1
        pos_selection = strategies[competitor][draft_round - 1]
        player = draft_player(players_df, pos_selection, selected)
        if mode == 'mock':
            selected.add(player)
        row = {
            'Round': draft_round,
            'Team': competitor,
            'PickNum': j + 1,
            'Branch': str(branches[competitor]),
            'PickOrder': str(strategies[competitor]),
            **score_player(player, players_df, j)
        }
        draft_data.append(row)
    return pd.DataFrame(draft_data)

n_teams, n_rounds = 5, 8
positions = ['QB', 'RB', 'WR', 'TE', 'FLEX']
metrics = ['ADP_Score', 'FPTS_Score', 'Rank_Score']
# dimensions = metrics + ['DraftScore']

pdir = 'C:\\Users\\Brown Planning\\OneDrive\\Documents\\FantasyFootball'
players = pd.read_csv(f'{pdir}\\PlayerStats.csv')

trash_team = []
for position in [p for p in positions if not p.startswith('FL')]:
    trash_man = {'Player': 'Trash Man', 'Team': 'BAD', 'POS': position}
    trash_team.append(trash_man)

players = players.append(trash_team, ignore_index=True)
initiate_generated_strategies(positions, n_rounds)

# $2.20 NFL Week 2 Sunday SPLASH
rules_branch = dict(zip(positions, (1, 2, 3, 1, 1)))
rb_key = frozenset(rules_branch.items())
generated_strategies = {rb_key: generated_strategies[rb_key]}  # filter

try:
    draft_scores = pd.read_csv(f'{pdir}\\MockDraftStats.csv')
except FileNotFoundError:
    draft_scores = pd.DataFrame()
    
i, best, iterations = 1, 0, 0
max_iter = 10

draft = generate_draft(players, n_teams, 'best')
best_drafts = summarize_draft(draft, metrics)

_mins = [0 for _ in metrics]
_maxs = list(best_drafts[metrics].apply(max))

while best < 0.95 and iterations < max_iter:
    draft = generate_draft(players, n_teams)
    summary = summarize_draft(draft, metrics)

    curr_draft_scores = score_draft_strategies(summary, metrics, _mins, _maxs)
    draft_scores = draft_scores.append(curr_draft_scores, ignore_index=True)

    local_best = curr_draft_scores['DraftScore'].max()
    local_max = draft_scores[draft_scores['DraftScore'] == local_best].iloc[0]

    keys = list(generated_strategies.keys())
    if local_best > best:
        _mins = np.minimum(_mins, summary[metrics].min())
        _maxs = np.maximum(_maxs, summary[metrics].max())
        args = metrics, _mins, _maxs
        draft_scores = score_draft_strategies(draft_scores, *args)
        
        best = draft_scores['DraftScore'].max()
        row_filter = draft_scores['DraftScore'] == best
        best_row = draft_scores[row_filter].iloc[0]
        best_metrics = best_row[metrics]

        print_data = [f'{k} {v:3,.0f}' for k, v in best_metrics.items()]
        print_data = '  '.join(print_data)
        print_data = f'{i:05d} Best {best:.4f} {print_data}'
        print(f'{print_data} {best_metrics.sum():,.0f}')
    elif i % 1000 == 0:
        processed = iterations/max_iter*100
        print(f'{i} Best {best:.4f} Processed {processed:.0f}%')
    iterations = calc_iteration()
    i += 1

draft_scores.sort_values(by=['DraftScore'], inplace=True, ascending=False)
draft_scores.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
print(best_row.transpose(), '\n')

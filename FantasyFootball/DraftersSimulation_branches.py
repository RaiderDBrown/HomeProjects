import random, itertools, math
import pandas as pd
import numpy as np

global generated_strategies
generated_strategies = dict()

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

def generate_competitors(n_teams):
    competitors = set()
    while len(competitors) < n_teams:
        competitor = generate_team_name()
        competitors.add(competitor)
    return competitors

def calculate_num_strategies(draft_branch):
    """Calculate all of the permuations by position group for the branch"""
    total_permutations = math.factorial(sum(draft_branch.values()))
    for count in draft_branch.values():  # by position group
        total_permutations //= math.factorial(count)
    return total_permutations

def initiate_generated_strategies(pos_types, n_rounds):
    """Setup the generated_strategies object"""
    global generated_strategies
    n_pos = len(pos_types)
    generator = itertools.product(range(1, n_rounds + 1), repeat=n_pos)
    for i_tuple in generator:
        draft_branch = dict(zip(pos_types, i_tuple))
        draft_branch_key = frozenset(draft_branch.items())
        if sum(i_tuple) == n_rounds:
            generated_strategies[draft_branch_key] = set()

def select_branch():
    global generated_strategies
    n_branches = len(generated_strategies)
    random_index = random.randint(0, n_branches - 1)
    random_branch_key = list(generated_strategies)[random_index]
    return dict(random_branch_key)

def generate_strategy(draft_branch, n_rounds):
    """generated_strategies is global to act as a cache for the function"""
    global generated_strategies
    draft_branch_key = frozenset(draft_branch.items())

    strategy = []
    for position, freq in draft_branch.items():
        strategy.extend([position] * freq)

    # branch_size = len(generated_strategies[draft_branch_key])
    # branch_limit = calculate_num_strategies(draft_branch)

    while True:
        strategy = random.sample(strategy, len(strategy))
        strategy_tuple = tuple(strategy)

        if strategy_tuple not in generated_strategies[draft_branch_key]:
            generated_strategies[draft_branch_key].add(strategy_tuple)
            return strategy

        # if branch_size < branch_limit:
        #     generated_strategies[draft_branch_key].add(strategy_tuple)
        #     return strategy

def draft_player(player_pool, position):
    if position == 'FLEX':
        available = player_pool[player_pool['POS'].isin(['RB', 'WR', 'TE'])]
    else:
        available = player_pool[player_pool['POS'] == position ]
    # Filter rows where 'Rank' is not NaN
    valid_rows = available[~available['Rank'].isna()]
    if not valid_rows.empty:
        # Select the row with the lowest 'Rank'
        lowest_rank_row = valid_rows.loc[valid_rows['Rank'].idxmin()]
        player = lowest_rank_row['Player']
        return player
    try:
        # If 'Rank' contains only NaN values, select a row at random
        random_row = available.sample(n=1).iloc[0]
        player = random_row['Player']
    except ValueError:
        player = 'Trash Man'
    return player

def snake_draft(n_teams, n_rounds):
    """Generate team indices for the draft in each round."""
    draft_sort = []
    for round_num in range(1, n_rounds + 1):
        # Determine the order of teams for this round
        if round_num % 2 == 1:
            order = list(range(1, n_teams + 1))
        else:
            order = list(range(n_teams, 0, -1))
        draft_sort.extend(order)
    return draft_sort

def generate_draft_order(teams, n_teams, n_rounds):
    draft_sort = snake_draft(n_teams, n_rounds)
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

    branch_history = df[['Team', 'Branch']].drop_duplicates()
    draft_stats = draft_stats.merge(branch_history, on='Team')
    return draft_stats

def round_up(number):
    return math.ceil(number / 100) * 100

def score_draft_strategies(team_sums, metric_cols, mn, mx):
    min_values = pd.Series(dict(zip(metric_cols, mn)))
    diff_to_min = team_sums[metric_cols] - min_values[metric_cols]
    dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=1))

    max_values = pd.Series(dict(zip(metric_cols, mx)))
    diff_to_max = team_sums[metric_cols] - max_values[metric_cols]
    dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=1))

    team_sums['DraftScore'] = dist_to_min / (dist_to_max + dist_to_min)
    return team_sums

# Define a function to normalize a column to the range [0, 1]
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

def normalize_specific_columns(team_sums, columns_to_normalize):
    # Normalize the specified columns
    normalized_df = pd.DataFrame(columns=columns_to_normalize)
    for column in columns_to_normalize:
        normalized_df[column] = normalize_column(team_sums[column])
    normalized_df['Score'] = normalized_df[columns_to_normalize].sum(axis=1)
    normalized_df['Score'] /= len(columns_to_normalize)

    team_sums['DraftScore'] = normalized_df['Score']
    return team_sums

def generate_mock_draft(players_df, pos_types, n_teams, n_rounds):
    competitors = generate_competitors(n_teams)
    strategies = {k: select_branch() for k in competitors}
    picks_order = {k: generate_strategy(b, n_rounds) for k, b in strategies.items()}

    draft_order = generate_draft_order(competitors, n_teams, n_rounds)
    draft_data = []  # Collect draft data in a list of dictionaries
    selected = set()
    for j, competitor in enumerate(draft_order):
        draft_round = j // n_teams + 1
        filter_selected = ~players_df['Player'].isin(selected)
        player_pool = players_df[filter_selected]
        selection = picks_order[competitor][draft_round - 1]
        player = draft_player(player_pool, selection)
        selected.add(player)
        row = {'Round': draft_round, 'Team': competitor,'PickNum': j + 1}
        row.update(score_player(player, players_df, j))
        row.update({'Branch': str(strategies[competitor])})
        draft_data.append(row)
    mock_draft = pd.DataFrame(draft_data)
    return mock_draft

n_teams, n_rounds = 5, 8
positions = ['QB', 'RB', 'WR', 'TE', 'FLEX']
metrics = ['ADP_Score', 'FPTS_Score', 'Rank_Score']
dimensions = metrics + ['DraftScore']

pdir = 'C:\\Users\\Brown Planning\\OneDrive\\Documents\\FantasyFootball'
players = pd.read_csv(f'{pdir}\\PlayerStats.csv')

trash_team = []
for position in positions[:4]:
    trash_man = {'Player': 'Trash Man', 'Team': 'BAD', 'POS': position}
    trash_team.append(trash_man)

players = players.append(trash_team, ignore_index=True)
initiate_generated_strategies(positions, n_rounds)

# $2.20 NFL Week 2 Sunday SPLASH
rules_branch = dict(zip(positions, (1, 2, 3, 1, 1)))
generated_strategies = {frozenset(rules_branch.items()): set()}
#

draft_scores = pd.DataFrame()
i, best = 1, 0
_mins, _maxs = [0 for _ in metrics], [0 for _ in metrics]
while best < 0.945:
    draft = generate_mock_draft(players, positions, n_teams, n_rounds)
    summary = summarize_draft(draft, metrics)

    curr_draft_scores = score_draft_strategies(summary, metrics, _mins, _maxs)
    draft_scores = draft_scores.append(curr_draft_scores, ignore_index=True)

    local_best = curr_draft_scores['DraftScore'].max()
    local_max = draft_scores[draft_scores['DraftScore'] == local_best].iloc[0]

    keys = list(generated_strategies.keys())
    processed = sum([len(generated_strategies[b]) for b in keys])
    possible = sum([calculate_num_strategies(dict(b)) for b in keys])

    if local_best > best:
        _mins = np.minimum(_mins, summary[metrics].min())
        _maxs = np.maximum(_maxs, summary[metrics].max()).apply(round_up)
        draft_scores = score_draft_strategies(summary, metrics, _mins, _maxs)
        score_col = draft_scores['DraftScore']

        best = score_col.max() if i > 1 else score_col.min() # stop premature end
        row_filter = draft_scores['DraftScore'] == best
        best_row = draft_scores[row_filter].iloc[0]
        best_metrics = best_row[metrics]
        best_picks = draft[draft['Team'] == best_row['Team']]

        print_data = [f'{k} {v:3,.0f}' for k, v in best_metrics.items()]
        print_data = '  '.join(print_data)
        print_data = f'{i:05d} Best {best:.4f} {print_data}'
        print(f'{print_data} {best_metrics.sum():,.0f}')
    elif i % 1000 == 0:
        print(f'{i} Best {best:.4f} Processed {(processed/possible*100):.0f}%')
    elif processed == possible:
        print('The End \n')
        break
    i += 1

draft_scores.sort_values(by=['DraftScore'], inplace=True, ascending=False)
draft_scores.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
print(best_row.transpose(), '\n')
print(best_picks[['Round', 'PickNum', 'Pick', 'POS', 'ADP_Score'
                  , 'FPTS_Score', 'Rank_Score']])

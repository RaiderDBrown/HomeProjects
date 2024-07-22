import random
import pandas as pd
import numpy as np

global generated_strategies  
global generated_drafts
generated_strategies = set() 
generated_drafts = set()

pdir = 'C:\\Users\\Brown Planning\\OneDrive\\Documents\\FantasyFootball'
players = pd.read_csv(f'{pdir}\\PlayerStats.csv')

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

def generate_competitors(n_teams=12):
    competitors = set()
    while len(competitors) < n_teams:
        competitor = generate_team_name()
        competitors.add(competitor)
    return competitors

def generate_strategy(choices=tuple(['QB', 'RB', 'WR', 'TE']), n_rounds=20):
    """generated_strategies is global to act as cache for function"""
    while True:
        strategy = [random.choice(choices) for _ in range(n_rounds)]
        if tuple(strategy) not in generated_strategies:
            generated_strategies.add(tuple(strategy))  # Add the strategy to the set
            return strategy  # Return the unique strategy

def draft_player(player_pool, position):
    available = player_pool[player_pool['POS'] == position ]
    # Filter rows where 'Rank' is not NaN
    valid_rows = available[~available['Rank'].isna()]
    if not valid_rows.empty:
        # Select the row with the lowest 'Rank'
        lowest_rank_row = valid_rows.loc[valid_rows['Rank'].idxmin()]
        return lowest_rank_row['Player']
    # If 'Rank' contains only NaN values, select a row at random
    random_row = available.sample(n=1).iloc[0]
    return random_row['Player']

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

def generate_draft_order(teams, n_teams=12, n_rounds=20):
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
    pos_rank_player = max_rank if pd.isna(pos_value) else pos_value
    return max_rank - pos_rank_player

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
    return draft_stats

def score_draft_strategies(team_sums, metric_cols):
    # Hard coding temporarily
    mn = [-2750, 1100, 850]
    mx = [750, 3750, 2650]
    
    #min_values = team_sums.min()
    min_values = pd.Series(dict(zip(metric_cols, mn)))
    diff_to_min = team_sums[metric_cols] - min_values[metric_cols]
    dist_to_min = np.sqrt(np.sum(diff_to_min ** 2, axis=1))
    
    #max_values = team_sums.max()
    max_values = pd.Series(dict(zip(metric_cols, mx)))
    diff_to_max = team_sums[metric_cols] - max_values[metric_cols]
    dist_to_max = np.sqrt(np.sum(diff_to_max ** 2, axis=1))    
    team_sums['DraftScore'] = dist_to_min / (dist_to_max + dist_to_min)
    return team_sums

def normalize_specific_columns(team_sums, columns_to_normalize):
    # Define a function to normalize a column to the range [0, 1]
    def normalize_column(col):
        return (col - col.min()) / (col.max() - col.min())

    # Normalize the specified columns
    normalized_df = pd.DataFrame(columns=columns_to_normalize)
    for column in columns_to_normalize:
        normalized_df[column] = normalize_column(team_sums[column])
    normalized_df['Score'] = normalized_df[columns_to_normalize].sum(axis=1)
    normalized_df['Score'] /= len(columns_to_normalize)
    
    team_sums['DraftScore'] = normalized_df['Score']
    return team_sums

def generate_mock_draft(players_df, n_teams=12, n_rounds=20):
    while True:
        competitors = generate_competitors()
        strategies = {k: generate_strategy() for k in competitors}
        draft_order = generate_draft_order(competitors, n_teams, n_rounds)
    
        draft_data = []  # Collect draft data in a list of dictionaries
        selected = set()
        for j, competitor in enumerate(draft_order):
            draft_round = j // n_teams + 1
            filter_selected = ~players_df['Player'].isin(selected)
            player_pool = players_df[filter_selected]
            selection = strategies[competitor][draft_round - 1]
            player = draft_player(player_pool, selection)
            selected.add(player)
            row = {'Round': draft_round, 'Team': competitor,'PickNum': j + 1}
            row.update(score_player(player, players_df, j))
            draft_data.append(row)
        mock_draft = pd.DataFrame(draft_data)
        
        draft_combo = tuple(map(tuple, mock_draft[['PickNum', 'Pick']].values))
        if draft_combo not in generated_drafts:
            generated_drafts.add(draft_combo)
            return mock_draft
        return 'Ran out of drafts to simulate'

metrics = ['ADP_Score', 'FPTS_Score', 'Rank_Score']
dimensions = metrics + ['DraftScore']

draft_scores = pd.DataFrame()
i, best = 0, 0
while best < 0.89:
    draft = generate_mock_draft(players)
    summary = summarize_draft(draft, metrics)
    curr_draft_scores = score_draft_strategies(summary, metrics)
    draft_scores = draft_scores.append(curr_draft_scores, ignore_index=True)
    
    current_best = curr_draft_scores['DraftScore'].max()
    local_max = draft_scores[draft_scores['DraftScore'] == current_best]
    if current_best > best:
        best = current_best
        best_metrics = local_max[dimensions].to_dict(orient='records')[0]
        global_max = draft_scores[draft_scores['DraftScore'] == best]
        formatted_data = {k: f'{v:,.4f}' for k, v in best_metrics.items()}
        print(f'{i:05d} tries Best: {formatted_data}')
    elif i % 100 == 0:
        print(i, best)
    i += 1

draft_scores.sort_values(by=['DraftScore'], inplace=True, ascending=False)
draft_scores.to_csv(f'{pdir}\\MockDraftStats.csv', index=False)
print(global_max)


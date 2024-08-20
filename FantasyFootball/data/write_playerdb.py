# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 01:12:31 2024

@author: DouglasBrown
"""
import os
import glob
import re
import Levenshtein as lev
import pandas as pd
import nfl_data_py as nfl

def similarity(input_string, comparison_string):
    comparison_string = str(comparison_string)
    distance = lev.distance(input_string, comparison_string)
    max_distance = max(len(input_string), len(comparison_string))
    ratio = 1 - distance / max_distance
    return ratio

def find_closest(plyr_str, comp_group):
    """Find the closest match to the input within a list of values"""
    best_match = None
    max_similarity = -float('inf')

    for candidate in comp_group:
        dist = similarity(plyr_str, candidate)
        if dist > max_similarity:
            max_similarity = dist
            best_match = candidate

    if best_match is not None:
        return best_match
    return plyr_str

def fetch_pid(row):
    """Find players without team assignments"""
    global roster

    plyr, pos, yr = row['Player'], row['Pos'], row['season']
    key = tuple([plyr, pos, yr])

    if pos == 'DST':
        return f"{row['Team']}_DST"

    if key in roster.index:
        df = roster.loc[key]
    elif key not in roster.index:
        cross_sect = roster.xs(yr, level='season', drop_level=False)
        csindex = cross_sect.index
        pnames_list = list(csindex.get_level_values('player_name').unique())

        pname = find_closest(row['Player'], pnames_list)
        df = cross_sect.loc[(pname)]

    if len(df) == 1:
        return df['player_id'].values[0]
    elif len(df) > 1:  # players with the same name
        distances = (df['total_yards'] - row['TYDS']) ** 2
        return df.loc[distances.idxmin(), 'player_id']

def fetch_team(row):
    global roster_pid
    team_aliases = {'LAR':['LAR', 'SL', 'LA'], 'HOU':['HOU', 'HST']
            , 'BAL':['BAL', 'BLT'], 'CLE':['CLE', 'CLV'], 'ARI':['ARI', 'ARZ']
            , 'JAC':['JAC', 'JAX'], 'LV':['LV', 'OAK'], 'LAC':['LAC', 'SD']
            }
    tmap = {itm: key for key, vals in team_aliases.items() for itm in vals}

    if row['player_id'] in roster_pid.index:
        rdf = roster_pid.loc[(row['player_id'], row['season'])]
        team = rdf['team'].values[0]
        if team in tmap:
            return tmap[team]
        return team
    return row['Team']

def fix_colnames(df, pos):
    if re.search(r'WR|TE', pos):
        fix = {'YDS': 'PYDS', 'YDS.1': 'RYDS', 'TD': 'PTD', 'TD.1': 'RTD'}
    elif re.search('RB', pos):
        fix = {'YDS': 'RYDS', 'YDS.1': 'PYDS', 'TD': 'RTD', 'TD.1': 'PTD'}
    elif re.search('QB', pos):
        fix = {'YDS': 'PYDS', 'YDS.1': 'RYDS', 'TD': 'PTD', 'TD.1': 'RTD'}
        df['ATT'] = df['ATT'] + df['ATT.1']
        df = df.drop('ATT.1', axis=1)
    else:
        fix = {}

    df.columns = [fix[col] if col in fix else col for col in df.columns]
    for col in ['PYDS', 'RYDS', 'PTD', 'RTD', 'FGA', 'XPA', 'DEF TD', 'SPC TD']:
        if col in df.columns:
            df[col] = df[col].replace({',': ''}, regex=True).astype(int)

    if re.search(r'WR|TE|RB|QB', pos):
        df['TYDS'] = df['PYDS'] + df['RYDS']
        df['TTDS'] = df['PTD'] + df['RTD']
    elif re.search('K', pos):
        df['ATT'] = df['FGA'] + df['XPA']
        df['TYDS'] = 0
    elif re.search('DST', pos):
        df['TTDS'] = df['DEF TD'] + df['SPC TD']
        df['TYDS'] = 0
    return df

def parse_player(plyr_str):
    """Parse the team assignments from player name field"""
    pattern = re.compile(r'([\w\'-.]+)\s*([\w\'-.]+)\s*([\w.]+)*\s*\((\w+)\)')
    match = pattern.search(plyr_str)
    pname, team = plyr_str, None
    if match:
        nmparts = match.groups()
        pname, team = ' '.join(filter(None, nmparts[:3])), nmparts[3]
    return pname, team

module_dir = os.path.dirname(os.path.abspath(__file__))
filename = 'FantasyPros_Fantasy_Football_Statistics'

roster = nfl.import_seasonal_rosters(range(2002, 2024))
stats = nfl.import_seasonal_data(range(2002, 2024))
roster = pd.merge(roster, stats, on=['player_id', 'season'], how='left')

roster['rushing_yards'] = roster['rushing_yards'].fillna(0)
roster['passing_yards'] = roster['passing_yards'].fillna(0)
roster['total_yards'] = roster['rushing_yards'] + roster['passing_yards']

idx = ['player_name', 'position', 'season', 'team']
roster = roster.set_index(idx).sort_index()  # Ensure index is sorted
roster = roster.loc[~roster.index.duplicated(keep='first')]  # drop duplicates
roster = roster[['player_id', 'total_yards']]

roster_pid = roster.reset_index()
roster_pid = roster_pid.set_index(['player_id', 'season']).sort_index()

data = pd.DataFrame()
for year_dir in os.listdir(module_dir):
    if os.path.isdir(year_dir):
        print(f"Processing {year_dir}...")
        filepath = os.path.join(module_dir, year_dir, filename)
        for data_file in glob.glob(f"{filepath}*.csv"):
            adf = pd.read_csv(data_file)
            pos = re.search(r'_([A-Z]+).csv', str(data_file)).group(1)
            adf['Pos'] = pos
            adf['season'] = int(year_dir)
            adf = adf.dropna(subset=['Player'])  # remove blanks
            adf = fix_colnames(adf, pos)

            plyrteam_tuples = adf['Player'].apply(parse_player)
            adf[['Player', 'Team']] = plyrteam_tuples.apply(pd.Series)

            adf['player_id'] = adf.apply(fetch_pid, axis=1)
            adf['Team'] = adf.apply(fetch_team, axis=1)
            data = pd.concat([data, adf], axis=0).reset_index(drop=True)
cols = ['season', 'Pos', 'Player', 'player_id', 'Team', 'ATT'
        , 'PYDS', 'RYDS', 'TYDS', 'TTDS', 'FPTS']
data = data[cols]
numerics = data.select_dtypes(include='number')
data[numerics.columns] = numerics.fillna(0)

# data = data.sort_values(['Player', 'Season', 'Team'])
# data[['Actual_FPTS', 'Actual_G']] = data[['FPTS', 'G']].shift(-1)

data.to_csv('player_db.csv', index=False)

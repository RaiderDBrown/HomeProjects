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

def find_closest(input_group, comp_groups):
    """Find the closest match to the input within a list of values"""
    plyr_name, plyr_lname = input_group
    best_match_lastname = None
    best_match_any = None
    max_similarity_lastname = -float('inf')
    max_similarity_any = -float('inf')

    for fullname, lastname in comp_groups:
        dist = similarity(plyr_name, fullname)
        if plyr_lname == lastname:
            if dist > max_similarity_lastname:
                max_similarity_lastname = dist
                best_match_lastname = fullname, lastname
        elif dist > max_similarity_any:
            max_similarity_any = dist
            best_match_any = fullname, lastname

    if best_match_lastname is not None:
        return best_match_lastname
    return best_match_any

def fetch_pid(df, year):
    """Find players without team assignments"""
    roster = nfl.import_seasonal_rosters([year])
    roster = roster.set_index(['player_name', 'last_name'])
    roster = roster.sort_index()
    roster = roster[['team', 'player_id']]

    result = []
    key = ['Player', 'last_name', 'Pos', 'team']
    for plyr, lname, pos, dfteam in df[key].values:
        try:
            team, pid = roster.loc[(plyr, lname)].values.tolist()[0]
        except:
            plyr_db = list(roster.index)
            closest, lastname = find_closest((plyr, lname), plyr_db)
            team, pid = roster.loc[closest, lastname].values.tolist()[0]
        if dfteam == 'FA':
            result.append({'Team': team, 'player_id': pid})
        else:
            result.append({'Team': dfteam, 'player_id': pid})
    return pd.DataFrame(result)

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
    elif re.search('DST', pos):
        df['TTDS'] = df['DEF TD'] + df['SPC TD']
    return df

def parse_team(df):
    """Parse the team assignments from player name field"""
    pattern = r'([\w\'-.]+)\s*([\w\'-.]+)\s*([\w.]+)*\s*\((\w+)\)'
    results = []
    for player in df['Player'].values:
        pname, team = player, 'FA'
        match = re.search(pattern, player)
        if match:
            fname, lname, suffix, team = match.groups()
            if suffix is not None:
                pname = f'{fname} {lname} {suffix}'
                last_name = f'{lname} {suffix}'
            else:
                pname = f'{fname} {lname}'
                last_name = f'{lname}'

        info = {'Player': pname, 'Team': team, 'last_name': last_name}
        results.append(info)
    return pd.DataFrame(results)

module_dir = os.path.dirname(os.path.abspath(__file__))
filename = 'FantasyPros_Fantasy_Football_Statistics'

data = pd.DataFrame()
for year_dir in os.listdir(module_dir):
    if os.path.isdir(year_dir):
        print(f"Processing {year_dir}...")
        filepath = os.path.join(module_dir, year_dir, filename)
        for data_file in glob.glob(f"{filepath}*.csv"):
            adf = pd.read_csv(data_file)
            pos = re.search(r'_([A-Z]+).csv', str(data_file)).group(1)
            adf['Pos'] = pos
            adf['Season'] = year_dir
            adf = adf.dropna(subset=['Player'])  # remove blanks
            adf = fix_colnames(adf, pos)
            adf[['Player', 'team', 'last_name']] = parse_team(adf)
            adf[['team', 'player_id']] = fetch_pid(adf, int(year_dir))
            data = pd.concat([data, adf], axis=0).reset_index(drop=True)
cols = ['Season', 'Pos', 'Player', 'last_name'
        , 'player_id'
        , 'team', 'ATT'
        , 'PYDS', 'RYDS', 'TYDS', 'TTDS', 'FPTS']
data = data[cols]

# data = data.sort_values(['Player', 'Season', 'Team'])
# data[['Actual_FPTS', 'Actual_G']] = data[['FPTS', 'G']].shift(-1)

data.to_csv('player_db.csv')

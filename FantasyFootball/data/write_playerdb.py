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

def parse_team(df):
    """Parse the team assignments from player name field"""
    pattern = r'([\w\'-.]+)\s*([\w\'-.]+)\s*([\w.]+)*\s*\((\w+)\)'
    results = []
    for player in df['Player'].values:
        pname, team = player, 'FA'
        match = re.search(pattern, player)
        if match:
            if match.group(3) is not None:
                pname = f'{match.group(1)} {match.group(2)} {match.group(3)}'
            else:
                pname = f'{match.group(1)} {match.group(2)}'
            last_name = match.group(2)
            team = match.group(4)
        info = {'Player': pname.strip(), 'Team': team, 'last_name': last_name}
        results.append(info)
    return pd.DataFrame(results)

def similarity(input_string, comparison_string):
    comparison_string = str(comparison_string)
    distance = lev.distance(input_string, comparison_string)
    max_distance = max(len(input_string), len(comparison_string))
    ratio = 1 - distance / max_distance
    return ratio

def find_closest(fullname, groups):
    """Find the closest match to the input within a list of values"""
    lastname = fullname.split()[1]
    best_match_lastname = None
    best_match_any = None
    max_similarity_lastname = -float('inf')
    max_similarity_any = -float('inf')

    for player, plyr_lastname in groups:
        if fullname == player:
            return player, plyr_lastname

        dist = similarity(fullname, player)

        if lastname == plyr_lastname:
            if dist > max_similarity_lastname:
                max_similarity_lastname = dist
                best_match_lastname = player, plyr_lastname
        else:
            if dist > max_similarity_any:
                max_similarity_any = dist
                best_match_any = player, plyr_lastname

    if best_match_lastname is not None:
        return best_match_lastname
    return best_match_any

def fetch_orphans(df, year):
    """Find players without team assignments"""
    roster = nfl.import_seasonal_rosters([year])
    gp = roster.groupby(['player_name', 'last_name'])

    result = []
    key = ['Player', 'Team']
    for player, team in df[key].values:
        if re.search('FA', team):
            pname, lname = find_closest(player, gp.groups)
            if pname:
                team = gp.get_group((pname, lname))['team'].values[0]
        result.append(team)
    return result

module_dir = os.path.dirname(os.path.abspath(__file__))
filename = 'FantasyPros_Fantasy_Football_Statistics'

data = pd.DataFrame()
players = set()
for year_dir in os.listdir(module_dir):
    if os.path.isdir(year_dir):
        print(f"Processing {year_dir}...")
        filepath = os.path.join(module_dir, year_dir, filename)
        for data_file in glob.glob(f"{filepath}*.csv"):
            adf = pd.read_csv(data_file)[['Player', 'G', 'FPTS']]
            adf['Pos'] = re.search(r'_([A-Z]+).csv', str(data_file)).group(1)
            adf['Season'] = year_dir

            adf = adf.dropna(subset=['Player'])  # remove blanks
            adf = adf[adf['FPTS'] > 0]

            adf[['Player', 'Team', 'last_name']] = parse_team(adf)
            adf['Team'] = fetch_orphans(adf, int(year_dir))
            data = pd.concat([data, adf], axis=0).reset_index(drop=True)
data = data.sort_values(['Player', 'Team', 'Season'])
data[['Actual_FPTS', 'Actual_G']] = data[['FPTS', 'G']].shift(-1)

data.to_csv('foo.csv')

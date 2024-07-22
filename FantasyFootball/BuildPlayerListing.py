# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import re

def extract_digits(input_string):
    pattern = r'\d+'  # This regex pattern matches one or more digits
    match = re.search(pattern, input_string)  # Search for the pattern in the input string
    return match.group() if match else None

def extract_text(input_string):
    pattern = r'\d+'  # This regex pattern matches one or more digits
    parts = re.split(pattern, input_string)  # Split at first digits
    return parts[0] if parts else None

pdir = 'C:\\Users\\Brown Planning\\OneDrive\\Documents\\FantasyFootball'
adp = pd.read_csv(f'{pdir}\\FantasyPros_2023_Overall_ADP_Rankings.csv')
dst = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_DST.csv')
flx = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_FLX.csv')    
kik = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_K.csv')
qbs = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_QB.csv')
wrs = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_WR.csv')
rbs = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_RB.csv')
tes = pd.read_csv(f'{pdir}\\FantasyPros_Fantasy_Football_Projections_TE.csv')

alldfs = ['adp', 'dst', 'kik', 'qbs', 'wrs', 'rbs', 'tes', 'flx']
pos_dict = {'dst': 'DST', 'kik': 'K', 'qbs': 'QB'
            , 'wrs': 'WR', 'rbs': 'RB', 'tes': 'TE'}

allplayers = set()
for df_name in alldfs:
    df = locals()[df_name]
    for index, row in df.iterrows():
        player = row['Player']
        if type(player) == str and player not in ['\xa0']:
            allplayers.add(row['Player'])

flds = ['Player', 'Team', 'POS', 'ADP', 'FPTS', 'PosRank', 'Rank']
players = pd.DataFrame(columns=flds)

for player in allplayers:
    new_row = {'Player': player}
    for df_name in alldfs:
        df = locals()[df_name]
        if player in set(df['Player']):
            for index, row in df.iterrows():
                if player == row['Player']:
                    team = str(row['Team'])
                    if team.lower() != 'nan':
                        new_row['Team'] = team
                    if 'POS' in row and str(row['POS']).lower != 'nan':
                        new_row['POS'] = extract_text(row['POS'])
                    if df_name in pos_dict:
                        new_row['POS'] = pos_dict[df_name]
                        new_row['FPTS'] = row['FPTS']
                    if df_name == 'adp':
                        new_row['PosRank'] = extract_digits(row['POS'])
                        new_row['ADP'] = row['AVG']
                        new_row['Rank'] = row['Rank']
    players = players.append(new_row, ignore_index=True)

players.to_csv(f'{pdir}\\PlayerStats.csv', index=False)            
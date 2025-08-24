# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:13:18 2024

@author: BrownPlanning
"""
from pathlib import Path

import pandas as pd
import scipy.stats as stats

def abbrev_level(text):
    abbreviations = {
        'ECC': 'Early Childhood', 'ES': 'Elementary', 'MS': 'Middle', 'HS': 'High'
                    }
    for key, val in abbreviations.items():
        if val in text:
            return key

def t_test(series):
    t_statistic, p_value = stats.ttest_1samp(series, 0)
    if p_value < 0.05:
        conclusion = 'There is a significant difference.'
        print(f"Reject null hypothesis for {series.name} p-value {p_value:.2f} {conclusion}")
    else:
        conclusion = 'There is no significant difference.'
        print(f"Fail to reject null hypothesis for {series.name} p-value {p_value:.2f} {conclusion}")
    return p_value

def convert_to_numeric(column):
    return pd.to_numeric(column.str.replace(',', ''), errors='coerce')

if __name__ == "__main__":
    data_file = Path(__file__).with_name("Behavior_20240131.csv")
    df = pd.read_csv(data_file)

    months = [list(df.columns[2:])[i] for i in range(0, len(df.columns[2:]), 2)]
    years = sorted(list(set(df.iloc[0, 2:].values.tolist())))
    colnames = [f"{month} {year}" for month in months for year in years]
    df.columns = list(df.columns[:2]) + colnames
    df = df.drop(0)  # drop years row
    df[colnames] = df[colnames].apply(convert_to_numeric)

    # Select rows where 'School Type' contains 'Total'
    totals = df[df['School Type'].str.contains('Total', na=False)]
    totals = totals.drop('Behavior Category', axis=1)
    totals['School Type'] = totals['School Type'].apply(abbrev_level)

    differences = pd.DataFrame()
    differences['School Type'] = totals['School Type']
    columns_to_rename = list(totals.columns[1:])
    for month in months:
        last_year, curr_year = [f"{month} {year}" for year in years]
        differences[month] = totals[curr_year] - totals[last_year]
    differences = differences.set_index('School Type')
    differences = differences.transpose()

    differences.apply(t_test)

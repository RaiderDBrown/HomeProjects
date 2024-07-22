# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 01:01:04 2020

@author: BrownPlanning
"""

import string
import random
import pandas as pd
from dateutil.parser import parse
import time


def rand_df(rows=5, strcols=5, numcols=0):
    upr, lwr = string.ascii_uppercase, string.ascii_lowercase
    df = {}
    # string cols
    for k in range(strcols):
        name = ''.join(random.sample(upr, 2))
        df[name] = [''.join(random.sample(lwr, 3)) for x in range(rows)]

    keys = [x for x in df]
    int_col = random.choice(keys)  # insert number in categorical col
    nan_col = random.choice(keys)  # insert nan in categorical col
    for x in random.sample(range(rows), 3):
        if x % 2:
            df[int_col][x] = int(random.random() * 1000)
        else:
            df[nan_col][x] = float('nan')

    # add numeric cols
    for k in range(numcols):
        name = random.choice(string.ascii_uppercase)
        df[name] = [random.randint(0, 100) for x in range(rows)]
    return pd.DataFrame(df)

def str_cols(df):
    uniques = {k: list(df[k].unique()) for k in df}
    is_str = lambda v: any([isinstance(x, str) for x in v])
    return [k for k, v in uniques.items() if is_str(v)]

def indicators(df, categories):
    """ Create a full set of indicator variables on a subsetted dataframe. The
    subsetted dataframe may not contain all the values for each categorical
    variable.
    Args:
        df (pandas dataframe): Subset dataframe of a larger file.
        categories (dict): Unique values for each categorical variable in the
        larger file.
    Returns:
        matrix (pandas dataframe): Full set of indicator variables
    """
    matrix = df.copy()
    for k, v in categories.items():
        missing = list(set(v) - set(matrix[k].unique()))
        df[k] = df[k].astype('category')
        df[k] = df[k].cat.add_categories(missing)
    return pd.get_dummies(df, columns=categories.keys())  # create indicators

def create_interaction(df, var1, var2):
    name = var1 + "*" + var2
    try:
        df[name] = pd.Series(df[var1] * df[var2], name=name)
    except TypeError:
        var = [f"{x}_{y}" for x, y in zip(df[var1], df[var2])]
        df[name] = pd.Series(var, name=name)
    return df

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy, ignoretz=True)
        return True
    except (ValueError, OverflowError, TypeError):
        return False

def downcast(series):
    typ = str(series.dtypes)
    vals = list(series.unique())
    if 'int' in typ:
        return pd.to_numeric(series, downcast='integer')
    if 'float' in typ:
        return pd.to_numeric(series, downcast='float')
    if any([is_date(x) for x in vals]):
        return pd.to_datetime(series, errors='ignore')
    if any([isinstance(x, str) for x in vals]):
        return series.astype('category') 

def redux(df):
    for col in df:
        df[col] = downcast(df[col])
    return df

def categorize(df, col):
    return pd.get_dummies(df, columns=[col])

foo = rand_df(10, 3, 5)
# cat_cols = str_cols(foo)  # save categorical column names
# cat_cols = {k: categorize for k in cat_cols}
# print(foo)
# categories = {k: list(foo[k].unique()) for k in cat_cols}
# col = random.choice(list(categories.keys()))
# categories[col] = categories[col] + ['Yay']  # add categories to col
# foo = indicators(foo, categories)
# foo = create_interaction(foo, foo.columns[0], foo.columns[1])
# # print(foo)

# path = 'C:\\Users\\Brown Planning\\Documents\\Innova\\Datasets\\'

# tic = time.perf_counter()
# # Write to hdf5 and read back into memory
# name = 'data.h5'
# file = path + name
# foo.to_hdf(file, key='foo', index=False)
# bar = pd.read_hdf(file)
# toc = time.perf_counter()
# print(f"HDF5 processing took {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# # Write to csv and read back into memory
# name = 'data.csv'
# file = path + name
# foo.to_csv(file)
# bar = pd.read_csv(file)
# toc = time.perf_counter()
# print(f"CSV processing took {toc - tic:0.4f} seconds")

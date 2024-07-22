# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:14:20 2020
@author: Douglas Brown
"""
import random

committees = ['Huntington', 'Policy', 'Budget', 'Govt Relations',
              'New Horizons', 'NNEF', 'HRETA']
brd_members = ['Surles-Law', 'Aman', 'Eley', 'Best',
               'Harris', 'Hunter', 'Brown']
preferences = {'Surles-Law': ['Budget', 'Govt Relations', 'Policy'],
               'Aman': ['Policy', 'Budget', 'NNEF'],
               'Eley': ['Huntington', 'New Horizons', None],
               'Best': ['Huntington', 'Policy', 'Budget'],
               'Harris': ['New Horizons', 'Huntington', 'HRETA'],
               'Hunter': ['Huntington', 'Budget', 'NNEF'],
               'Brown': ['Budget', 'HRETA', None]}

def invert(x_dict):
    inverted = {}
    for k, v in x_dict.items():
        for x in v:
            if x in inverted:
                inverted[x].append(k)
            else:
                inverted[x] = [k]
    return inverted

def select(members, committees):
    """
    Randomly assign a primary board member and secondary board member to
    each committee. The primary board member cannot equal the secondary
    board member.

    Args:
        members (list): The board members to be assigned.
        committees (list): The committees for assignment.
    Returns:
        selections (dict): {member: [committee assignments]}.
    """
    choice1 = random.sample(committees, len(members))
    selections = {k: [choice1[i]] for i, k in enumerate(members)}
    choice2 = choice1
    while any([x == y for x, y in zip(choice1, choice2)]):
        choice2 = random.sample(committees, len(members))
    for i, k in enumerate(selections):
        selections[k].append(choice2[i])
    selections = {k: v for k, v in sorted(selections.items())}
    return selections

def get_scores(assignments):
    scores = {member: 0 for member in preferences}
    for member in assignments:
        first_pref, secnd_pref, third_pref = preferences[member]
        result = assignments[member]
        if first_pref in result and secnd_pref in result:
            scores[member] = 1
        elif first_pref in result and third_pref in result:
            scores[member] = 2
        elif first_pref in result:
            scores[member] = 3
        elif secnd_pref in result and third_pref in result:
            scores[member] = 4
        elif secnd_pref in result:
            scores[member] = 5
        elif third_pref in result:
            scores[member] = 8
        else:
            scores[member] = 10
    return scores


best_score = 16  # 14 is the best score
best = {}
for i in range(2):
    score = best_score
    j = 0
    while score >= best_score:
        selection = select(brd_members, committees)
        rating = get_scores(selection)
        score = sum(rating.values())
        j += 1
        if j % 2000000 == 0:
            print('{:,} attempts to improve {}'.format(j, best_score))
    print(i, score)
    best[i] = selection
    best_score = score

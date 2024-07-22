# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:36:23 2020

@author: BrownPlanning
"""
import random
import math
y = [random.randint(1, 4) for v in range(10)]
y
def scale(x, r_min, r_max, t_min, t_max):
    """
    Args:
        x (number): value to be transformed.
        r_min (number): min of range for x
        r_max (number): max of range for x
        t_min (number): min of target range
        t_max (number): max of target range
    Returns:
        scaled (number): value in new scale
    """
    x_range = (r_max - r_min)
    t_range = (t_max - t_min)
    scaled = (x - r_min)/x_range * t_range + t_min
    return scaled

s = [scale(v, 1, 4, 0.5, 4.5) for v in y]  # scale [0.5, 4.5]
s
y_transformed = [math.log(v) for v in s]
y_transformed

y_rescaled = [scale(math.exp(v), 0, 4.5, 1, 4) for v in y_transformed]
y_rescaled

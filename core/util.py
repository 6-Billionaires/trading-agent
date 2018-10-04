# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:36:28 2018
 @author: UK JO
"""
import numpy as np
dat = np.arange(1, 13) / 2.0


def discretize(data, bins):
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
    return discrete, cutoffs


def get_maxlen_of_binary_array(max_seconds):
    return len(np.binary_repr(max_seconds))


def seconds_to_binary_array(seconds, max_len):
    return np.binary_repr(seconds).zfill(max_len)
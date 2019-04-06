import os
import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT


### ITERATION CHECKS ###

def check_cost(network, data, labels, costs, ms, k=1, k_max=1):
    """ run every n iterations to check network cost  """
    if k % k_max == 0:
        costs, ms = get_cost_slope(network, data, labels, costs, ms)
        params = (k, costs[-1], ms[-1])
        print('Iter {}  Cost {:.6f}  Slope {:.6f}'.format(*params))
    return costs, ms

def check_cost_acc(network, data, labels, costs, ms, k=1, k_max=1):
    """ """
    if k % k_max == 0:
        costs, ms = get_cost_slope(network, data, labels, costs, ms)
        acc = network.get_accuracy(data, labels)
        params = (k, costs[-1], ms[-1], acc)
        print('Iter {}  Cost {:.6f}  Slope {:.6f}  Acc {:.6f}'.format(*params))
    return costs, ms


### HELPER ###

def subdata_me(data, h, w, do_subdata):
    """ """
    if not do_subdata:
        return data
    return DT.subdata(data, h, w)

def get_subset(data, n_subset, randomize, with_idxs=False):
    """ subdata of first n data points or random data points """
    if randomize:
        idxs = random.sample(range(len(data)), n_subset)
    else:
        idxs = list(range(n_subset))
    subset = np.array([data[i] for i in idxs])
    if not with_idxs:
        return subset
    return subset, idxs

def get_cost_slope(network, data, labels, costs, ms, max_len=30):
    """ returns updates cost list and calculate slope """
    if len(costs) == 0:
        return [network.get_cost(data, labels)], [0]
    costs.append(network.get_cost(data, labels))   
    costs = costs[1:] if len(costs) > max_len else costs
    avg, _ = np.polyfit(range(len(costs)), costs, 1)
    ms.append(avg)
    return costs, ms



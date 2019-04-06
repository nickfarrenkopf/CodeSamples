import os
import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.NeuralNetworks import TrainUtils as TU


### TRAIN REGRESSION ###

def train_data_iter(network, data, outs, n_train=10, alpha=1e-3, kmax_cost=10):
    """ iterate through data set to train CLASS newtork """
    costs, ms = [], []
    for k in range(n_train):
        network.train_network(data, outs, alpha)
        costs, ms = TU.check_cost(network, data, outs, costs, ms, k, kmax_cost)



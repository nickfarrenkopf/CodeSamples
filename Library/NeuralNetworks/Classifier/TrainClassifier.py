import os
import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT
from Library.NeuralNetworks import TrainUtils as TU


### TRAIN CLASS ###

def train_data_iter(auto_network, network, data, ls, h, w, n_train=100,
                    alpha=1e-3, kmax_cost=10, do_subdata=True):
    """ iterate through data set to train CLASS newtork """
    costs, ms = [], []
    for k in range(n_train):
        ds = TU.subdata_me(data, h, w, do_subdata)
        ds = np.reshape(auto_network.get_flat(ds), (-1, auto_network.flat_size))
        network.train_network(ds, labels, alpha)
        costs, ms = TU.check_cost_acc(network, ds, ls, costs, ms, k, kmax_cost)

def train_path_iter(network, auto_network, embed_network, filepaths, labels,
                    h, w, a=1e-3,
                    batch=16, do_subdata=True, n_train=100, kmax_cost=10):
    """ """
    k, costs, ms = 0, [], []
    while k < n_train:
        fs, idxs = TU.get_subset(filepaths, batch, False, with_idxs=True)
        ds = TU.subdata_me(FT.load_images(fs), h, w, do_subdata)
        ds = np.reshape(auto_network.get_flat(ds), (-1, auto_network.flat_size))
        ds = embed_network.get_encoded(ds)
        ls = [labels[i] for i in idxs]
        network.train_network(ds, ls, a)
        costs, ms = TU.check_cost_acc(network, ds, ls, costs, ms, k, kmax_cost)
        k += 1



def check_done_training(ms):
    """ TODO """
    return len(ms) > 20 and np.mean(np.abs(ms[-10:])) < 1e-3

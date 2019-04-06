import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks import TrainUtils as TU
from Library.NeuralNetworks.Autoencoder import TestAutoencoder as TA


### TRAIN AUTO ###

def train_data_iter(network, data, h, w, a=1e-3, n_train=100, do_subdata=False,
                    n_plot=16, plot_i=False, plot_r=False, b=1.0,
                    kmax_cost=10, kmax_img=10):
    """ iterate dataset to train newtork """
    costs = []
    for k in range(n_train):
        ds = TU.subdata_me(data, h, w, do_subdata)
        network.train_network(ds, a, b)
        costs = check_auto_cost(network, ds, costs, k, kmax_cost)
        check_auto_plot(network, ds, n_plot, plot_i, plot_r, k, kmax_img)

def train_path_iter(network, files, h, w, n_train=100, a=1e-3, do_subdata=True,
                    n_plot=16,  plot_i=False, plot_r=False,
                    kmax_cost=10, kmax_img=20):
    """ iterate filepaths to train newtork """
    costs, ms = [], []
    ds = TU.subdata_me(FT.load_images(DT.get_subset(files, n_plot*4, True)),
                       h, w, do_subdata)
    for k in range(n_train):
        ds = np.concatenate((TU.subdata_me(FT.load_images(DT.get_subset(files,
                                                                       n_plot*4, True))
                                          ,h, w, do_subdata), ds[:n_plot*4]))
        network.train_network(ds, a)
        costs = check_auto_cost(network, ds, costs, k, kmax_cost)
        check_auto_plot(network, ds, n_plot, plot_i, plot_r, k, kmax_img)
        

### HELPER ###

def check_auto_cost(network, ds, costs, k=1, k_max=1, max_len=30):
    """ run every n iterations to check network cost  """
    if k % k_max == 0:
        costs.append([network.get_loss(ds), network.get_io_loss(ds),
                      network.get_kl_loss(ds)])
        if len(costs) > max_len:
            costs.pop(0)
        params = ([k] + costs[-1])
        print('Iter {}  Loss {:.5f}  IO {:.5f}  KL {:.5f}'.format(*params))
    return costs

def check_auto_plot(network, data, n_plot, plot_i, plot_r, k=1, k_max=1):
    """ for every n iterations, plot data before/after/middle data """
    if k_max != 0 and k % k_max == 0:
        if plot_i:
            TA.plot_middle_train(network, data, n_plot, False, count=k)
        if plot_r:
            TA.plot_middle_train(network, data, n_plot, True, count=k)



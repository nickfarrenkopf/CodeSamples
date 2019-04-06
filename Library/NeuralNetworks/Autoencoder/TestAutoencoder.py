import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from threading import Thread

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks import TrainUtils as TU


### TRAINING ###

def plot_middle_train(network, data, n_plot, randomize, count=None, n_x=4,
                      n_y=9):
    """ """
    label = 'plt_rdm' if randomize else 'plt_mid'
    plot_data = get_plot_data_train(network, data, n_plot, randomize)
    path = '{}_{}'.format(label, count) if count or count == 0 else False
    DT.plot_data_multiple(plot_data, n_x=n_x, n_y=n_y, save_path=path)

def get_plot_data_train(network, data, n_plot, randomize):
    """ generate output and middle data """
    # get all data types
    input_data = TU.get_subset(data, n_plot, randomize)
    output_data = np.clip(network.get_outputs(input_data), 0, 1)
    middle_data = np.reshape(network.get_latent(input_data),
                             network.latent_plot_shape)
    # combine data
    plot_data = [[input_data[i], output_data[i], middle_data[i]]
                 for i in range(len(input_data))]
    plot_data = list(itertools.chain.from_iterable(plot_data))
    return plot_data

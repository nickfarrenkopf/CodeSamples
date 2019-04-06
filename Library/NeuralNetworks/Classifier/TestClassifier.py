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

def plot_middle_train(network, data, n_plot, randomize, count=None, n_x=2,
                      n_y=6):
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
    middle_data = np.reshape(network.get_flat(input_data), network.plot_shape)
    # combine data
    plot_data = [[input_data[i], output_data[i], middle_data[i]]
                 for i in range(len(input_data))]
    plot_data = list(itertools.chain.from_iterable(plot_data))
    return plot_data


### RUNTIME ###


def print_class_runtime(network, auto_network, embed_network, data_gen):
    """ """
    last_pred = -1
    for i in range(5000):
        ds = np.array(list(data_gen()) * 4)
        flat = np.reshape(auto_network.get_flat(ds)[0], (-1, auto_network.flat_size))
        mid = embed_network.get_encoded(flat)
        pred = network.get_preds(mid)
        if last_pred != pred:
            last_pred = pred
            print(pred)
        time.sleep(0.2)


def plot_middle_runtime_threaded(network, data_gen, n_repeat=1000,
                                 sleep_time=0.1):
    """ """
    t = Thread(target=plot_middle_runtime,
               args=(network, data_gen, n_repeat, sleep_time))
    t.start()



"""

middle layer data slices
average past N images
difference from past N averages
average past ALL images
difference from past ALL averages

"""


def plot_middle_runtime(network, data_gen, n_repeat=1000, sleep_time=0.1):
    """ """
    # set plot
    plt.ion()
    fig = plt.figure()
    # initial data
    plots = []
    ds = get_plot_data_runtime(network, data_gen)
    ds_n = [np.copy(ds)]
    ds_avgs_n = np.copy(ds)
    ds_avgs_all = np.copy(ds)
    # draw initial data
    for i in range(20):
        plots.append(fig.add_subplot(5,4,i+1).imshow(ds[:,:,i%4]))
    time.sleep(sleep_time)
    # repeat until done
    for j in range(n_repeat):
        ds = get_plot_data_runtime(network, data_gen)
        ds_n.append(ds)
        if len(ds_n) > 16:
            sub = ds_n[0]
            ds_n = ds_n[1:17]
        else:
            sub =  np.zeros(ds_n[0].shape)
        ds_avgs_n = np.mean(ds_n, axis=0)
        print(ds_avgs_n.shape)
        #for i in range(4):
            #ds_avgs_n[:,:,i] = np.mean(ds_n[:,:,i], axis=2)
            #print(ds_avgs_n[:,:,i].shape)
            #ds_avgs_n[:,:,i] = ds_avgs_n[:,:,i] + (ds[:,:,i]-sub[:,:,i])/len(ds)
            #ds_avgs_n[:,:,i] = (ds_avgs_n[:,:,i]*len(ds_n)+ds[:,:,i])/(len(ds_n)+1)
        for i in range(4):
            ds_avgs_all[:,:,i] = (ds_avgs_all[:,:,i]*len(ds_n) + ds[:, :, i]) / (len(ds_n) + 1)
        # current middle data
        for i in range(4):
            plots[i].set_data(ds[:, :, i])
        # partial average data
        for i in range(4, 8):
            plots[i].set_data(ds_avgs_n[:, :, i % 4])
        avgs = np.abs(ds_avgs_n - ds)
        for i in range(8, 12):
            plots[i].set_data(avgs[:, :, i % 4])
        # full everage data
        for i in range(12, 16):
            plots[i].set_data(ds_avgs_all[:, :, i % 4])
        avgs = np.abs(ds_avgs_all - ds)
        for i in range(16, 20):
            plots[i].set_data(avgs[:, :, i % 4])
        #if 1:
        #    image_plot.set_data(get_plot_data_runtime(network, data_gen))
        #if 0:
        #    ds = np.array(list(data_gen()) * 8)
        #    #print(ds.shape)
        #     image_plot.set_data(get_plot_data_train(network, ds, 4, True)[2])
        # reset plot 
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep_time)
    # close plot
    plt.ioff()
    plt.close()




def get_plot_data_runtime(network, data_gen):
    """ """
    ds = np.array(list(data_gen()) * 4)
    #print(ds.shape)
    mids = network.get_middle(ds)[0]
    #print(mids.shape)
    if 0:
        img = Image.fromarray(mids, 'RGBA')
        img2 = img.convert('RGB')
        ds = np.array(img2)
    else:
        #ds = np.reshape(mids, (16, 16))
        ds = mids[:, :, 0]
        
        ds = np.reshape(ds, (8, 8))
    return mids




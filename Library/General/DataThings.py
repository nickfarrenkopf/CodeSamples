import os
import random
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT

"""
TODO
 - clean up

"""


### IMAGE FILE ###

def load_data_labels(filepaths, randomize=False):
    """ ??? """
    if randomize:
        shuffle(filepaths)
    data = FT.load_images(filepaths)
    label_names = [os.path.basename(f).split('.')[0] for f in filepaths]
    label_names = [ln.split('_')[2] for ln in label_names]
    labels = to_one_hot(label_names)
    return data, labels
    

### DATA ###

def get_subset(data, n_subset, randomize):
    """ subdata of first n data points or random data points """
    if randomize:
        idxs = random.sample(range(len(data)), n_subset)
    else:
        idxs = list(range(n_subset))
    return np.array([data[i] for i in idxs])

def subdata(data, height, width):
    """ """
    i = np.random.randint(data.shape[1] - height + 1)
    j = np.random.randint(data.shape[2] - width + 1)
    return data[:, i: i + height, j: j + width, :]

def subdata_xy(data, height, width, x, y):
    """ """
    return data[:, x-height//2:x+height//2, y-width//2:y+width//2, :]

def subdata_points(data, p1, p2):
    """ """
    if len(data.shape) == 4:
        return data[:, p1[0]: p2[1], p1[0]: p2[0], :]
    if len(data.shape) == 3:
        return data[p1[0]: p2[0], p1[1]: p2[1], :]

def pad_me_4d(data, pad1, pad2):
    """ """
    return np.pad(data, ((0, 0), (pad1, pad1), (pad2, pad2), (0, 0)),
                  mode='constant', constant_values=0)


### LABELS ###

def new_label(idx, n_classes):
    """ """
    label = np.zeros(n_classes)
    label[idx] = 1
    return label

def to_one_hot(labels, n_classes=0):
    """ """
    label_set = list(sorted(set(labels)))
    n_classes = len(label_set) if n_classes == 0 else n_classes
    one_hot = [new_label(label_set.index(lab), n_classes) for lab in labels]
    return np.array(one_hot)


### PLOT ###

def plot_data_multiple(data, labels=None, n_x=8, n_y=9, figure_size=(16, 8),
                       save_path=False):
    """ """
    fig = plt.figure(figsize=figure_size)
    # subplots
    for i in range(min(n_x * n_y, len(data))):
        ax = fig.add_subplot(n_x, n_y, i + 1)
        if len(data[i].shape) == 3 and data[i].shape[2] == 1:
            dss = data[i].shape
            ax.imshow(np.reshape(data[i], (dss[0], dss[1])))
        else:
            ax.imshow(data[i])
        ax.set_aspect('equal')
        if labels is not None:
            ax.set_title(labels[i])
        ax.axis('off')
    # other
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig.set_tight_layout(False)
    _ = [fig.savefig(save_path) if save_path else fig.show() for i in range(1)]
    return fig



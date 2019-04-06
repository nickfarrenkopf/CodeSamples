import os
import time
import sklearn
import sklearn.cluster
import sklearn.manifold
import sklearn.decomposition
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gensim.models.word2vec as w2v

import data_things


### WBED ###

def word_check():
    """ """
    return word in vocab


### DATA ###

def decomp(my_data):
    """ """
    pca = sklearn.decomposition.PCA(n_components=20, random_state=0)
    my_data = pca.fit_transform(my_data)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    my_data = tsne.fit_transform(my_data)
    return my_data
    

def random_data():
    """ """
    idx = np.random.randint(len(vocab) - plot_size)
    my_words = vocab[idx:idx + plot_size]
    my_data = np.array([wbed.wv[word] for word in my_words])
    return decomp(my_data), my_words

def word_data(word):
    """ """
    my_words = [word[0] for word in wbed.most_similar(word, topn=plot_size//2)]
    my_words += [word]
    my_data = np.array([wbed.wv[word] for word in my_words])
    return decomp(my_data), my_words


### PLOT ###

def plot_data(my_data):
    """ """
    plt.scatter(my_data.T[0], my_data.T[1], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()

def plot_clusters(my_data, my_words, algorithm, args, kwds, best_word=None):
    """ """
    labels = algorithm(*args, **kwds).fit_predict(my_data)
    palette = sns.color_palette('dark', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    # plot
    plt.scatter(my_data.T[0], my_data.T[1], c=colors, **plot_kwds)
    for i, txt in enumerate(my_words):
        if txt == best_word:
            plt.annotate(txt, (my_data[i, 0], my_data[i, 1]), size=font_size*2)
        else:
            plt.annotate(txt, (my_data[i, 0], my_data[i, 1]), size=font_size)
    # extra
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters - {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.show()

def plot_k():
    """ """
    my_data, my_words = random_data()
    plot_clusters(my_data, my_wordscluster.KMeans, (), {'n_clusters':6})

def plot_affinity():
    """ """
    my_data, my_words = random_data()
    plot_clusters(my_data, my_words, sklearn.cluster.AffinityPropagation, (),
                  {'preference':-5.0, 'damping':0.95})

def plot_aff(word):
    """ """
    my_data, my_words = word_data(word)
    plot_clusters(my_data, my_words, sklearn.cluster.AffinityPropagation, (),
                  {'preference':-5.0, 'damping':0.95}, best_word=word)


### PARAMS ###

# location
base_path = os.path.dirname(os.path.realpath(__file__))
network_path = os.path.join(base_path, 'models')

# plot
font_size = 10
plot_size = 1000
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

# wbed 
wbed = data_things.load_embedding(network_path, 'test')
vocab = [k for k, v in wbed.wv.vocab.items()]


### PROGRAM ###

#data = random_data()



plot_aff('red')






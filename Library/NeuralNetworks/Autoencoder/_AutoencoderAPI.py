import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks.Autoencoder import CreateAutoencoder
from Library.NeuralNetworks.Autoencoder import NetworkAutoencoder
from Library.NeuralNetworks.Autoencoder import TrainAutoencoder
from Library.NeuralNetworks.Autoencoder import TestAutoencoder as TEST


new = CreateAutoencoder.create
load = NetworkAutoencoder.AutoencoderNetwork

train_data_iter = TrainAutoencoder.train_data_iter
#train_data_full = TrainAutoencoder.train_data_full
train_path_iter = TrainAutoencoder.train_path_iter
#train_path_full = TrainAutoencoder.train_path_full

 

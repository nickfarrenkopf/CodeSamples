import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks.Classifier import CreateClassifier
from Library.NeuralNetworks.Classifier import NetworkClassifier
from Library.NeuralNetworks.Classifier import TrainClassifier
from Library.NeuralNetworks.Classifier import TestClassifier as TEST


new = CreateClassifier.create
load = NetworkClassifier.ClassifierNetwork

train_data_iter = TrainClassifier.train_data_iter
train_path_iter = TrainClassifier.train_path_iter



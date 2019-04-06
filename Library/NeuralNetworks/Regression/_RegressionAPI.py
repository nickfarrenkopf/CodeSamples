import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks.Regression import CreateRegression
from Library.NeuralNetworks.Regression import NetworkRegression
from Library.NeuralNetworks.Regression import TrainRegression


new = CreateRegression.create
load = NetworkRegression.RegressionNetwork

train_data_iter = TrainRegression.train_data_iter



import os
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks import NetworkUtils as NU


class RegressionNetwork(NU.TensorflowNetwork):
    """ """

    def __init__(self, name, json_data):
        """ initializes network and load extras """
        self.read_json(json_data['network']['reg'][name])
        NU.TensorflowNetwork.__init__(self, self.save_path)
        self.load_extra_layers()


    ### FILE ###

    def read_json(self, json_data):
        """ read saved json information about network """
        self.full_name = json_data['full_name']
        self.save_path = json_data['filepath']
        self.input_size = json_data['input_size']
        self.input_shape = json_data['input_shape']
        self.layers = json_data['layers']
        self.n_output = json_data['n_output']
        self.epislon = json_data['epsilon']
        self.alpha = json_data['alpha']
        self.create_time = json_data['create_time']
        self.trained = json_data['trained']
        
    def load_extra_layers(self):
        """ load additional network layers """
        with self.sess.as_default():
            with self.graph.as_default():
                get_tensor = self.graph.get_tensor_by_name
                self.logits = get_tensor('{}/logits:0'.format(self.name))


    ### LAYERS ###

    def get_logits(self, input_data):
        """ get predicted output values """
        feed = {self.inputs: input_data}
        return self.sess.run(self.logits, feed)

    def get_cost(self, input_data, output_data):
        """ return cost of network given feed """
        feed = {self.inputs: input_data, self.outputs: output_data}
        return self.sess.run(self.cost, feed)

    def train_network(self, input_data, output_data, alpha):
        """ train network given feed """
        feed = {self.inputs: input_data, self.outputs: output_data,
                self.alpha: alpha}
        return self.sess.run(self.train, feed)



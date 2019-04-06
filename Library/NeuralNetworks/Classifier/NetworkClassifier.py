import os
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks import NetworkUtils as NU


class ClassifierNetwork(NU.TensorflowNetwork):
    """ """

    def __init__(self, name, json_data):
        """ initializes network and load extras """
        self.read_json(json_data['network']['class'][name])
        NU.TensorflowNetwork.__init__(self, self.filepath)
        self.load_extra_layers()


    ### FILE ###

    def read_json(self, json_data):
        """ read saved json information about network """
        self.full_name = json_data['full_name']
        self.filepath = json_data['filepath']
        self.input_size = json_data['input_size']
        self.input_shape = json_data['input_shape']
        self.layers = json_data['layers']
        self.n_classes = json_data['n_classes']
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
                self.preds = get_tensor('{}/preds:0'.format(self.name))
                self.probs = get_tensor('{}/probs:0'.format(self.name))
                self.correct = get_tensor('{}/correct:0'.format(self.name))
                self.accuracy = get_tensor('{}/accuracy:0'.format(self.name))


    ### LAYERS ###

    def get_logits(self, input_data):
        """ """
        feed = {self.inputs: input_data}
        return self.sess.run(self.logits, feed)

    def get_preds(self, input_data):
        """ """
        feed = {self.inputs: input_data}
        return self.sess.run(self.preds, feed)

    def get_probs(self, input_data, output_data):
        """ """
        feed = {self.inputs: input_data, self.outputs: output_data}
        return self.sess.run(self.probs, feed)

    def get_correct(self, input_data, output_data):
        """ """
        feed = {self.inputs: input_data, self.outputs: output_data}
        return self.sess.run(self.correct, feed)

    def get_accuracy(self, input_data, output_data):
        """ """
        feed = {self.inputs: input_data, self.outputs: output_data}
        return self.sess.run(self.accuracy, feed)

    def get_cost(self, input_data, output_data):
        """ """
        feed = {self.inputs: input_data, self.outputs: output_data}
        return self.sess.run(self.cost, feed)

    def train_network(self, input_data, output_data, alpha):
        """ """
        feed = {self.inputs: input_data, self.outputs: output_data,
                self.alpha: alpha}
        return self.sess.run(self.train, feed)


    ### HELPER ###

    def print_info(self):
        """ """
        print('Classifier: {}'.format(self.name))
        print(' - input shape: {}'.format(self.input_shape))
        print(' - n_classes: {}'.format(self.n_classes))
        print(' - n_layers: {}'.format(len(self.layers)))
        

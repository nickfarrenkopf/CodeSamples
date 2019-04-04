import os
import itertools
import numpy as np

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks import NetworkUtils as NU


class AutoencoderNetwork(NU.TensorflowNetwork):
    """ Convolutional Autoencoder Network """

    def __init__(self, name, json_data):
        """ initializes network and load extras """
        self.read_json(json_data['network']['auto'][name])
        NU.TensorflowNetwork.__init__(self, self.filepath)
        self.load_extra_layers()


    ### FILE ###

    def read_json(self, json_data):
        """ read saved json information about network """
        # file
        self.filename = json_data['filename']
        self.filepath = json_data['filepath']
        self.create_time = json_data['create_time']
        # encoder
        self.input_shape = json_data['input_shape']
        self.hidden_encoder = json_data['hidden_encoder']
        self.pools_encoder = json_data['pools_encoder']
        self.act_encoder = json_data['activation_encoder']
        self.encoded_shape = json_data['encoded_shape']
        # latent
        self.hidden_pre_latent = json_data['hidden_latent']
        self.latent_shape = json_data['latent_shape']
        self.act_post_latent = json_data['activation_post_latent']
        # decoder
        self.act_decoder = json_data['activation_decoder']
        self.act_output = json_data['activation_output']

    def load_extra_layers(self):
        """ load additional network layers """
        with self.sess.as_default():
            with self.graph.as_default():
                get_tensor = self.graph.get_tensor_by_name
                self.beta = get_tensor('{}/beta:0'.format(self.name))
                self.mu = get_tensor('{}/mu:0'.format(self.name))
                self.sigma = get_tensor('{}/sigma:0'.format(self.name))
                self.latent = get_tensor('{}/latent:0'.format(self.name))
                self.io_loss = get_tensor('{}/io_loss:0'.format(self.name))
                self.kl_loss = get_tensor('{}/kl_loss:0'.format(self.name))   
        self.latent_plot_shape = self.get_plot_shape(self.latent_shape[-1])
    

    ### LAYERS ###

    def run_sess(self, operation, data, alpha, beta):
        """ """
        feed = {self.inputs: data, self.alpha: alpha, self.beta: beta}
        return self.sess.run(operation, feed)

    def get_mu(self, data):
        """ return variational mean layer """
        return self.run_sess(self.mu, data, 0, 1)

    def get_sigma(self, data):
        """ return variational standard deviation layer """
        return self.run_sess(self.sigma, data, 0, 1)
    
    def get_latent(self, data):
        """ return variational latent layer """
        return self.run_sess(self.latent, data, 0, 1)

    def get_outputs(self, data):
        """ return generated outputs """
        return self.run_sess(self.outputs, data, 0, 1)
             
    def get_io_loss(self, data):
        """ return input-output loss """
        return np.mean(self.run_sess(self.io_loss, data, 0, 1))

    def get_kl_loss(self, data, beta=1.0):
        """ return KL divergence """
        return np.mean(self.run_sess(self.kl_loss, data, 0, 1))

    def get_loss(self, data, beta=1.0):
        """ return cost of network """
        return self.run_sess(self.loss, data, 0, beta)

    def train_network(self, data, alpha=1e-3, beta=1.0):
        """ train network given on input data """
        self.run_sess(self.train, data, alpha, beta)


    ### HELPER ###

    def get_plot_shape(self, flat):
        """ perfect square vs non perfect square plot shape """
        s = int(np.sqrt(flat))
        if s * s == int(flat):
            return (-1, s, s)
        s = int(np.sqrt(flat / 2))
        return (-1, s, 2 * s)



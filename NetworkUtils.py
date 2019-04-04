import os
import tensorflow as tf


class TensorflowNetwork(object):
    """ base network structure """

    def __init__(self, filepath):
        """ main tf network params - name, path, session, and graph """
        self.save_path = filepath
        self.name = os.path.split(filepath)[1]
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True),
                               graph=self.graph)
        self.load_main()


    ### FILE ###

    def load_main(self):
        """ load main parts of network """
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(self.save_path + '.meta')
                saver.restore(self.sess, self.save_path)
                get_tensor = self.graph.get_tensor_by_name
                self.inputs = get_tensor('{}/inputs:0'.format(self.name))
                self.outputs = get_tensor('{}/outputs:0'.format(self.name))
                self.alpha = get_tensor('{}/alpha:0'.format(self.name))
                self.loss = get_tensor('{}/loss:0'.format(self.name))
                self.train = tf.get_collection('{}_train'.format(self.name))[0]

    def save_network(self, step=0):
        """ saves network to save path """
        print('Saving network {}'.format(self.name))
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                if step == 0:
                    saver.save(self.sess, self.save_path)
                else:
                    saver.save(self.sess, self.save_path, global_step=step)



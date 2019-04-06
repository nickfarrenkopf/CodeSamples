import os
import time
import numpy as np
import tensorflow as tf

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks import CreateUtils as CU


### CREATE ###

def create(paths, name, size, hidden, n_output, e=1e-8, print_me=True):
    """ create multi class classification network """

    with tf.device('/gpu:0'):

        # shapes
        n_layers = len(hidden) 
        input_shape = (None, size)
        output_shape = (None, n_output)

        # name and save path
        reg_name = 'REG_{}_{}_{}_{}'.format(name, size, n_layers, n_output)
        save_path = os.path.join(paths.network_path, reg_name)
        print(save_path)
        
        # set variable scope
        with tf.variable_scope(reg_name):

            # initial params
            print('Creating regression network...')
            start = time.time()

            # placeholders
            inputs = tf.placeholder(tf.float32, input_shape, name='inputs')
            outputs = tf.placeholder(tf.float32, output_shape, name='outputs')
            alpha = tf.placeholder(tf.float32, name='alpha')
            CU.print_line(' - input shape: {}'.format(inputs.shape), print_me)
            
            # hidden layers
            current = inputs
            for i in range(len(hidden)):
                
                # weights, bias, and output
                W = CU.weight([int(current.shape[1]), hidden[i]])
                b = CU.bias([hidden[i]])
                output = tf.nn.relu(tf.add(tf.matmul(current, W), b))
                current = output
                CU.print_line(' - feedforward: {}'.format(current.shape),
                              print_me)

            # final layer
            W_f = CU.weight([int(current.shape[1]), n_output])
            b_f = CU.bias([n_output])
            logits = tf.add(tf.matmul(current, W_f), b_f, name='logits')
            CU.print_line(' - final: {}'.format(logits.shape), print_me)
            
            # training
            loss = tf.reduce_mean(tf.square(outputs - logits), name='cost')
            optimizer = tf.train.AdamOptimizer(alpha, epsilon=e).minimize(loss)
            tf.add_to_collection('{}_train'.format(reg_name), optimizer)

            # load session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            # save network
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            timed = time.time() - start
            CU.print_line('Created {} network in {}\n'.format(reg_name, timed),
                          print_me)

    # write to json
    data = paths.load_json()
    data['network']['reg'].update({name: {}})
    data['network']['reg'][name] = {'full_name': reg_name,
                                    'filepath': save_path,
                                    'input_size': size,
                                    'input_shape': input_shape,
                                    'layers': hidden,
                                    'n_output': n_output,
                                    'epsilon': e,
                                    'alpha': 0,
                                    'create_time': timed,
                                    'trained': 0}
    paths.write_json(data)



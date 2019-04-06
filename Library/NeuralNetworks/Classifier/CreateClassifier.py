import os
import time
import numpy as np
import tensorflow as tf

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks import CreateUtils as CU


### CREATE ###

def create(paths, name, size, hidden, n_classes, e=1e-8, print_me=True):
    """ create multi class classification network """

    with tf.device('/gpu:0'):

        # shapes
        n_layers = len(hidden) 
        input_shape = (None, size)
        output_shape = (None, n_classes)

        # name and save path
        class_name = 'CLASS_{}_{}_{}_{}'.format(name, size, n_layers, n_classes)
        save_path = os.path.join(paths.network_path, class_name)
        
        # set variable scope
        with tf.variable_scope(class_name):

            # initial params
            print('Creating classification network...')
            start = time.time()

            # placeholders
            inputs = tf.placeholder(tf.float32, input_shape, name='inputs')
            outputs = tf.placeholder(tf.float32, output_shape, name='outputs')
            alpha = tf.placeholder(tf.float32, name='alpha')
            CU.msg(' - input shape: {}'.format(inputs.shape), print_me)
            
            # hidden layers
            current = inputs
            for i in range(len(hidden)):
                
                # weight and bias
                W = CU.weight([int(current.shape[1]), hidden[i]])
                b = CU.bias([hidden[i]])

                # output
                output = tf.nn.relu(tf.add(tf.matmul(current, W), b))
                current = output
                CU.msg(' - feedforward: {}'.format(current.shape), print_me)

            # final layer
            W_f = CU.weight([int(current.shape[1]), n_classes])
            b_f = CU.bias([n_classes])
            logits = tf.add(tf.matmul(current, W_f), b_f, name='logits')
            CU.msg(' - final: {}'.format(logits.shape), print_me)

            # metrics
            preds = tf.argmax(logits, axis=1, name='preds')
            probs = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=outputs,
                                                            name='probs')
            correct = tf.equal(preds, tf.argmax(outputs, axis=1),
                               name='correct')
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),
                                      name='accuracy')

            # training
            loss = tf.reduce_mean(probs, name='cost')
            optimizer = tf.train.AdamOptimizer(alpha, epsilon=e).minimize(loss)
            tf.add_to_collection('{}_train'.format(class_name), optimizer)

            # load session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            # save network
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            timed = time.time() - start
            CU.msg('Created {} network in {}\n'.format(class_name, timed),
                          print_me)

    # write to json
    data = paths.load_json()
    data['network']['class'].update({name: {}})
    data['network']['class'][name] = {'full_name': class_name,
                                      'filepath': save_path,
                                      'input_size': size,
                                      'input_shape': input_shape,
                                      'layers': hidden,
                                      'n_classes': n_classes,
                                      'epsilon': e,
                                      'alpha': 0,
                                      'create_time': timed,
                                      'trained': 0}
    paths.write_json(data)



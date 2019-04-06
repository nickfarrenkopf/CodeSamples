import os
import time
import json
import tensorflow as tf

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks import CreateUtils as CU


"""
TODO
 - implement beta

"""

### HELPER ###

conv2d = tf.layers.conv2d
deconv2d = tf.layers.conv2d_transpose

relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid


### CREATE ###

def create(paths, name, h, w, length=1, patch=4, pad ='SAME', print_me=True,
           hidden_encode=[64,64,64], pools_encode=[2,2,1], act_enc=lrelu,
           hidden_latent=[24], n_latent=8, act_d1=relu, act_dec=relu,
           act_d2=sigmoid, pools_decode=[1,1,1], hidden_decode=[32,32,32]):
    """ create convolutional autoencoder network """

    # shapes
    input_shape = (None, h, w, length)
    input_shape_full = (-1, h, w, length)  
    flat_size =  h * w * length
    flat_shape = (-1, flat_size)
    
    # file info
    params = (name,h,w,length,len(hidden_encode),len(hidden_latent),n_latent)
    filename = 'AUTO' + ('_{}' * len(params)).format(*params)
    filepath = os.path.join(paths.network_path, filename)

    # initial
    print('Creating autoencoder network...')
    print(' - {}'.format(filepath))
    start = time.time()

    # use GPU and set variable scope
    with tf.device('/gpu:0'): 
        with tf.variable_scope(filename):           
            
            # placeholders
            inputs = tf.placeholder(tf.float32, input_shape, name='inputs')
            alpha = tf.placeholder(tf.float32, name='alpha')
            beta = tf.placeholder(tf.float32, name='beta')
            CU.msg(' - input shape: {}'.format(inputs.shape), print_me)

            # encoder
            current = inputs
            for h, p in zip(hidden_encode, pools_encode):
                current = conv2d(current, h, patch, p, pad, activation=lrelu)
                CU.msg(' - encode: {}'.format(current.shape), print_me)
            enc_shp = current.shape.as_list()
            encode_shape = (-1, enc_shp[1], enc_shp[2], 1)
            encode_size_flat = enc_shp[1] * enc_shp[2] * 1
            
            # flatten
            current = tf.contrib.layers.flatten(current)
            flat_encode_size = current.shape.as_list()[-1]
            CU.msg(' - flat: {}'.format(current.shape), print_me)
            
            # variational
            MU = tf.identity(CU.dense(current, n_latent, None), name='mu')
            SD = tf.identity(0.5*CU.dense(current,n_latent,None),name='sigma')
            EPSILON = tf.random_normal(tf.stack([tf.shape(MU)[0], n_latent]))
            Z = MU + tf.multiply(EPSILON, tf.exp(SD))

            # latent
            current = tf.identity(Z, name='latent')
            latent_shape = current.shape.as_list()
            CU.msg(' - latent: {}'.format(current.shape), print_me)

            # dense layers
            hidden_latent.append(encode_size_flat)
            for h in hidden_latent:
                current = CU.dense(current, h, act_d1)
                CU.msg(' - dense: {}'.format(current.shape), print_me)

            # unflatten
            current = tf.reshape(current, encode_shape)
            CU.msg(' - reshaped: {}'.format(current.shape), print_me)

            # decoder
            #hidden_encode.reverse()
            hidden_decode.append(length)
            #pools_encode.reverse()
            #pools_encode.append(1)
            for h, p in zip(hidden_decode, pools_decode):
                #act = relu if h == hidden_encode else sigmoid
                act = relu
                current = deconv2d(current, h, patch, p, pad, activation=relu)
                CU.msg(' - decode: {}'.format(current.shape), print_me)

            # dense layers
            current = tf.contrib.layers.flatten(current)
            #print(flat_size)
            #a = cc
            current = CU.dense(current, flat_size, act_d2)
            outputs = tf.reshape(current, input_shape_full, name='outputs')
            CU.msg(' - dense: {}'.format(current.shape), print_me)
            CU.msg(' - output: {}'.format(current.shape), print_me)

            # standard metrics
            inputs_flat = tf.reshape(inputs, flat_shape)
            IO_cost = tf.squared_difference(current, inputs_flat)
            IO_loss = tf.reduce_sum(IO_cost, 1, name='io_loss')

            # variational metrics
            KL_cost = 1.0 + 2.0 * SD - tf.square(MU) - tf.exp(2.0 * SD)
            KL_loss = tf.identity(-0.5*1.1*tf.reduce_sum(KL_cost,1), name='kl_loss')

            # optimizer
            loss = tf.reduce_mean(IO_loss + KL_loss, name='loss')
            optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)
            tf.add_to_collection('{}_train'.format(filename), optimizer)

            # load GPU session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
            sess.run(tf.global_variables_initializer())

            # save network
            saver = tf.train.Saver()
            saver.save(sess, filepath)
            create_time = time.time() - start
            params = (filename, create_time)
            CU.msg('Created {} network in {}\n'.format(*params), print_me)

    # write to json
    data = paths.load_json()
    data['network']['auto'].update({name: {}})
    data['network']['auto'][name] = {'filename': filename,
                                     'filepath': filepath,
                                     'create_time': create_time,
                                     
                                     'input_shape': input_shape,
                                     'hidden_encoder': hidden_encode,
                                     'pools_encoder': pools_encode,
                                     'activation_encoder': act_enc.__name__,
                                     'encoded_shape': encode_shape,
                                     
                                     'hidden_latent': hidden_latent,
                                     'latent_shape': latent_shape,
                                     'activation_post_latent': act_d1.__name__,

                                     'activation_decoder': act_dec.__name__,
                                     'activation_output': act_d2.__name__}
    paths.write_json(data)



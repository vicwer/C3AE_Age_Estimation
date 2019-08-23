#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import sys
sys.path.append('..')
import numpy as np
from config import cfg
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

PRINT_LAYER_LOG = cfg.PRINT_LAYER_LOG

def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6.

def network_arg_scope(
        is_training=True, weight_decay=cfg.train.weight_decay, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=False):
    batch_norm_params = {
        'is_training': is_training, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        #'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        'trainable': cfg.train.bn_training,
    }

    with slim.arg_scope(
            [slim.conv2d, slim.separable_convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=h_swish,
            #activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            padding='valid'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class Network(object):
    def __init__(self):
        pass

    def inference(self, mode, inputs, scope='C3AE'):
        is_training = mode
        with slim.arg_scope(network_arg_scope(is_training=is_training)):
            with tf.variable_scope(scope, reuse=False):
                conv1 = se_module(inputs, 32, 1, name='conv_1')
                avg1 = avg_pool(conv1, name='avg_1')
                conv2 = se_module(avg1, 32, 1, name='conv_2')
                avg2 = avg_pool(conv2, name='avg_2')
                conv3 = se_module(avg2, 32, 1, name='conv_3')
                avg3 = avg_pool(conv3, name='avg_3')
                conv4 = se_module(avg3, 32, 1, name='conv_4')
                conv5 = slim.conv2d(conv4, 32, [1,1], 1, scope='conv_5')
                print(conv5.name, conv5.get_shape())
                # avg5 = tf.reduce_mean(conv5, [1, 2], keepdims=True, name='avg_pool')
                # print(avg5.name, avg5.get_shape())
                concat = reshape(conv5, [-1, avg5.get_shape()[3] * 3], 'reshape')
                feats = fully_connected(concat, 12, name='fc_1')
                pred = fully_connected(feats, 1, name='fc_2')
                if is_training:
                    print(tf.losses.get_regularization_losses()[-2])
                    feats_l1_loss = tf.add_n([tf.losses.get_regularization_losses()[-2]])
                    return feats, pred, feats_l1_loss
                else:
                    return feats, pred

def se_module(inputs, c_outputs, s, name):
    output = conv2d(inputs, c_outputs, s, name)
    if cfg.use_se_module:
        global_pooling = tf.reduce_mean(output, [1, 2], keepdims=True, name=name+'_avg_pool')
        fc1 = pw_conv(global_pooling, int(output.get_shape()[-1] / 2), name+'_fc1')
        fc2 = pw_conv(fc1, c_outputs, name+'_fc2')
        sigmoid = tf.sigmoid(fc2)
        output = output * sigmoid
    return output

def reshape(inputs, shape, name):
    output = tf.reshape(inputs, shape=shape, name=name)
    print(name, output.get_shape())
    return output

def avg_pool(inputs, name):
    output = slim.avg_pool2d(inputs,
                             kernel_size=[2, 2],
                             stride=2,
                             scope=name)
    print(name, output.get_shape())
    return output

def conv2d(inputs, c_outputs, s, name):
    output = slim.conv2d(inputs,
                         num_outputs=c_outputs,
                         kernel_size=[3,3],
                         stride=s,
                         scope=name)
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def maxpool2x2(input, name):
    output = slim.max_pool2d(input, kernel_size=[2, 2], stride=2, scope=name)
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def fully_connected(input, c_outputs, name):
    output = slim.fully_connected(input,
                                  c_outputs,
                                  weights_regularizer=slim.l1_regularizer(0.00001),
                                  weights_initializer=slim.variance_scaling_initializer(),
                                  normalizer_fn=None,
                                  activation_fn=None,
                                  scope=name)
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def d_p_conv(inputs, c_outputs, s, name):
    output = slim.separable_convolution2d(inputs,
                                          num_outputs=None,
                                          stride=s,
                                          depth_multiplier=1,
                                          kernel_size=[3, 3],
                                          normalizer_fn=slim.batch_norm,
                                          scope=name+'_d_conv')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())

    output = slim.conv2d(output,
                         num_outputs=c_outputs,
                         kernel_size=[1,1],
                         stride=1,
                         scope=name+'_p_conv')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())

    return output

def dw_conv(inputs, s, name):
    output = slim.separable_convolution2d(inputs,
                                          num_outputs=None,
                                          stride=s,
                                          depth_multiplier=1,
                                          kernel_size=[3, 3],
                                          normalizer_fn=slim.batch_norm,
                                          scope=name+'_ds_conv')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def pw_conv(inputs, c_outputs, name):
    output = slim.conv2d(inputs,
                         num_outputs=c_outputs,
                         kernel_size=[1,1],
                         stride=1,
                         scope=name+'_p_conv')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

def route(input_list, name):
    with tf.name_scope(name):
        output = tf.concat(input_list, 3, name='concat')
    if PRINT_LAYER_LOG:
        print(name, output.get_shape())
    return output

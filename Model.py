# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import tensorflow as tf
WEIGHT_DECAY = 0.0 #0.000001 #0.001
WEIGHT_STDDEV = 0.01  #0.001

# Set seed number #
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

# Set normalization #
xavier = tf.keras.initializers.RandomUniform(-0.1, 0.1, seed=42)


def _get_variable(name, shape, initializer, weight_decay, dtype=tf.float32, trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)


def fc(x, num_classes):
    num_units_in = x.get_shape().as_list()[1]
    num_units_out = num_classes
    shape = [num_units_in, num_units_out]
    weight = _get_variable('weight', shape=shape, dtype=tf.float32, initializer=xavier, weight_decay=WEIGHT_DECAY)
    bias = _get_variable('bias', shape=[num_units_out],  initializer=tf.zeros_initializer(), weight_decay=WEIGHT_DECAY)
    return tf.nn.xw_plus_b(x, weight, bias)


def create_initial_state(batch_size, state_size):
    list_initial_state = []
    for ii, size in enumerate(state_size):
        get_initial_state1 = _get_variable('initial_state'+str(ii), shape=[batch_size, size], initializer=xavier, weight_decay=WEIGHT_DECAY, dtype=tf.float32)
        list_initial_state.append(get_initial_state1)
    return tuple(list_initial_state)


class LSTM:
    def __init__(self, input1, input2, state_size, day, dlen, outlen, dtype=tf.float32):
        self.x1 = input1
        self.x2 = input2
        self.dim1 = self.x1.get_shape().as_list()
        self.dim2 = self.x2.get_shape().as_list()
        self.h_size = state_size
        self.day = day
        self.dlen = dlen
        self.outlen = outlen

        # Weights in FC #
        self.W = _get_variable('W', shape=[self.h_size[-1], 1], initializer=xavier, weight_decay=WEIGHT_DECAY, dtype=tf.float32)
        self.B = _get_variable('B', shape=[1], initializer=tf.zeros_initializer(), weight_decay=WEIGHT_DECAY, dtype=tf.float32)

        # Initial state in Encoder
        self.En_c0 = _get_variable('c_0', shape=[self.dim1[1], self.h_size[0]], initializer=xavier, weight_decay=WEIGHT_DECAY, dtype=tf.float32)
        self.En_c1 = _get_variable('c_1', shape=[self.dim1[1], self.h_size[1]], initializer=xavier, weight_decay=WEIGHT_DECAY, dtype=tf.float32)
        self.En_h0 = _get_variable('h_0', shape=[self.dim1[1], self.h_size[0]], initializer=xavier, weight_decay=WEIGHT_DECAY, dtype=tf.float32)
        self.En_h1 = _get_variable('h_1', shape=[self.dim1[1], self.h_size[1]], initializer=xavier, weight_decay=WEIGHT_DECAY, dtype=tf.float32)

    def predict(self):
        with tf.variable_scope('Encoder'):
            lstm = [tf.contrib.rnn.LSTMCell(size, activation=tf.sin, initializer=xavier) for size in self.h_size]
            cell = tf.contrib.rnn.MultiRNNCell(lstm)
            init = (tf.contrib.rnn.LSTMStateTuple(self.En_c0, self.En_h0), tf.contrib.rnn.LSTMStateTuple(self.En_c1, self.En_h1))
            output1, state1 = tf.nn.dynamic_rnn(cell, self.x1, initial_state=init, time_major=True, dtype=tf.float32)

        with tf.variable_scope('Decoder'):
            lstm2 = [tf.contrib.rnn.LSTMCell(size, activation=tf.sin, initializer=xavier) for size in self.h_size]
            cell2 = tf.contrib.rnn.MultiRNNCell(lstm2)
            output2, state2 = tf.nn.dynamic_rnn(cell2, self.x2, initial_state=state1, time_major=True, dtype=tf.float32)

        if self.day == 'D0': DeOutput = output2[:self.dlen]
        if self.day == 'D1': DeOutput = output2[self.dlen:self.dlen+4]
        if self.day == 'D2': DeOutput = output2[self.dlen+4:]

        logits = []
        for ii in range(self.outlen):
            fc_out = tf.nn.xw_plus_b(DeOutput[ii], self.W, self.B)
            logits.append(fc_out)
        logits = tf.stack(logits)

        return logits


def RMSE(logits, target):
    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits,target)))
    return loss

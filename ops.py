import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)



def weight_variable_glorot(input_dim, output_dim, name = ""):
    """
    Create a weight variable with Glorot&Bengio (AISTATS 2010) initialization
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval = -init_range,
                                maxval = init_range, dtype = tf.float32)
    return tf.Variable(initial, name = name)
    # return tf.get_variable(name, initial, tf.float32)



class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, inputFeatures, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv3d(input_, inputFeatures, output_dim,
           k_d=3, k_h=3, k_w=3, d_d=2, d_h=2, d_w=2, name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))


        return conv


def conv3d_3x3(input_, inputFeatures, output_dim,
           k_d=3, k_h=3, k_w=3, d_d=2, d_h=2, d_w=2, name="conv3d_3x3"):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_d, k_h, k_w, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv2d_3x3(input_, inputFeatures, output_dim,
           k_d=3, k_h=3, d_d=2, d_h=2, name="conv2d_3x3"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_d, d_h, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))


        return conv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer(seed=23))
        # matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.contrib.layers.variance_scaling_initializer(factor=3.6,mode='FAN_IN',
        #                                                                         uniform=False,seed=23))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias
        # return tf.matmul(input_, matrix)



def denseV1(input_, output_size, filter_size, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "denseV"):
        # matrix = tf.get_variable("Matrix", [output_size, shape[1]], tf.float32,
        #                          tf.contrib.layers.xavier_initializer(seed=23),
        #                          regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        # matrix = tf.get_variable("Matrix", [shape[2], output_size, filter_size], tf.float32,
        #                          tf.contrib.layers.xavier_initializer(seed=23))

        # matrix = tf.get_variable("Matrix", [shape[2], output_size], tf.float32,
        #                          tf.contrib.layers.xavier_initializer(uniform=False,seed=23))

        matrix = tf.get_variable("Matrix", [shape[2], output_size], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer(factor=3.6,mode='FAN_IN',
                                                                                uniform=False,seed=23))


        #
        # matrix = tf.get_variable("Matrix", [shape[2], output_size], tf.float32,
        #                          initializer=tf.constant_initializer(0.0))

        return matrix




def denseV(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "denseV"):
        # matrix = tf.get_variable("Matrix", [output_size, shape[1]], tf.float32,
        #                          tf.contrib.layers.xavier_initializer(seed=23),
        #                          regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        matrix = tf.get_variable("Matrix", [output_size, shape[1]], tf.float32,
                                 tf.contrib.layers.xavier_initializer(uniform=False,seed=23))

        # matrix = weight_variable_glorot(output_size, shape[1], "Matrix")

        return matrix

def denseW(input_, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "denseW"):
        # matrix = tf.get_variable("Matrix", [shape[1]], tf.float32,tf.contrib.layers.xavier_initializer(uniform=False,seed=23),
        #                          regularizer = tf.contrib.layers.l2_regularizer(scale=0.01))

        matrix = tf.get_variable("Matrix", [shape[1]], tf.float32, tf.constant_initializer(0.0))

        # matrix = tf.get_variable("Matrix", [shape[1]], tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer(seed=23))

        bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))

        return bias + tf.reduce_sum(tf.multiply(matrix, input_), 1, keepdims=True)

def denseW1(input_, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "denseW"):
        # matrix = tf.get_variable("Matrix", [shape[1]], tf.float32,tf.contrib.layers.xavier_initializer(uniform=False,seed=23),
        #                          regularizer = tf.contrib.layers.l2_regularizer(scale=0.01))

        matrix = tf.get_variable("Matrix", [shape[1]], tf.float32, initializer=tf.constant_initializer(0.1))

        # matrix = tf.get_variable("Matrix", [shape[1]], tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer(seed=23))

        bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))

        matrix = tf.concat([bias,matrix],0)

        augterm = tf.ones(shape=tf.stack([tf.shape(input_)[0], 1]))

        input_ = tf.concat([augterm, input_],1)

        return tf.multiply(matrix, input_)

'''

def fm_fusion(self, inputs_, reuse=False, test=False):
    with tf.variable_scope("fm_fusion") as scope:
        if reuse:
            scope.reuse_variables()

        V = denseV(inputs_, 7, 'Interaction_Vector_T')

        pair_interactions = (0.5 * tf.subtract(
            tf.pow(tf.matmul(inputs_, tf.transpose(V)), 2),
            tf.matmul(tf.pow(inputs_, 2), tf.transpose(tf.pow(V, 2)))))

        if test:
            pair_interactions = self.BN_T(tf.nn.relu(pair_interactions), train=False)
        else:
            pair_interactions = self.BN_T(tf.nn.relu(pair_interactions))

        Bilinear = tf.reduce_sum(pair_interactions, 1, keepdims=True)

        linear_terms = denseW(inputs_, 'linear_terms')

        pred = linear_terms + Bilinear

        return pred



def attention_fusion(input_, name="Fusion_Attn"):

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [1,1], tf.float32, tf.contrib.layers.xavier_initializer(seed=23))
        # matrix = tf.get_variable("Matrix", [1,1], tf.float32, tf.constant_initializer(value=1.0))

        weights_f = tf.get_variable("Weights", [10, 10], tf.float32, tf.contrib.layers.xavier_initializer(seed=23))
        # weights_f = tf.get_variable("Weights", [1, 1], tf.float32, tf.constant_initializer(value=1.0))
        # let's say number of data points is 100
        # input_ is 100 by 7, matrix is scalar, p_f is 100 by 7

        # new_input =  tf.expand_dims(input_,1)
        # print("epanded dimension ---->", new_input.shape)
        # p_f = tf.nn.tanh(tf.einsum('jl,ilk->jk',matrix, new_input))
        # print("p_f shape ---->", p_f.shape)

        new_input = tf.expand_dims(input_, 1)
        print("epanded dimension ---->", new_input.shape)
        p_f = tf.nn.tanh(tf.einsum('jl,ilk->jk', matrix, new_input))
        print("p_f shape ---->", p_f.shape)

        # get weights into right shapec
        # input is 100 by 7, weights_f is 7 by 1, new_weights_f is 100 by 1
        # new_weights_f = tf.einsum('')
        # new_weights_f = tf.scalar_mul('lj,jk->lk',tf.transpose(weights_f), p_f)
        # print("new weights shape ---->", new_weights_f.shape)

        alphas = tf.matmul(weights_f, p_f)

        # transposed new_weights_f is 1x100, p_f is 100 by 7, alphas is 1 by 7
        alphas = tf.nn.softmax(alphas)
        print("alphas shape ---->", alphas.shape)

        # input is 100 by 7, alphas transposed is 7 by 1, the function returns 100 by 1
        return tf.matmul(input_, tf.transpose(alphas))

'''

# def attention(input_, dim, scope=None):
#     assert len(input_.get_shape()) == 3 and input_.get_shape()[-1].value is not None
#     shape = input_.get_shape().as_list()
#
#     with tf.variable_scope(scope or 'attention'):
#       Weight = tf.get_variable('Weight', [shape[-2], dim], tf.float32,
#                              tf.contrib.layers.xavier_initializer(seed=23))
#
#       Alpha = tf.get_variable('Atten_Weight', [dim, 1], tf.float32,
#                                tf.contrib.layers.xavier_initializer(seed=23))
#
#       # bias = tf.get_variable("bias", [shape[2]], initializer=tf.constant_initializer(0.0))
#
#       # projection = tf.nn.tanh(tf.einsum('ij,kjl->kil', Weight, input_)) + bias
#       # projection = tf.nn.tanh(tf.einsum('ij,kjl->kil', Weight, input_))
#       projection = tf.nn.tanh(tf.einsum('ijk,kl->ijl', input_, Weight))
#
#       print("projection --->", projection.shape)
#       print("Alpha --->", Alpha.shape)
#
#       alpha = tf.einsum('ijk,kl->ij1',projection, Alpha)
#       # alpha = tf.einsum('lj,ijk->ilk',Alpha, projection)
#
#       alpha = tf.transpose(alpha, perm=[0,2,1])
#
#       alpha = tf.nn.softmax(alpha)
#
#       print("alpha --->",alpha.shape)
#
#       alpha = tf.transpose(alpha, perm=[0,2,1])
#
#       # outputs = tf.matmul(input_, alpha)
#       # outputs = tf.einsum('ijk,ikl->ijl',input_, alpha)
#       outputs = tf.einsum('ijk,ikl->ijl',tf.transpose(input_, perm=[0,2,1]), alpha)
#
#       return outputs


def attention(input_, dim, scope=None):
    assert len(input_.get_shape()) == 3 and input_.get_shape()[-1].value is not None
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or 'attention'):
      # Weight = tf.get_variable('Weight', [dim, shape[1]], tf.float32,
      #                        tf.contrib.layers.xavier_initializer(seed=23))


      Weight = tf.get_variable('Weight', [dim, shape[1]], tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=23))


      Alpha = tf.get_variable('Atten_Weight', [1, dim], tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=23))

      bias = tf.get_variable("bias", [shape[2]], initializer=tf.constant_initializer(0.0))

      print("bias shape -->", bias.shape)
      print('input shape--->',input_.shape)

      # projection = tf.nn.tanh(tf.einsum('ij,kjl->kil', Weight, input_)) + bias
      projection = tf.einsum('ijk,kl->ijl', tf.transpose(input_, perm=[0,2,1]),tf.transpose(Weight))
      projection = tf.nn.tanh(tf.transpose(projection, perm = [0,2,1]))

      # projection = tf.nn.tanh(tf.matmul(Weight, input_))

      print("projection --->", projection.shape)
      print("Alpha --->", Alpha.shape)

      alpha = tf.einsum('ijk,kl->ijl', tf.transpose(projection, perm = [0,2,1]), tf.transpose(Alpha))
      alpha = tf.transpose(alpha, perm=[0,2,1])


      alpha = tf.layers.flatten(alpha)
      alpha = tf.nn.softmax(alpha)

      print("alpha --->",alpha.shape)

      alpha = tf.expand_dims(alpha, -1)

      # alpha = tf.transpose(alpha, perm=[0,2,1])

      outputs = tf.einsum('ijk,ikl->ijl',input_, alpha)

      return outputs



def conv3d_revised(input_, kernel_size, strides, padding='SAME' ,name="conv3d"):
    with tf.variable_scope(name):
        # w = tf.get_variable('w', kernel_size, initializer=tf.contrib.layers.xavier_initializer(seed=23))
        w = tf.get_variable('w', kernel_size, initializer=tf.contrib.layers.variance_scaling_initializer(seed=23))
        conv = tf.nn.conv3d(input_, w, strides = strides, padding= padding)

        biases = tf.get_variable('biases', kernel_size[-1], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv2d_revised(input_, kernel_size, strides, dial=[1,1,1,1] ,name="conv2d", padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', kernel_size, initializer=tf.contrib.layers.xavier_initializer(seed=23))
        # w = tf.get_variable('w', kernel_size, initializer=tf.contrib.layers.variance_scaling_initializer(seed=23))
        conv = tf.nn.conv2d(input_, w, strides = strides, padding=padding, dilations=dial)

        biases = tf.get_variable('biases', kernel_size[-1], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

    
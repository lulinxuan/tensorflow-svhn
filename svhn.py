# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

import svhn_input
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'train_data/',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = svhn_input.IMAGE_SIZE
NUM_CLASSES = svhn_input.NUM_CLASSES
NUM_DIGITS = svhn_input.NUM_DIGITS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = svhn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = svhn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def bias_variable(name, shape, value=0.0, trainable = True):
  initial = tf.constant(value, shape=shape)
  with tf.device("/gpu:0"):
    var = tf.Variable(initial, name = name, trainable=trainable)
  return var


def weight_variable(name, shape, wd, conv=True):
  with tf.device("/gpu:0"):
    if not conv:
      var = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=shape), name = name)
    else:
      var = tf.Variable(tf.contrib.layers.xavier_initializer_conv2d()(shape=shape), name = name)

  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return svhn_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return svhn_input.inputs(data_dir=FLAGS.data_dir,
                              batch_size=FLAGS.batch_size)


def conv2d(name, l_input, w, b, k,is_training, padding = 'SAME'):
  conv = tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, k, k, 1], padding=padding),b)
  conv = bn(conv, is_training)#tf.nn.local_response_normalization(conv, name = 'norm')
  return tf.nn.relu(conv, name=name)

def max_pool(name, l_input, k, h=2):
  return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, h, h, 1], padding='SAME', name=name)

def bn(x, is_training):
  with tf.variable_scope('bn') as scope:
    bn_train = batch_norm(x, decay=0.99, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, trainable=True, scope = scope)
    bn_inf = batch_norm(x, decay=0.99, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, trainable=True, scope = scope)
  return tf.cond(is_training, lambda: bn_train, lambda: bn_inf)


def net_1(_X, _dropout):
  is_training = tf.constant(_dropout<1.)

  with tf.variable_scope('conv1') as scope:
    conv1 = conv2d('conv1', _X, weight_variable('weights', shape=[3,3,3,96], wd=0.004)
                                , bias_variable('biases', [96], 0.0), 1, is_training, padding = 'VALID')
    pool1 = max_pool('pool1', conv1, k=2)

  with tf.variable_scope('conv2') as scope:
    conv2 = conv2d('conv2', pool1, weight_variable('weights', shape=[3,3,96,128], wd=0.004)
                                , bias_variable('biases', [128], 0.0),1, is_training, padding = 'VALID')
    pool2 = max_pool('pool2', conv2, k=2)

  with tf.variable_scope('conv3') as scope:
    conv3 = conv2d('conv3', pool2, weight_variable('weights', shape=[3,3,128,160], wd=0.004)                                
                                , bias_variable('biases', [160], 0.0),1, is_training, padding = 'SAME')

  with tf.variable_scope('conv4') as scope:
    conv4 = conv2d('conv4', conv3, weight_variable('weights', shape=[3,3,160,196], wd=0.004)                                
                                , bias_variable('biases', [196], 0.0),1, is_training, padding = 'VALID')
    pool4 = max_pool('pool4', conv4, k=2)
  

  with tf.variable_scope('conv5') as scope:
    conv5 = conv2d('conv5', pool4, weight_variable('weights', shape=[3,3,196,256], wd=0.004)                                
                                , bias_variable('biases', [256], 0.0),1, is_training, padding = 'SAME')

  with tf.variable_scope('conv6') as scope:
    conv6 = conv2d('conv6', conv5, weight_variable('weights', shape=[3,3,256,312], wd=0.004)                                
                                , bias_variable('biases', [312], 0.0),1, is_training, padding = 'VALID')
    pool6 = max_pool('pool6', conv6, k=2)

  with tf.variable_scope('conv7') as scope:
    conv7 = conv2d('conv7', pool6, weight_variable('weights', shape=[3,3,312,360], wd=0.004)                                
                                , bias_variable('biases', [360], 0.0),1, is_training, padding = 'VALID')

  with tf.variable_scope('fc1') as scope:
    dim = 1
    for d in conv7.get_shape()[1:].as_list():
      dim *= d
    dense1 = tf.reshape(conv7, [-1, dim])      
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, weight_variable('weights', shape=[dim,384], wd=0.002, conv=False))
                                , bias_variable('biases', [384], 0.0)), name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)

  with tf.variable_scope('fc2') as scope:     
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, weight_variable('weights', shape=[384,196], wd=0.002, conv=False))
                                , bias_variable('biases', [196], 0.0)), name='fc2')
    dense2 = tf.nn.dropout(dense2, _dropout)


  with tf.variable_scope('out1') as scope:
    out1 = tf.add(tf.matmul(dense2, weight_variable('weights', shape=[196, NUM_CLASSES], wd=0.001, conv=False))
                                , bias_variable('biases', [NUM_CLASSES], 0.0))
  with tf.variable_scope('out2') as scope:
    out2 = tf.add(tf.matmul(dense2, weight_variable('weights', shape=[196, NUM_CLASSES], wd=0.001, conv=False))
                                , bias_variable('biases', [NUM_CLASSES], 0.0))
  with tf.variable_scope('out3') as scope:
    out3 = tf.add(tf.matmul(dense2, weight_variable('weights', shape=[196, NUM_CLASSES], wd=0.001, conv=False))
                                , bias_variable('biases', [NUM_CLASSES], 0.0))
  with tf.variable_scope('out4') as scope:
    out4 = tf.add(tf.matmul(dense2, weight_variable('weights', shape=[196, NUM_CLASSES], wd=0.001, conv=False))
                                , bias_variable('biases', [NUM_CLASSES], 0.0))
  with tf.variable_scope('out5') as scope:
    out5 = tf.add(tf.matmul(dense2, weight_variable('weights', shape=[196, NUM_CLASSES], wd=0.001, conv=False))
                                , bias_variable('biases', [NUM_CLASSES], 0.0))
  with tf.variable_scope('out6') as scope:
    out6 = tf.add(tf.matmul(dense2, weight_variable('weights', shape=[196, NUM_CLASSES], wd=0.001, conv=False))
                                , bias_variable('biases', [NUM_CLASSES], 0.0))

  return out1,out2,out3,out4,out5,out6


def loss(logits1,logits2,logits3,logits4,logits5,logits6, labels):
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, NUM_DIGITS])

  # Calculate the average cross entropy loss across the batch.
  cross_entropy_mean = \
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels[:,0], logits=logits1)) + \
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels[:,1], logits=logits2)) + \
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels[:,2], logits=logits3)) + \
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels[:,3], logits=logits4)) + \
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels[:,4], logits=logits5)) + \
    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels[:,5], logits=logits6))

  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss). 
  loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return loss


def _add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):

  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=False)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  extra_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]+extra_update_op):
    train_op = tf.no_op(name='train')

  return train_op

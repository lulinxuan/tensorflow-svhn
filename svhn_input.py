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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange
import tensorflow as tf
import scipy.io as sio

from tensorflow.python.platform import gfile

IMAGE_SIZE = 64

NUM_DIGITS = 6
NUM_CHANNEL = 3
NUM_CLASSES = 11 # 0-9, 10 stands for Nan
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_svhn(filename_queue):
  reader = tf.TFRecordReader()
  _, examples = reader.read(filename_queue)
  features = tf.parse_single_example(examples, 
    features={
      'size': tf.FixedLenFeature([], tf.int64),
      'label': tf.FixedLenFeature([], tf.string),
      'image': tf.FixedLenFeature([], tf.string)
    })
  return features


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):

  num_preprocess_threads = 16

  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size, NUM_DIGITS])


def distorted_inputs(data_dir, batch_size):

  filenames = [os.path.join(data_dir, 'full_train_imgs.tfrecords'), os.path.join(data_dir, 'full_extra_imgs.tfrecords')]             
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.  
  read_input = read_svhn(filename_queue)

  label=tf.cast(tf.reshape(tf.decode_raw(read_input['label'], tf.uint8), [NUM_DIGITS]), tf.int32)
  reshaped_image = tf.cast(tf.reshape(tf.decode_raw(read_input['image'], tf.uint8), [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]), tf.float32)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  distorted_image = tf.image.random_brightness(reshaped_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d SVHN images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size)


def inputs(data_dir, batch_size):
  
  filenames = [os.path.join(data_dir, 'full_test_imgs.tfrecords')]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_svhn(filename_queue)
  label=tf.cast(tf.reshape(tf.decode_raw(read_input['label'], tf.uint8), [NUM_DIGITS]), tf.int32)
  reshaped_image = tf.cast(tf.reshape(tf.decode_raw(read_input['image'], tf.uint8), [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]), tf.float32)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(reshaped_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size)

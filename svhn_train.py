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

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt

import numpy as np
from six.moves import xrange
import tensorflow as tf

import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'svhn_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/',
                           """Directory where to read model checkpoints.""")

def train():
  with tf.Graph().as_default() as graph:
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = svhn.distorted_inputs()

    logits1,logits2,logits3,logits4,logits5,logits6 = svhn.net_1(images, 0.71)
    
    loss = svhn.loss(logits1,logits2,logits3,logits4,logits5,logits6, labels)

    pred = tf.stack([tf.argmax(tf.nn.softmax(logits1), 1),\
      tf.argmax(tf.nn.softmax(logits2), 1),\
      tf.argmax(tf.nn.softmax(logits3), 1),\
      tf.argmax(tf.nn.softmax(logits4), 1),\
      tf.argmax(tf.nn.softmax(logits5), 1),\
      tf.argmax(tf.nn.softmax(logits6), 1)], axis=1)


    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = svhn.train(loss, global_step)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      variable_averages = tf.train.ExponentialMovingAverage(
          svhn.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      
      saver = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

      
      """ Save Whole Model to model.pb
      for v in tf.trainable_variables():
        # assign the ExponentialMovingAverage value to the real variable
        name = v.name.split(':')[0]+'/ExponentialMovingAverage'
        sess.run(tf.assign(v, variables_to_restore[name]))
      out_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['out1/Add','out2/Add','out3/Add','out4/Add','out5/Add','out6/Add', 'shuffle_batch'])
      with tf.gfile.GFile("model.pb", "wb") as f:
        f.write(out_graph_def.SerializeToString())
      """

      tf.train.start_queue_runners(sess=sess)

      summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                              graph=sess.graph)


      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value, prediction, label = sess.run([train_op, loss, pred, labels])
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 100 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          true_count = 0
          for x in range(num_examples_per_step):
            current_pred = np.array(prediction[x]).astype(int).tostring()
            correct_pred = np.array(label[x]).astype(int).tostring()
            if current_pred == correct_pred:
              true_count += 1

          format_str = ('%s: step %d, loss = %.6f, acc = %.6f%% (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value, 100*(true_count / num_examples_per_step),
                               examples_per_sec, sec_per_batch))
          # print(prediction)

        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)



def main(argv=None):
  if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
  gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


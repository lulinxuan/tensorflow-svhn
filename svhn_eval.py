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
import math
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'svhn_eval/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")


def eval_once(logits, labels, saver):

  summary_op = tf.summary.merge_all()

  with tf.Session() as sess:
    global_step = tf.Variable(0, trainable=False)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph=sess.graph)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions, label = sess.run([logits, labels])
        for x in range(FLAGS.batch_size):
          pred = np.array(predictions[x][1:]).astype(int).tostring()
          correct_pred = np.array(label[x][1:]).astype(int).tostring()

          if pred == correct_pred:
            true_count += 1
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary_writer.add_summary(sess.run(summary_op), global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

""" Load the model.pb
def load_model(images):
  print("loading model")
  with gfile.FastGFile("model.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
    for node in graph_def.node:
      if node.op == 'AssignSub':
        node.op='Sub'
        if 'use_locking' in node.attr:
          del(node.attr['use_locking'])

    logits1,logits2,logits3,logits4,logits5,logits6 = tf.import_graph_def(graph_def, input_map={'shuffle_batch:0': images}, return_elements=['out1/Add:0','out2/Add:0','out3/Add:0','out4/Add:0','out5/Add:0','out6/Add:0'])
    print("done")
    return logits1,logits2,logits3,logits4,logits5,logits6
"""

def evaluate():
  images, labels = svhn.inputs()

  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 6])
    # Build a Graph that computes the logits predictions from the
    # inference model.  

  logits1,logits2,logits3,logits4,logits5,logits6 = svhn.net_1(images, 1.)
    
  logits = tf.stack([tf.argmax(tf.nn.softmax(logits1), 1), \
      tf.argmax(tf.nn.softmax(logits2), 1),\
      tf.argmax(tf.nn.softmax(logits3), 1),\
      tf.argmax(tf.nn.softmax(logits4), 1),\
      tf.argmax(tf.nn.softmax(logits5), 1),\
      tf.argmax(tf.nn.softmax(logits6), 1)], axis=1)

  # Restore the moving average version of the learned variables for eval.
  variable_averages = tf.train.ExponentialMovingAverage(
        svhn.MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
  eval_once(logits, sparse_labels, saver)


def main(argv=None):
  if gfile.Exists(FLAGS.eval_dir):
    gfile.DeleteRecursively(FLAGS.eval_dir)
  gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()

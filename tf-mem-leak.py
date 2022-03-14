#!/usr/bin/env python3

"""
https://github.com/rwth-i6/returnn/issues/990
"""

import sys
import numpy
import tensorflow as tf
tf1 = tf.compat.v1

tf1.disable_eager_execution()
tf1.disable_control_flow_v2()


n_layers = 20
n_feat = 100
n_hidden = 200
n_steps = 100
n_epochs = sys.maxsize


def layer(x, n_out):
  n_in = x.get_shape().as_list()[-1]
  weights = tf.Variable(tf.random.normal((n_in, n_out), stddev=0.01))
  x = tf.matmul(x, weights)
  bias = tf.Variable(tf.zeros((n_out,)))
  x += bias
  return x


def model(x):
  x = layer(x, n_hidden)
  for _ in range(n_layers):
    x_ = x
    x_ = layer(x_, n_hidden * 2)
    x_ = tf.nn.relu(x_)
    x_ = layer(x_, n_hidden)
    x = x + x_
  x = layer(x, n_feat)
  return x


def run_epoch():
  print("Setup TF graph")
  graph = tf1.Graph()
  with graph.as_default():
    print("Setup TF session")
    session = tf1.Session(graph=graph)
    with session.as_default():
      print("Setup model and computation")
      x = tf1.placeholder(tf.float32, shape=(None, None, n_feat))
      targets = tf1.placeholder(tf.int32, shape=(None, None))
      y = model(x)
      loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=targets))
      opt = tf1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
      print("Initialize variables")
      session.run(tf1.global_variables_initializer())
      print("Run epoch")
      for step in range(n_steps):
        n_batch = numpy.random.randint(1, 10)
        n_time = numpy.random.randint(10, 100)
        feed_dict = {
          x: numpy.random.normal(0.0, 1.0, (n_batch, n_time, n_feat)).astype(numpy.float32),
          targets: numpy.random.randint(n_feat, size=(n_batch, n_time)).astype(numpy.int32)
        }
        _, loss_v = session.run((opt, loss), feed_dict=feed_dict)
      print("Last loss:", loss_v / (n_batch * n_time))


def main():
  for epoch in range(n_epochs):
    print(f"*** Epoch {epoch}")
    run_epoch()


if __name__ == '__main__':
  main()

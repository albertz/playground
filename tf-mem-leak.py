#!/usr/bin/env python3

"""
https://github.com/rwth-i6/returnn/issues/990
"""

import sys
import tensorflow as tf
tf1 = tf.compat.v1

tf1.disable_eager_execution()
tf1.disable_control_flow_v2()


n_layers = 1000
n_feat = 10
n_steps = 100
n_epochs = sys.maxsize


def model(x):
  for layer in range(n_layers):
    weights = tf.Variable(tf.random.normal((n_feat, n_feat * 2), stddev=0.1))
    bias = tf.Variable(tf.zeros((n_feat * 2,)))
    x_ = x
    x_ = tf.matmul(x_, weights)
    x_ += bias
    x_ = tf.nn.relu(x_)
    weights2 = tf.Variable(tf.random.normal((n_feat * 2, n_feat), stddev=0.1))
    bias2 = tf.Variable(tf.zeros((n_feat,)))
    x_ = tf.matmul(x_, weights2)
    x_ += bias2
    x += x_
  return x


def run_epoch():
  print("Setup TF graph")
  graph = tf1.Graph()
  with graph.as_default():
    print("Setup TF session")
    session = tf1.Session(graph=graph)
    with session.as_default():
      print("Setup model and computation")
      n_batch = tf.random.uniform(shape=[], minval=1, maxval=10, dtype=tf.int32)
      n_time = tf.random.uniform(shape=[], minval=10, maxval=100, dtype=tf.int32)
      x = tf.random.normal((n_batch, n_time, n_feat))
      y = model(x)
      targets = tf.random.uniform(shape=[n_batch, n_time], minval=0, maxval=n_feat, dtype=tf.int32)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=targets)
      opt = tf1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
      print("Initialize variables")
      session.run(tf1.global_variables_initializer())
      print("Run epoch")
      for step in range(n_steps):
        _, loss_v = session.run((opt, loss))
      print("Last loss:", loss_v)


def main():
  for epoch in range(n_epochs):
    print(f"*** Epoch {epoch}")
    run_epoch()


if __name__ == '__main__':
  main()

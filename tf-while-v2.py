
# https://github.com/tensorflow/tensorflow/issues/54458

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def main():
  g = tf.Graph()
  with g.as_default():
    session = tf.compat.v1.Session(graph=g)
    with session.as_default():

      v = tf.Variable(1.)

      def cond(i, x):
        return tf.less(i, 10)

      def body(i, x):
        return i + 1, x * 1.

      j, y = tf.while_loop(cond, body, [0., v])
      loss = tf.reduce_sum(y ** 2)

      session.run(tf.compat.v1.global_variables_initializer())

      opt = tf.compat.v1.train.GradientDescentOptimizer(0.1)
      opt_op = opt.minimize(loss)

      no_op = tf.no_op()
      session.run(no_op)  # here it crashes already!

      print(session.run((j, y, opt_op)))


if __name__ == '__main__':
  main()

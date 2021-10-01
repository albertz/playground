

import tensorflow as tf

print("TF:", tf.__version__)
tf.compat.v1.disable_eager_execution()

filename = "test-ckpt-diff-shape.model"


with tf.Graph().as_default() as graph:
    with tf.compat.v1.Session(graph=graph) as session:
        shape1 = (3,3,40,32)
        v = tf.compat.v1.get_variable(name="W", shape=shape1)
        print(v)
        saver = tf.compat.v1.train.Saver(var_list=[v])
        session.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess=session, save_path=filename)


with tf.Graph().as_default() as graph:
    with tf.compat.v1.Session(graph=graph) as session:
        shape2 = (3,3,1,32)
        v = tf.compat.v1.get_variable(name="W", shape=shape2)
        print(v)
        saver = tf.compat.v1.train.Saver(var_list=[v])
        saver.restore(sess=session, save_path=filename)
        v_raw = session.run(v)
        print(v)
        print(v_raw.shape)
        assert v.shape.as_list() == list(shape2)
        assert v.shape.as_list() == list(v_raw.shape)



def test_float_div_tensor():
  """
  https://youtrack.jetbrains.com/issue/PY-34893
  """
  import tensorflow as tf
  x = tf.placeholder(tf.float32)
  assert isinstance(x, tf.Tensor)
  y = 1.0 / x  # Expected type 'float', got 'Tensor' instead

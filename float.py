"""
Playing around with binary representation of floating points

https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

"""


import struct
import tensorflow as tf


def dump_float32_struct(x):
  print(bin(struct.unpack('!i', struct.pack('!f', x))[0]))


def dump_float32_tf(x):
  print(bin(tf.bitcast(tf.cast(x, tf.float32), tf.int32).numpy()))


def dump_bfloat16_tf(x):
  print(bin(tf.bitcast(tf.cast(x, tf.bfloat16), tf.int16).numpy()))


def float32_sign_exp_frac(x):
  i32 = tf.bitcast(tf.cast(x, tf.float32), tf.uint32)
  sign = tf.bitwise.bitwise_and(i32, 0x80000000)
  sign = tf.bitwise.right_shift(sign, 31)
  exp = tf.bitwise.bitwise_and(i32, 0x7F800000)
  exp = tf.bitwise.right_shift(exp, 23)
  frac = tf.bitwise.bitwise_and(i32, 0x007FFFFF)  # mantissa
  bit_flip = tf.cast(tf.random.uniform(maxval=2, shape=tf.shape(frac), dtype=tf.int32), tf.uint32)
  frac = tf.bitwise.bitwise_xor(frac, bit_flip)
  return sign, exp, frac


def float32_manual(x):
  sign, exp, frac = float32_sign_exp_frac(x)
  sign = tf.cast(sign, tf.bool)
  exp = tf.cast(exp, tf.float32)
  frac = tf.cast(frac, tf.float32)
  return tf.where(sign, -1., 1.) * tf.pow(2.0, exp - 127.0) * (1.0 + frac * 1.0 / 0x800000)


def float32_lsb_random_bitflip(x):
  i32 = tf.bitcast(tf.cast(x, tf.float32), tf.uint32)
  bit_flip = tf.cast(tf.random.uniform(maxval=2, shape=tf.shape(i32), dtype=tf.int32), tf.uint32)
  i32 = tf.bitwise.bitwise_xor(i32, bit_flip)
  return tf.bitcast(i32, tf.float32)


def bfloat16_lsb_random_bitflip(x):
  i16 = tf.bitcast(tf.cast(x, tf.bfloat16), tf.uint16)
  bit_flip = tf.cast(
    tf.less(tf.random.uniform(shape=tf.shape(i16)), 0.5),
    tf.uint16)
  print(bit_flip.numpy())
  i16 = tf.bitwise.bitwise_xor(i16, bit_flip)
  return tf.bitcast(i16, tf.bfloat16)


def main():
  tf.random.set_seed(123)
  while True:
    x = 1.2345
    print("***", x)
    dump_bfloat16_tf(x)
    # x_ = float32_manual(x)
    x_ = bfloat16_lsb_random_bitflip(x)
    print(x_.numpy())
    dump_bfloat16_tf(x_)


if __name__ == '__main__':
  main()

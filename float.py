"""
Playing around with binary representation of floating points

https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

"""


from typing import Optional
import struct
import tensorflow as tf


def dump_float32_struct(x, *prefix):
  print(*prefix, bin(struct.unpack('!i', struct.pack('!f', x))[0]))


def dump_float32_tf(x, *prefix):
  print(*prefix, bin(tf.bitcast(tf.cast(x, tf.float32), tf.int32).numpy()))


def dump_bfloat16_tf(x, *prefix):
  print(*prefix, bin(tf.bitcast(tf.cast(x, tf.bfloat16), tf.int16).numpy()))


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


def bfloat16_lsb_random_bitflip(x, *, bit_flip: Optional[bool] = None):
  i16 = tf.bitcast(tf.cast(x, tf.bfloat16), tf.uint16)
  if bit_flip is None:
    bit_flip = tf.cast(
      tf.less(tf.random.uniform(shape=tf.shape(i16)), 0.5),
      tf.uint16)
  else:
    bit_flip = tf.cast(bit_flip, tf.uint16)
  i16 = tf.bitwise.bitwise_xor(i16, bit_flip)
  return tf.bitcast(i16, tf.bfloat16)


def main():
  tf.random.set_seed(123)
  while True:
    x = 1.2345
    print("***", x)
    x_f32 = tf.cast(x, tf.float32)
    print("  float32:", x_f32.numpy())
    dump_float32_tf(x_f32, "  float32 bin:")
    x_bf16 = tf.cast(x_f32, tf.bfloat16)
    print("  bfloat16:", tf.cast(x_bf16, tf.float32).numpy())
    print("  bfloat16 abs error:",
          tf.abs(x_f32 - tf.cast(x_bf16, tf.float32)).numpy())
    print("  bfloat16 rel error:",
          tf.abs(x_f32 - tf.cast(x_bf16, tf.float32)).numpy() / tf.abs(x_f32).numpy())
    dump_bfloat16_tf(x_bf16, "  bfloat16 bin:            ")
    # x_ = float32_manual(x)
    x_bf16_rnd_bitflip = bfloat16_lsb_random_bitflip(x_bf16, bit_flip=True)
    print("  bfloat16 LSB bitflip:", x_bf16_rnd_bitflip.numpy())
    dump_bfloat16_tf(x_bf16_rnd_bitflip, "  bfloat16 LSB bitflip bin:")
    print("  bfloat16 LSB bitflip abs error:",
          tf.abs(x_f32 - tf.cast(x_bf16_rnd_bitflip, tf.float32)).numpy())
    print("  bfloat16 LSB bitflip rel error:",
          tf.abs(x_f32 - tf.cast(x_bf16_rnd_bitflip, tf.float32)).numpy() / tf.abs(x_f32).numpy())
    break


if __name__ == '__main__':
  main()

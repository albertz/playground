#!/usr/bin/env python3


import numpy
import tensorflow as tf

from tensorflow.contrib.image import dense_image_warp


def load_image():
  """
  :return: (width,height,channel=3)
  :rtype: numpy.ndarray
  """
  fn = "img-DSC06584-HDR.jpg"
  import PIL.Image  # pip install pillow
  img = PIL.Image.open(fn)
  img = numpy.array(img)
  if img.dtype == numpy.uint8:
    img = img.astype(numpy.float32) / 255.
  assert img.ndim == 3 and img.shape[-1] == 3 and img.dtype == numpy.float32
  img = img.transpose([1, 0, 2])
  img = numpy.flip(img, axis=1)
  return img


def gaussian_kernel_2d(size, std):
  """
  :param int size:
  :param float std:
  :return: (size,size), float32
  :rtype: tf.Tensor
  """
  d = tf.distributions.Normal(0.0, std)
  values = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
  values = tf.einsum('i,j->ij', values, values)
  values.set_shape((size * 2 + 1, size * 2 + 1))
  return values / tf.reduce_sum(values)


def blur(image, size=None, std=None):
  """
  :param tf.Tensor image: (batch,width,height,channel)
  :param int size:
  :param float std:
  :return: image
  :rtype: tf.Tensor
  """
  if std is None:
    std = 1.
  if size is None:
    size = int(std * 2 + 1)
  image.set_shape((None, None, None, None))
  orig_shape = tf.shape(image)
  image = tf.transpose(image, [0, 3, 1, 2])  # (B,C,W,H)
  image = tf.reshape(image, [orig_shape[0] * orig_shape[3], orig_shape[1], orig_shape[2], 1])  # (B*C,W,H,1)
  gauss_kernel = gaussian_kernel_2d(size=size, std=std)
  gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
  image = tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
  image = tf.reshape(image, [orig_shape[0], orig_shape[3], orig_shape[1], orig_shape[2]])  # (B,C,W,H)
  image = tf.transpose(image, [0, 2, 3, 1])  # (B,W,H,C)
  return image


class Transformer:
  def __init__(self):
    self.session = tf.Session()
    self.img_in = tf.placeholder(tf.float32, shape=(None, None, 3), name="img_in")  # (width,height,channel)
    self.img_ins = tf.expand_dims(self.img_in, axis=0)  # (batch,width,height,channel)
    self.flow = self._create_flow(shape=tf.shape(self.img_ins)[:-1], scale=50.)
    # self.img_outs = dense_image_warp(self.img_ins, flow=self.flow)
    self.img_outs = self._flow_x_to_img(self.flow)  # debug, to visualize the flow
    self.img_out = tf.squeeze(self.img_outs, axis=0)  # (width,height,channel)

  def _create_flow(self, shape, scale=None):
    """
    :param shape:
    :param scale:
    :return: [batch, height, width, 2]
    :rtype: tf.Tensor
    """
    scale_x = scale
    scale_y = scale
    # [batch, height, width, 2]
    flow1 = tf.random.normal(shape=shape, stddev=scale_x)
    flow2 = tf.random.normal(shape=shape, stddev=scale_y)
    flow = tf.concat([tf.expand_dims(flow1, axis=-1), tf.expand_dims(flow2, axis=-1)], axis=-1)
    flow.set_shape((None, None, None, 2))
    flow = blur(flow, std=5.)
    return flow

  def _flow_x_to_img(self, flow):
    """
    :param tf.Tensor flow:
    :return: [batch, height, width, 3]
    :rtype: tf.Tensor
    """
    flow.set_shape((None, None, None, 2))
    flow = flow[:, :, :, :1]
    return tf.concat([flow, flow, flow], axis=-1)

  def transform(self, image):
    """
    :param numpy.ndarray image: (width,height,channel=3)
    :return: transformed image, (width,height,channel)
    :rtype: numpy.ndarray
    """
    return self.session.run(self.img_out, feed_dict={self.img_in: image})


class Gui:
  def __init__(self, transformer):
    """
    :param Transformer transformer:
    """
    from kivy.app import App
    from kivy.uix.widget import Widget
    from kivy.graphics.texture import Texture
    from kivy.graphics import Rectangle

    self.transformer = transformer
    self.image = load_image()
    self.app = App()
    self.app.root = Widget()

    size = tuple(numpy.array(self.image.shape[:2]).transpose())
    print("image size:", size)

    self.texture = Texture.create(size=size, colorfmt="rgb", bufferfmt="float")
    self._blit(self.image)

    with self.app.root.canvas:
      Rectangle(texture=self.texture, pos=(0, 0), size=numpy.array(size) * 2)

  def run(self):
    from kivy.clock import Clock
    Clock.schedule_interval(self.callback, 1.)
    self.app.run()

  def _blit(self, img):
    """
    :param numpy.ndarray image: (width,height,channel=3), float32
    """
    assert img.ndim == 3 and img.shape[-1] == 3 and img.dtype == numpy.float32
    self.texture.blit_buffer(img.transpose([1, 0, 2]).tostring(), colorfmt="rgb", bufferfmt="float")

  def callback(self, dt):
    img = self.transformer.transform(self.image)
    self._blit(img)
    self.app.root.canvas.ask_update()


def main():
  import better_exchook
  better_exchook.install()
  transformer = Transformer()
  gui = Gui(transformer=transformer)
  gui.run()


if __name__ == "__main__":
  main()

#!/usr/bin/env python3


import numpy
import tensorflow as tf


def interpolate_bilinear(grid, query_points, name='interpolate_bilinear', indexing='ij'):
  """
  Similar to Matlab's interp2 function.
  Finds values for query points on a grid using bilinear interpolation.
  Adapted from tensorflow.contrib.image.dense_image_warp, from newer TF version which supports variable-sized images.

  :param tf.Tensor grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
  :param tf.Tensor query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
  :param str name: a name for the operation (optional).
  :param str indexing: whether the query points are specified as row and column (ij), or Cartesian coordinates (xy).
  :returns: a 3-D `Tensor` with shape `[batch, N, channels]`
  :rtype: tf.Tensor
  """
  assert indexing in ('ij', 'xy')

  with tf.name_scope(name):
    grid = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)

    shape = tf.shape(grid)
    batch_size, height, width, channels = [shape[i] for i in range(grid.get_shape().ndims)]
    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    query_points.set_shape((None, None, 2))
    num_queries = tf.shape(query_points)[1]

    with tf.control_dependencies([
        tf.assert_greater_equal(height, 2, message='Grid height must be at least 2.'),
        tf.assert_greater_equal(width, 2, message='Grid width must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      index_order = [0, 1] if indexing == 'ij' else [1, 0]
      unstacked_query_points = tf.unstack(query_points, axis=2)

    for dim in index_order:
      with tf.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = tf.constant(0.0, dtype=query_type)
        floor = tf.minimum(tf.maximum(min_floor, tf.floor(queries)), max_floor)
        int_floor = tf.cast(floor, tf.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)
    assert len(alphas) == len(floors) == len(ceils) == len(index_order) == 2

    with tf.control_dependencies([
        tf.assert_less_equal(
          tf.to_float(batch_size) * tf.to_float(height) * tf.to_float(width), numpy.iinfo(numpy.int32).max / 8.,
          message="""The image size or batch size is sufficiently large
                     that the linearized addresses used by array_ops.gather
                     may exceed the int32 limit.""")
    ]):
      flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
      batch_offsets = tf.reshape(tf.range(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
      """
      :param tf.Tensor y_coords:
      :param tf.Tensor x_coords:
      :param str name:
      :rtype: tf.Tensor
      """
      with tf.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = tf.gather(flattened_grid, linear_coordinates)
        return tf.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with tf.name_scope('interpolate'):
      interp_top = alphas[1] * (top_right - top_left) + top_left
      interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
      interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def dense_image_warp(image, flow, name='dense_image_warp'):
  """
  Image warping using per-pixel flow vectors.
  Adapted from tensorflow.contrib.image.dense_image_warp, from newer TF version which supports variable-sized images.

  :param tf.Tensor image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
  :param tf.Tensor flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
  :param str name: A name for the operation (optional).
  :returns: A 4-D float `Tensor` with shape`[batch, height, width, channels]` and same type as input image.
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    image_shape = tf.shape(image)
    batch_size, height, width, channels = [image_shape[i] for i in range(image.get_shape().ndims)]
    # The flow is defined on the image grid. Turn the flow into a list of query points in the grid space.
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the image grid.
    interpolated = interpolate_bilinear(image, query_points_flattened)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated


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
  :param int|(int,int) size:
  :param float|(float,float) std:
  :return: (size_x*2+1,size_y*2+1), float32
  :rtype: tf.Tensor
  """
  if isinstance(size, (tuple, list)):
    size_x, size_y = size
  else:
    size_x, size_y, = size, size
  if isinstance(std, (tuple, list)):
    std_x, std_y = std
  else:
    std_x, std_y = std, std
  values_x = tf.range(start=-size_x, limit=size_x + 1, dtype=tf.float32)
  values_y = tf.range(start=-size_y, limit=size_y + 1, dtype=tf.float32)
  dx = tf.distributions.Normal(0.0, std_x)
  dy = tf.distributions.Normal(0.0, std_y)
  values_x = dx.prob(values_x)
  values_y = dy.prob(values_y)
  values = tf.einsum('i,j->ij', values_x, values_y)
  values.set_shape((size_x * 2 + 1, size_y * 2 + 1))
  return values / tf.reduce_sum(values)


def gaussian_blur_2d(image, kernel_size=None, kernel_std=None):
  """
  :param tf.Tensor image: (batch,width,height,channel)
  :param int|(int,int)|None kernel_size:
  :param float|(float,float)|None kernel_std:
  :return: image
  :rtype: tf.Tensor
  """
  if kernel_std is None:
    kernel_std = 1.
  if kernel_size is None:
    if isinstance(kernel_std, (tuple, list)):
      assert len(kernel_std) == 2
      kernel_size = (int(kernel_std[0] * 2 + 1), int(kernel_std[1] * 2 + 1))
    else:
      kernel_size = int(kernel_std * 2 + 1)
  image.set_shape((None, None, None, None))
  orig_shape = tf.shape(image)
  orig_shape = [orig_shape[i] for i in range(image.get_shape().ndims)]
  image = tf.transpose(image, [0, 3, 1, 2])  # (B,C,W,H)
  image = tf.reshape(image, [orig_shape[0] * orig_shape[3], orig_shape[1], orig_shape[2], 1])  # (B*C,W,H,1)
  gauss_kernel = gaussian_kernel_2d(size=kernel_size, std=kernel_std)
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
    self.flow = self._create_flow(shape=tf.shape(self.img_ins)[:-1], std=50.)
    self.img_outs = dense_image_warp(self.img_ins, flow=self.flow)
    # self.img_outs = self._flow_x_to_img(self.flow)  # debug, to visualize the flow
    self.img_out = tf.squeeze(self.img_outs, axis=0)  # (width,height,channel)

  def _create_flow(self, shape, std=None, scale=5., blur_std=None):
    """
    :param tf.Tensor shape: 1D, contains (batch,height,width)
    :param float std:
    :param float scale:
    :return: [batch, height, width, 2]
    :rtype: tf.Tensor
    """
    if blur_std is None:
      blur_std = std * 0.5
    shape.set_shape((3,))  # b,h,w
    small_shape = [shape[0], shape[1] // int(scale), shape[2] // int(scale)]
    scale_x = std
    scale_y = std
    # [batch, height, width, 2]
    flow1 = tf.random_normal(shape=small_shape, stddev=scale_x)
    flow2 = tf.random_normal(shape=small_shape, stddev=scale_y)
    flow = tf.stack([flow1, flow2], axis=-1)
    flow.set_shape((None, None, None, 2))
    flow = gaussian_blur_2d(flow, kernel_std=blur_std // scale)
    flow = tf.image.resize_images(flow, size=[shape[1], shape[2]])
    flow *= scale
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
      Rectangle(texture=self.texture, pos=(0, 0), size=numpy.array(size))

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
  print("TensorFlow:", tf.VERSION)
  transformer = Transformer()
  gui = Gui(transformer=transformer)
  gui.run()


if __name__ == "__main__":
  main()

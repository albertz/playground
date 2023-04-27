
import numpy
import numpy.testing
import tensorflow as tf
import torch


try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ImportError:
    pass


rnd = numpy.random.RandomState(42)
n_batch = 1
n_time = 16


x = rnd.randn(n_batch, n_time)
x_tf = tf.convert_to_tensor(x)
x_pt = torch.from_numpy(x)


frame_step = 1
# frame_length = fft_length, and power of two (e.g. 8), or just even, then looks the same
frame_length = 6
fft_length = 8

assert fft_length % 2 == 0 and frame_length % 2 == 0  # broken otherwise?

y_tf = tf.signal.stft(x, frame_step=frame_step, frame_length=frame_length, fft_length=fft_length)
print(y_tf)

x_pt = torch.nn.functional.pad(x_pt, ((fft_length - frame_length) // 2, (fft_length - frame_length) // 2))

y_pt = torch.stft(
    x_pt,
    n_fft=fft_length, hop_length=frame_step, win_length=frame_length,
    window=torch.hann_window(frame_length),
    center=False,
    return_complex=True,
)
y_pt = y_pt.transpose(1, 2)
print(y_pt)


y_tf_np = y_tf.numpy()
y_pt_np = y_pt.numpy()

print("TF shape:", y_tf_np.shape)
print("PT shape:", y_pt_np.shape)

assert y_tf_np.shape == y_pt_np.shape, f"TF shape {y_tf_np.shape} != PT shape {y_pt_np.shape}"
numpy.testing.assert_allclose(y_tf_np, y_pt_np, rtol=1e-5, atol=1e-5)

"""
tf.signal.stft vs torch.stft

https://github.com/pytorch/pytorch/issues/100177
https://github.com/librosa/librosa/issues/695
https://github.com/librosa/librosa/issues/596
"""

import numpy
import numpy.testing
import tensorflow as tf
import torch
import librosa
import scipy


try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ImportError:
    pass


rnd = numpy.random.RandomState(42)
n_batch = 1
n_time = 16


x = rnd.randn(n_batch, n_time)


# frame_step has no influence on the tests, however frame_step 1 is easier for debugging and understanding.
frame_step = 1
# frame_length = fft_length, and even, then all agree.
# frame_length < fft_length, and even, they disagree, but we can replicate it.
# frame_length uneven, it's even stranger.
# fft_length uneven does not seem to matter, they still all agree.
frame_length = 5
fft_length = 8

x_tf = tf.convert_to_tensor(x)
y_tf = tf.signal.stft(x, frame_step=frame_step, frame_length=frame_length, fft_length=fft_length)
print(y_tf)
y_tf_np = y_tf.numpy()

x_pt = torch.from_numpy(x)
x_pt = torch.nn.functional.pad(x_pt, (0, (fft_length - frame_length)))

window_pt = torch.hann_window(frame_length - frame_length % 2)
window_pt = torch.nn.functional.pad(window_pt, (0, (fft_length - frame_length + frame_length % 2)))

y_tf_like_pt = torch.stft(
    x_pt,
    n_fft=fft_length, hop_length=frame_step, win_length=fft_length,
    window=window_pt,
    center=False,
    return_complex=True,
)
y_tf_like_pt = y_tf_like_pt.transpose(1, 2)
print(y_tf_like_pt)
y_tf_like_pt_np = y_tf_like_pt.numpy()


x_pt = torch.from_numpy(x)
y_pt = torch.stft(
    x_pt,
    n_fft=fft_length, hop_length=frame_step, win_length=frame_length,
    window=torch.hann_window(frame_length),
    center=False,
    return_complex=True,
)
y_pt = y_pt.transpose(1, 2)
print(y_pt)
y_pt_np = y_pt.numpy()


def ceildiv(a, b):
    return -(-a // b)


y_tf_like_np = numpy.zeros(
    (n_batch, ceildiv(n_time - frame_length + 1, frame_step), fft_length // 2 + 1), dtype="complex64")
window_np = numpy.hanning(frame_length + 1 - frame_length % 2)[:frame_length]
for b in range(y_tf_like_np.shape[0]):
    for t in range(y_tf_like_np.shape[1]):
        for f in range(y_tf_like_np.shape[2]):
            for k in range(frame_length):
                y_tf_like_np[b, t, f] += (
                    window_np[k]
                    * x[b, t * frame_step + k]
                    * numpy.exp(-2j * numpy.pi * f * k / fft_length))

print("TF-like in pure Numpy:")
print(y_tf_like_np)

y_pt_like_np = numpy.zeros(
    (n_batch, ceildiv(n_time - fft_length + 1, frame_step), fft_length // 2 + 1), dtype="complex64")
window_np = numpy.hanning(frame_length + 1)[:frame_length]
for b in range(y_pt_like_np.shape[0]):
    for t in range(y_pt_like_np.shape[1]):
        for f in range(y_pt_like_np.shape[2]):
            for k in range(frame_length):
                y_pt_like_np[b, t, f] += (
                    window_np[k]
                    * x[b, t * frame_step + k + (fft_length - frame_length) // 2]
                    * numpy.exp(-2j * numpy.pi * f * (k + (fft_length - frame_length) // 2) / fft_length))

print("PT-like in pure Numpy:")
print(y_pt_like_np)

y_lr_np = librosa.stft(
    x, n_fft=fft_length, hop_length=frame_step, win_length=frame_length,
    center=False
)
y_lr_np = y_lr_np.transpose(0, 2, 1)

window_np = numpy.hanning(frame_length + 1 - frame_length % 2)[:frame_length]
_, _, y_sp_np = scipy.signal.stft(
    x,
    nperseg=frame_length,
    noverlap=frame_length - frame_step,
    nfft=fft_length,
    # With frame_length even, it is consistent.
    # With frame_length uneven, the windowing logic is slightly different.
    window="hann" if frame_length % 2 == 0 else window_np,
    padded=False, boundary=None)
y_sp_np = y_sp_np.transpose(0, 2, 1)
sp_inv_scale = numpy.sqrt(scipy.signal.get_window("hann", frame_length - (frame_length % 2), fftbins=True).sum()**2)
y_sp_np *= sp_inv_scale

print("TF shape:", y_tf_np.shape)
print("TF-like in NP shape:", y_tf_like_np.shape)
print("TF-like in PT shape:", y_tf_like_pt_np.shape)
print("Scipy shape:", y_sp_np.shape)

assert y_tf_np.shape == y_tf_like_pt_np.shape == y_tf_like_np.shape == y_sp_np.shape, (
    f"TF shape {y_tf_np.shape} == TF-like PT shape {y_tf_like_pt_np.shape}"
    f"== TF-like NP shape {y_tf_like_np.shape} == Scipy shape {y_sp_np.shape}")
numpy.testing.assert_allclose(y_tf_np, y_tf_like_np, rtol=1e-5, atol=1e-5, err_msg="TF != TF-like NP")
numpy.testing.assert_allclose(y_tf_np, y_tf_like_pt_np, rtol=1e-5, atol=1e-5, err_msg="TF != TF-like PT")
numpy.testing.assert_allclose(y_tf_np, y_sp_np, rtol=1e-5, atol=1e-5, err_msg="TF != Scipy")

print("PT shape:", y_pt_np.shape)
print("PT-like in NP shape:", y_pt_like_np.shape)
print("Librosa shape:", y_lr_np.shape)
assert y_pt_np.shape == y_pt_like_np.shape == y_lr_np.shape, (
    f"PT shape {y_pt_np.shape} == PT-like NP shape {y_pt_like_np.shape} == Librosa shape {y_lr_np.shape}")
numpy.testing.assert_allclose(y_pt_np, y_pt_like_np, rtol=1e-5, atol=1e-5, err_msg="PT != PT-like NP")
numpy.testing.assert_allclose(y_pt_np, y_lr_np, rtol=1e-5, atol=1e-5, err_msg="PT != Librosa")

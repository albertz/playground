"""
Utility functions to write wave files
"""

import wave
import numpy


def write_wave_file(filename: str, samples: numpy.ndarray, *, sr: int = 16_000, w: int = 2):
    """
    Write a wave file to disk

    :param filename:
    :param samples: 1D, float, -1 to 1
    :param sr: sample rate
    :param w: sample width in bytes
    """
    assert samples.ndim == 1
    samples = samples.clip(-1, 1)
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setframerate(sr)
        f.setsampwidth(w)
        samples_int = (samples * (2 ** (8 * w - 1) - 1)).astype({1: "int8", 2: "int16", 4: "int32"}[w])
        f.writeframes(samples_int.tobytes())
        f.close()

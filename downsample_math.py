import numpy as np


def ceildiv_generic(a, b):
    return -(-a // b)


def ceildiv(a, b):
    return (a + b - 1) // b


def ceildiv_v2(a, b):
    return (a - 1) // b + 1


def wave_to_align_len_ref_peter(wave_len: int) -> int:
    feat_len = (wave_len - 200) // 80 + 1
    align_len = np.ceil(feat_len / 2)
    align_len = np.ceil(align_len / 2)
    return int(align_len)


def wave_to_align_len(wave_len: int) -> int:
    return (wave_len - 200) // (80 * 2 * 2) + 1


def wave_to_align_len_v2(wave_len: int) -> int:
    return ceildiv(wave_len - 200 + 1, 80 * 2 * 2)


def test_wave_to_align_len():
    for i in range(0, 10_000):
        a = wave_to_align_len_v2(i)
        b = wave_to_align_len_ref_peter(i)
        assert a == b, (i, a, b)

#!/usr/bin/env python3
"""
Try to reproduce:
RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR
https://github.com/rwth-i6/returnn/issues/1465
https://github.com/pytorch/pytorch/issues/119420
"""

import torch
import sys


def main():
    """main"""
    print("PyTorch:", torch.__version__)
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name())

    window = torch.hann_window(512, device=dev)

    mem = []
    blob_size = 100_000_000
    while True:
        blob = torch.rand(blob_size, device=dev)
        mem.append(blob)
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.max_memory_allocated()
        print(
            f"Total GPU memory {human_bytes_size(total)},"
            f" alloc {human_bytes_size(alloc)},"
            f" free {human_bytes_size(free)} ({free / total * 100:.1f}%)"
        )
        blob_size = min(int(free * 0.8 / 4), blob_size)
        if free / total < 0.01:
            break

    torch.manual_seed(1337)
    step = 0
    count_oom = 0
    count_runtime_error = 0
    while True:
        try:
            n_batch = torch.randint(1, 100, ())
            # n_time = torch.randint(1_000, 500_000, ())
            max_time = min(int(blob_size / n_batch / 2), 500_000)
            # n_time = torch.randint(min(1_000, max_time // 2), min(max_time, 500_000), ())
            n_time = torch.randint(max_time // 100, max_time, ())
            x = torch.randn(n_batch, n_time, device=dev)
            print("x:", x)
            step += 1
        except torch.cuda.OutOfMemoryError:
            count_oom += 1
            continue

        try:
            y = torch.stft(
                x,
                n_fft=512,
                hop_length=160,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            y = y.real
            print("y:", y)
        except torch.cuda.OutOfMemoryError:
            count_oom += 1
        except RuntimeError as exc:
            print("RuntimeError:", exc)
            count_runtime_error += 1
            print(f"(step {step}, count OOM {count_oom}, count RuntimeError {count_runtime_error})")
            if count_runtime_error > 0:
                sys.exit(1)


def human_size(n, factor=1000, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: for each of the units K, M, G, T
    :param float frac: when to go over to the next bigger unit
    :param int prec: how much decimals after the dot
    :return: human readable size, using K, M, G, T
    :rtype: str
    """
    postfixes = ["", "K", "M", "G", "T"]
    i = 0
    while i < len(postfixes) - 1 and n > (factor ** (i + 1)) * frac:
        i += 1
    if i == 0:
        return str(n)
    return ("%." + str(prec) + "f") % (float(n) / (factor**i)) + postfixes[i]


def human_bytes_size(n, factor=1024, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: see :func:`human_size`. 1024 by default for bytes
    :param float frac: see :func:`human_size`
    :param int prec: how much decimals after the dot
    :return: human readable byte size, using K, M, G, T, with "B" at the end
    :rtype: str
    """
    return human_size(n, factor=factor, frac=frac, prec=prec) + "B"


if __name__ == "__main__":
    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass

    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)

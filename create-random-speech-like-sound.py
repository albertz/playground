#!/usr/bin/env python3
"""
Create sound
"""

import argparse
import wave
import io
import struct
import numpy as np


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("output_file")

    group = arg_parser.add_argument_group("WAV file parameters")
    group.add_argument("--rate", type=int, default=16_000)
    group.add_argument("--channels", type=int, default=1)
    group.add_argument("--sample-width-bytes", type=int, default=2)

    group = arg_parser.add_argument_group("Sound parameters")
    group.add_argument("--duration", type=float, default=5.0, help="in seconds")
    group.add_argument("--frequency", type=float, default=150.0, help="default 150, in between male/female")
    group.add_argument("--amplitude", type=float, default=0.3)
    group.add_argument("--num-random-freqs-per-sec", type=int, default=15, help="like speech rate")
    group.add_argument("--random-seed", type=int, default=42)

    args = arg_parser.parse_args()

    f = wave.open(args.output_file, "wb")
    f.setframerate(args.rate)
    f.setnchannels(args.channels)
    f.setsampwidth(args.sample_width_bytes)

    num_frames = int(args.rate * args.duration)
    rnd = np.random.RandomState(args.random_seed)
    rnd_frequencies = (
            (rnd.normal(size=[int(args.duration * args.num_random_freqs_per_sec) + 1]) * 0.5 + 1.0)
            * args.frequency)

    frame_idxs = np.arange(num_frames, dtype=np.int64)
    freq_idx_f = (frame_idxs * args.num_random_freqs_per_sec) / args.rate
    freq_idx = freq_idx_f.astype(np.int64)
    next_freq_idx = np.minimum(freq_idx + 1, len(rnd_frequencies) - 1)
    freq = rnd_frequencies[freq_idx] * (1 - freq_idx_f % 1) + rnd_frequencies[next_freq_idx] * (freq_idx_f % 1)

    ts = np.cumsum(freq / args.rate)
    samples = np.sin(2 * np.pi * ts)

    samples *= args.amplitude * (0.666 + 0.333 * np.sin(2 * np.pi * frame_idxs * 2 / args.rate))
    samples_int = (
        (samples * (2 ** (8 * args.sample_width_bytes - 1) - 1))
        .astype({1: np.int8, 2: np.int16, 4: np.int32}[args.sample_width_bytes]))

    f.writeframes(samples_int.tobytes())
    f.close()


if __name__ == "__main__":
    main()

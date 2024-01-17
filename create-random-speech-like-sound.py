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
    data = io.BytesIO()
    struct_fmt = "<" + {1: "b", 2: "h", 4: "i"}[args.sample_width_bytes]
    t = 0.0
    for i in range(num_frames):
        freq_idx_f = (i * args.num_random_freqs_per_sec) / args.rate
        freq_idx = int(freq_idx_f)
        next_freq_idx = min(freq_idx + 1, len(rnd_frequencies) - 1)
        freq = rnd_frequencies[freq_idx] * (1 - freq_idx_f % 1) + rnd_frequencies[next_freq_idx] * (freq_idx_f % 1)

        t += freq / args.rate

        sample = np.sin(2 * np.pi * t)

        sample *= min(1.0, args.amplitude * (1.0 + 0.5 * np.sin(2 * np.pi * i / args.rate)))

        sample_int = int(sample * (2 ** (8 * args.sample_width_bytes - 1) - 1))
        data.write(struct.pack(struct_fmt, sample_int))

    f.writeframes(data.getvalue())
    f.close()


if __name__ == "__main__":
    main()

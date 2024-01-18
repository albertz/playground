#!/usr/bin/env python3
"""
Create sound
"""

import argparse
import wave
import torch


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
    group.add_argument("--num-random-freqs-per-sec", type=float, default=15, help="like speech rate")
    group.add_argument("--random-seed", type=int, default=42)
    group.add_argument("--amplitude-frequency", type=float, default=2)

    args = arg_parser.parse_args()

    torch.manual_seed(args.random_seed)

    f = wave.open(args.output_file, "wb")
    f.setframerate(args.rate)
    f.setnchannels(args.channels)
    f.setsampwidth(args.sample_width_bytes)

    num_frames = int(args.rate * args.duration)
    rnd_frequencies = (
            (torch.randn(size=[int(args.duration * args.num_random_freqs_per_sec) + 1]) * 0.5 + 1.0)
            * args.frequency)

    frame_idxs = torch.arange(num_frames, dtype=torch.int64)
    freq_idx_f = (frame_idxs * args.num_random_freqs_per_sec) / args.rate
    freq_idx = freq_idx_f.to(torch.int64)
    next_freq_idx = torch.clip(freq_idx + 1, 0, len(rnd_frequencies) - 1)
    freq = rnd_frequencies[freq_idx] * (1 - freq_idx_f % 1) + rnd_frequencies[next_freq_idx] * (freq_idx_f % 1)

    ts = torch.cumsum(freq / args.rate, dim=0)
    samples = torch.sin(2 * torch.pi * ts)

    samples *= (
        args.amplitude
        * (0.666 + 0.333 * torch.sin(2 * torch.pi * frame_idxs * args.amplitude_frequency / args.rate)))

    samples_int = (
        (samples * (2 ** (8 * args.sample_width_bytes - 1) - 1))
        .to({1: torch.int8, 2: torch.int16, 4: torch.int32}[args.sample_width_bytes]))

    f.writeframes(samples_int.numpy().tobytes())
    f.close()


if __name__ == "__main__":
    main()

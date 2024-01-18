#!/usr/bin/env python3
"""
Create sound
"""

from typing import Optional
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
    group.add_argument("--amplitude-frequency", type=float, default=None)

    args = arg_parser.parse_args()

    torch.manual_seed(args.random_seed)

    f = wave.open(args.output_file, "wb")
    f.setframerate(args.rate)
    f.setnchannels(args.channels)
    f.setsampwidth(args.sample_width_bytes)

    num_frames = int(args.rate * args.duration)
    samples = generate(
        1,
        num_frames,
        samples_per_sec=args.rate,
        frequency=args.frequency,
        num_random_freqs_per_sec=args.num_random_freqs_per_sec,
        amplitude=args.amplitude,
        amplitude_frequency=args.amplitude_frequency,
    )  # [B,T]
    samples = samples[0]

    samples_int = (samples * (2 ** (8 * args.sample_width_bytes - 1) - 1)).to(
        {1: torch.int8, 2: torch.int16, 4: torch.int32}[args.sample_width_bytes]
    )

    f.writeframes(samples_int.numpy().tobytes())
    f.close()


def generate(
    batch_size: int,
    num_frames: int,
    *,
    samples_per_sec: int = 16_000,
    frequency: float = 150.0,
    num_random_freqs_per_sec: int = 15,
    amplitude: float = 0.3,
    amplitude_frequency: Optional[float] = None,
) -> torch.Tensor:
    """
    generate audio

    :return: shape [batch_size,num_frames]
    """
    frame_idxs = torch.arange(num_frames, dtype=torch.int64)  # [T]

    samples = _integrate_rnd_frequencies(
        batch_size,
        frame_idxs,
        base_frequency=frequency,
        samples_per_sec=samples_per_sec,
        num_random_freqs_per_sec=num_random_freqs_per_sec,
    )  # [T,B]

    if amplitude_frequency is None:
        amplitude_frequency = frequency / 75.0
    amplitude_variations = _integrate_rnd_frequencies(
        batch_size,
        frame_idxs,
        base_frequency=amplitude_frequency,
        samples_per_sec=samples_per_sec,
        num_random_freqs_per_sec=amplitude_frequency,
    )  # [T,B]

    samples *= amplitude * (0.666 + 0.333 * amplitude_variations)
    return samples.permute(1, 0)  # [B,T]


def _integrate_rnd_frequencies(
    batch_size: int,
    frame_idxs: torch.Tensor,
    *,
    base_frequency: float,
    samples_per_sec: int,
    num_random_freqs_per_sec: float,
) -> torch.Tensor:
    rnd_freqs = (
        torch.randn(size=(int(len(frame_idxs) * num_random_freqs_per_sec / samples_per_sec) + 1, batch_size)) * 0.5
        + 1.0
    ) * base_frequency  # [T',B]

    freq_idx_f = (frame_idxs * num_random_freqs_per_sec) / samples_per_sec
    freq_idx = freq_idx_f.to(torch.int64)
    next_freq_idx = torch.clip(freq_idx + 1, 0, len(rnd_freqs) - 1)
    frac = (freq_idx_f % 1)[:, None]  # [T,1]
    freq = rnd_freqs[freq_idx] * (1 - frac) + rnd_freqs[next_freq_idx] * frac  # [T,B]

    ts = torch.cumsum(freq / samples_per_sec, dim=0)  # [T,B]
    return torch.sin(2 * torch.pi * ts)


def _setup_better_exchook():
    try:
        import better_exchook
    except ImportError:
        return
    better_exchook.install()


def _setup_lovely_tensors():
    try:
        import lovely_tensors
    except ImportError:
        pass
    else:
        lovely_tensors.monkey_patch()


if __name__ == "__main__":
    _setup_better_exchook()
    _setup_lovely_tensors()
    main()

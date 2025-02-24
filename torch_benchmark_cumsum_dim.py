"""
Figure out, what is the optimal ``dim`` for ``torch.cumsum``.
"""

from typing import Tuple
import argparse
import torch
import torch.utils.benchmark as benchmark


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--shape", type=_parse_shape, default=(1000, 1000, 1000))
    args = arg_parser.parse_args()

    print("Torch:", torch.__version__)
    dev = torch.device(args.device)
    print("Device:", dev)

    x = torch.randn(args.shape, device=dev)

    for dim in range(x.dim()):
        timer = benchmark.Timer(stmt=f"_torch_cumsum(x, dim={dim})", globals={"x": x, "_torch_cumsum": _torch_cumsum})
        print(timer.timeit(100))

    for dim in range(1, x.dim()):
        timer = benchmark.Timer(
            stmt=f"_torch_cumsum_move(x, dim={dim})", globals={"x": x, "_torch_cumsum_move": _torch_cumsum_move}
        )
        print(timer.timeit(100))


def _torch_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cumsum(x, dim=dim)


def _torch_cumsum_move(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.moveaxis(dim, 0)
    return torch.cumsum(x, dim=0)


def _parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(map(int, s.split(",")))


if __name__ == "__main__":
    main()

import torch


class Network(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.param = torch.nn.Parameter(torch.randn((n,)))

    def forward(self):
        x = torch.randn((self.n,))
        x = mod1(x)
        x = mod2(x)
        x = mod3(x) * self.param
        x = mod4(x)
        x = mod1(x)
        x = mod2(x)
        x = mod2(x)
        x = mod5(x)
        return x.sum()


def mod1(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def mod2(x: torch.Tensor) -> torch.Tensor:
    return x.exp()


def mod3(x: torch.Tensor) -> torch.Tensor:
    return x - 2


def mod4(x: torch.Tensor) -> torch.Tensor:
    x.subtract_(-3.5)
    return x


def mod5(x: torch.Tensor) -> torch.Tensor:
    return x / x


def main():
    torch.manual_seed(42)

    net = Network(5)

    with torch.autograd.detect_anomaly():
        loss = net()
        loss.backward()


if __name__ == "__main__":
    main()

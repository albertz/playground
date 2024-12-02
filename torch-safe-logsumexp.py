"""
https://discuss.pytorch.org/t/gradients-of-torch-where/26835/6
"""

from typing import Union
import torch


def safe_logsumexp(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        max_x, _ = x.max(dim=dim, keepdim=True)
        max_x = max_x.detach()
        max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
    return max_x_ + torch.where(max_x_.isneginf(), 0.0, (x - max_x).exp().sum(dim=dim, keepdim=keepdim).log())


def safe_logsumexp2(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        max_x, _ = x.max(dim=dim, keepdim=True)
        max_x = max_x.detach()
        max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
        mask = max_x_.isneginf()

    sum_ = (x - max_x.masked_fill(mask, 0)).exp().sum(dim=dim, keepdim=keepdim)
    return max_x_ + sum_.masked_fill_(mask, 1).log()


def safe_logsumexp3(x: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """safe logsumexp, handles the case of -inf values. otherwise, when all are -inf, you get nan"""
    with torch.no_grad():
        max_x, _ = x.max(dim=dim, keepdim=True)
        max_x = max_x.detach()
        max_x_ = max_x if keepdim else max_x.squeeze(dim=dim)
    diff = torch.where(max_x.isneginf(), 0, x - max_x)
    return max_x_ + diff.exp().sum(dim=dim, keepdim=keepdim).log()


def safe_where(mask: torch.Tensor, a: Union[torch.Tensor, float], b: torch.Tensor) -> torch.Tensor:
    """safe where, handles the case of nan values"""
    a, b = torch.broadcast_tensors(torch.as_tensor(a), torch.as_tensor(b))
    a_ = a[mask]
    b_ = b[~mask]
    res = torch.empty_like(mask, dtype=a.dtype)
    res[mask] = a_
    res[~mask] = b_
    return res


if __name__ == "__main__":
    print("Torch:", torch.__version__)
    # torch.autograd.set_detect_anomaly(True)

    t = torch.tensor([float("-inf"), float("-inf"), float("-inf")], requires_grad=True)
    sum_t = torch.logsumexp(t, dim=0)
    sum_t.backward()
    print(sum_t, t.grad)

    t = torch.tensor([float("-inf"), float("-inf"), float("-inf")], requires_grad=True)
    sum_t = safe_logsumexp(t, dim=0)
    sum_t.backward()
    print(sum_t, t.grad)

    t = torch.tensor([float("-inf"), float("-inf"), float("-inf")], requires_grad=True)
    sum_t = safe_logsumexp2(t, dim=0)
    sum_t.backward()
    print(sum_t, t.grad)

    t = torch.tensor([float("-inf"), float("-inf"), float("-inf")], requires_grad=True)
    sum_t = safe_logsumexp3(t, dim=0)
    sum_t.backward()
    print(sum_t, t.grad)

    t = torch.tensor([1., 2., float("-inf")], requires_grad=True)
    sum_t = torch.logsumexp(t, dim=0)
    sum_t.backward()
    print(sum_t, t.grad)

    t = torch.tensor([1., 2., 3.], requires_grad=True)
    y = float("nan") * t
    y.masked_fill_(t >= 0., 0)
    y = t * t + y
    y.sum().backward()
    print(y, t.grad)


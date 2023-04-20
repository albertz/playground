"""
https://github.com/pytorch/pytorch/issues/99638
"""

import torch

batch_dim = 3
in_dim = 5
classes_dim = 5
time_dim_sizes = torch.tensor([4, 3, 2], dtype=torch.int32)
time_dim = 4


def own_pack_padded(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    :param x: [B,T,...]
    :param sizes: [B]
    :return: [sum(sizes),...]
    """
    batch_dim, time_dim, *remaining_dims = x.shape
    mask = torch.arange(time_dim, device=x.device)[None, :] < sizes[:, None]  # [B,T]
    mask: torch.Tensor
    mask_bc = mask.view(batch_dim, time_dim, *[1] * len(remaining_dims))  # [B*T]
    packed = torch.masked_select(x, mask_bc)
    packed = packed.view(-1, *remaining_dims)
    return packed


def loss_own_packed(logits: torch.Tensor, targets: torch.Tensor, sizes: torch.Tensor):
    logits_packed = own_pack_padded(logits, sizes)
    targets_packed = own_pack_padded(targets, sizes)
    loss_pt_packed_raw = torch.nn.CrossEntropyLoss(reduction="sum")(
        logits_packed, targets_packed.long()
    )
    return loss_pt_packed_raw


def loss_packed(logits: torch.Tensor, targets: torch.Tensor, sizes: torch.Tensor):
    logits_packed = torch.nn.utils.rnn.pack_padded_sequence(
        logits, sizes, batch_first=True, enforce_sorted=False
    )
    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets, sizes, batch_first=True, enforce_sorted=False
    )
    loss = torch.nn.CrossEntropyLoss(reduction="sum")(
        logits_packed.data, targets_packed.data.long()
    )
    return loss


def loss_padded(logits: torch.Tensor, targets: torch.Tensor, sizes: torch.Tensor):
    loss = torch.nn.CrossEntropyLoss(reduction="none")(logits.transpose(1, 2), targets.long())  # [B,T]
    mask = torch.arange(time_dim, device=loss.device)[None, :] < sizes[:, None]  # [B,T]
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    loss = loss.sum()
    return loss


try:
    import better_exchook
    better_exchook.install()
except ImportError:
    pass


try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ImportError:
    pass


for loss_fn in [loss_packed, loss_padded, loss_own_packed]:
    print("***", loss_fn.__name__)
    torch.manual_seed(42)

    net = torch.nn.Conv1d(in_dim, classes_dim, 5, padding="same")

    inputs = torch.randn(batch_dim, time_dim, in_dim, requires_grad=True)
    targets = torch.randint(0, classes_dim, (batch_dim, time_dim))

    x = inputs.transpose(1, 2)
    x = net(x)
    x = x.transpose(1, 2)

    loss = loss_fn(x, targets, time_dim_sizes)
    print("loss:", loss)
    print("bias grad:", torch.autograd.grad(loss, net.bias, create_graph=True))

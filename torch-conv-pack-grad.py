"""
https://github.com/pytorch/pytorch/issues/99638
"""

import torch

batch_dim = 3
in_dim = 5
classes_dim = 5
time_dim_sizes = torch.tensor([4, 3, 2], dtype=torch.int32)
time_dim = 4


def pack_padded_masked_select(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    :param x: [B,T,...]
    :param sizes: [B]
    :return: [sum(sizes),...]
    """
    batch_dim, time_dim, *remaining_dims = x.shape
    mask = torch.arange(time_dim, device=x.device)[None, :] < sizes[:, None]  # [B,T]
    mask: torch.Tensor
    mask_bc = mask.view(batch_dim, time_dim, *[1] * len(remaining_dims))  # [B,T,...]
    # This, together with convolution, will cause a bad gradient?
    packed = torch.masked_select(x, mask_bc)
    packed = packed.view(-1, *remaining_dims)
    return packed


def pack_padded_index_select(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    :param x: [B,T,...]
    :param sizes: [B]
    :return: [sum(sizes),...]
    """
    batch_dim, time_dim, *remaining_dims = x.shape
    mask = torch.arange(time_dim, device=x.device)[None, :] < sizes[:, None]  # [B,T]
    mask: torch.Tensor
    mask = mask.reshape(-1)  # [B*T]
    packed = torch.index_select(x.reshape(-1, *remaining_dims), dim=0, index=mask.reshape(-1).nonzero().flatten())
    return packed


def loss_packed_masked_select(logits: torch.Tensor, targets: torch.Tensor, sizes: torch.Tensor):
    logits_packed = pack_padded_masked_select(logits, sizes)
    targets_packed = pack_padded_masked_select(targets, sizes)
    loss_pt_packed_raw = torch.nn.CrossEntropyLoss(reduction="sum")(
        logits_packed, targets_packed.long()
    )
    return loss_pt_packed_raw


def loss_packed_index_select(logits: torch.Tensor, targets: torch.Tensor, sizes: torch.Tensor):
    logits_packed = pack_padded_index_select(logits, sizes)
    targets_packed = pack_padded_index_select(targets, sizes)
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


for loss_fn in [loss_packed, loss_padded, loss_packed_index_select, loss_packed_masked_select]:
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


# https://stackoverflow.com/questions/52988876/how-can-i-visualize-what-happens-during-loss-backward
print('Tracing back tensors:')


def trace_back_grad_funcs(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
            except AttributeError as e:
                trace_back_grad_funcs(n[0])


trace_back_grad_funcs(loss.grad_fn)

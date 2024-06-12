import argparse
import torch


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--device", type=str, default="cpu")
    arg_parser.add_argument("--eps", type=float, default=1e-5)
    arg_parser.add_argument("--momentum", type=float, default=0.1)
    args = arg_parser.parse_args()
    print("PyTorch:", torch.__version__)
    _setup_lovely_tensors()

    dev = torch.device(args.device)
    print("Using device:", dev)
    eps = args.eps
    momentum = args.momentum
    x_orig = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], device=dev)[:, :, None]  # [B,T,F]
    loss_per_frame = torch.tensor([[-1.0, 1.0, 2.0], [1.0, -2.0, 1.0]], device=dev)[:, :, None]  # [B,T,F]
    n = x_orig.size(0) * x_orig.size(1)

    x = x_orig.detach()
    x.requires_grad = True
    x_mean = x.mean(dim=(0, 1), keepdim=True)  # [1,1,F]
    x_mean_sg = x_mean.detach()
    x_var = ((x - x_mean_sg) ** 2).mean(dim=(0, 1), keepdim=True)  # [1,1,F]
    x_var_sg = x_var.detach()
    # (x_var * loss_per_frame).sum().backward()
    y = (x - x_mean) * (x_var + eps).rsqrt()
    print("y:", y)
    (y * loss_per_frame).sum().backward()
    print("x grad:", x.grad)

    bn_torch = torch.nn.BatchNorm1d(1, eps=eps, momentum=momentum, affine=False, device=dev)
    bn_torch.train()
    x = x_orig.detach()
    x.requires_grad = True
    y = bn_torch(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)  # [B,T,F]
    print("y:", y)
    (y * loss_per_frame).sum().backward()
    print("x grad:", x.grad)

    print("x mean (mom):", x_mean.flatten() * momentum)
    print("running_mean:", bn_torch.running_mean)
    print("x var (mom):", x_var.flatten() * (n / (n - 1)) * momentum + (1.0 - momentum))
    print("running_var:", bn_torch.running_var)


def _setup_lovely_tensors():
    try:
        import lovely_tensors
    except ImportError:
        return
    lovely_tensors.monkey_patch()


if __name__ == "__main__":
    main()

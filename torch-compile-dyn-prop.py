"""
Test dynamic property in class.
"""


class WrappedTensor:
    """wrapped tensor"""

    __slots__ = ("_tensor",)

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        """get"""
        return self._tensor


def func(wrapped_tensor: WrappedTensor):
    """func"""
    return wrapped_tensor.tensor.sum()


if __name__ == "__main__":
    import torch

    print("Torch:", torch.__version__)

    # This is an optimization.
    # However, this breaks torch.compile.
    # PyTorch 2.7 torch.compile throws an error here.
    # PyTorch 2.9 torch.compile runs without error,
    # but fails to see through the property access.
    WrappedTensor.tensor = property(WrappedTensor._tensor.__get__)

    wt = WrappedTensor(torch.tensor([1, 2, 3]))
    _f = torch.compile(func)
    result = _f(wt)
    print(result)  # Should print tensor(6)

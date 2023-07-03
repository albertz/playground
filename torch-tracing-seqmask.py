"""
https://github.com/pytorch/pytorch/issues/104521
"""

import torch
import onnxruntime
import numpy


class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        i_ = torch.arange(x.shape[1])  # [T]
        return i_[None, :] < seq_lens[:, None]  # [B, T]


print("Torch version:", torch.__version__)

torch.onnx.export(
    Mod(),
    (torch.zeros([3, 7]), torch.tensor([3, 2, 1])),
    f="test.onnx",
    verbose=True,
    input_names=["input", "seq_lens"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "time"}, "seq_lens": {0: "batch"}, "output": {0: "batch", 1: "time"}}
)

ort_session = onnxruntime.InferenceSession("test.onnx")
ort_inputs = {
    "input": numpy.ones([2, 5], dtype=numpy.float32),
    "seq_lens": numpy.array([5, 3], dtype=numpy.int64)
}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)

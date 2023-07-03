"""
https://github.com/pytorch/pytorch/issues/104521
"""

import torch
import onnxruntime
import numpy


class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.arange(x.shape[0])


print("Torch version:", torch.__version__)

torch.onnx.export(
    Mod(),
    torch.zeros([10]),
    f="test.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": [0], "output": [0]}
)

ort_session = onnxruntime.InferenceSession("test.onnx")
ort_inputs = {"input": numpy.ones([5], dtype=numpy.float32)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)

assert numpy.alltrue(ort_outs[0] == numpy.arange(5))

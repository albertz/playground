"""
Extract dyn shape
"""

import torch
import sympy

# noinspection PyProtectedMember
from torch._subclasses.fake_tensor import FakeTensorMode

from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic, SymInt

# noinspection PyProtectedMember
from torch._dynamo.source import ConstantSource


# demo module
m = torch.nn.ConvTranspose1d(1, 1, 1, stride=5, padding=3)


shape_env = ShapeEnv()
with FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env) as fake_tensor_mode:
    # time_sym = shape_env.create_symbol(100, ConstantSource("T"))
    time_src = ConstantSource("T")
    time_sym = shape_env.create_symbol(100, time_src, dynamic_dim=DimDynamic.DYNAMIC)
    time_sym_ = shape_env.create_symintnode(time_sym, hint=100, source=time_src)

    x = torch.empty((1, 1, time_sym_))
    y = m(x)
    print(y)
    print(f"{y.shape=}")
    out_size = y.shape[-1]
    print(f"{out_size=} {type(out_size)=}")
    assert isinstance(out_size, SymInt)
    print(f"{out_size.node=} {type(out_size.node)=}")
    print(f"{out_size.node.expr=} {type(out_size.node.expr)=}")
    out_sym = out_size.node.expr
    assert isinstance(out_sym, sympy.Expr)
    for t in [1, 100, 45, 1000]:
        print(f"out size for time={t}: {out_sym.xreplace({time_sym: t})}")

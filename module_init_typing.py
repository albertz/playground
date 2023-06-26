"""
https://github.com/rwth-i6/i6_models/pull/21#discussion_r1242054557
"""

from __future__ import annotations
from typing import TypeVar, Generic, Type
from torch import nn
from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    pass


ConfigType = TypeVar("ConfigType", bound=ModelConfiguration)
ModuleType = TypeVar("ModuleType", bound=nn.Module)


@dataclass
class ModuleFactoryV1(Generic[ConfigType, ModuleType]):
    """
    Dataclass for a combination of a Subassembly/Part and the corresponding configuration.
    Also provides a function to construct the corresponding object through this dataclass
    """

    module_class: Type[ModuleType]
    cfg: ConfigType

    def __call__(self) -> ModuleType:
        """Constructs an instance of the given module class"""
        return self.module_class(self.cfg)


@dataclass
class MyModuleConfiguration(ModelConfiguration):
    pass


class MyModule(nn.Module):
    def __init__(self, cfg: MyModuleConfiguration):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        return x


def test():
    mod_factory = ModuleFactoryV1(MyModule, MyModuleConfiguration())
    mod = mod_factory()
    assert isinstance(mod, mod_factory.module_class)

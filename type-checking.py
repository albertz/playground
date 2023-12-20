"""
https://news.ycombinator.com/item?id=38707645
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def agi_model(query: torch.Tensor) -> torch.Tensor:
    """AGI!"""
    # Problem here: torch not imported. Most IDEs don't show a problem here.
    torch.cuda.init()
    ...

    return query


def student_agi_model(query: torch.Tensor) -> torch.Tensor:
    """AGI!"""
    import torch
    torch.cuda.init()
    ...

    return query

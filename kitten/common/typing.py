from typing import Any, TypeAlias

import torch
import torch.nn as nn

# General
Shape: TypeAlias = tuple[int, ...]


def shape_annotation(shape: Shape) -> str:
    return " ".join(map(str, shape))


Device: TypeAlias = str | torch.device

# Logging
Log: TypeAlias = dict[str, Any]
ModuleNamePair: TypeAlias = tuple[nn.Module, str]
ModuleNamePairs: TypeAlias = list[ModuleNamePair]

from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

# General
Shape: TypeAlias = tuple[int, ...]
TorchCompatible: TypeAlias = int | float | bool | complex | torch.Tensor | NDArray[Any]


def shape_annotation(shape: Shape) -> str:
    return " ".join(map(str, shape))


Device: TypeAlias = str | torch.device
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# Logging
Log: TypeAlias = dict[str, Any]
ModuleNamePair: TypeAlias = tuple[nn.Module, str]
ModuleNamePairs: TypeAlias = list[ModuleNamePair]

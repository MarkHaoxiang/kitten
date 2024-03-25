from typing import Any, TypeAlias

import torch
import torch.nn as nn

# General
Size : TypeAlias = tuple[int, ...]
Device: TypeAlias = str | torch.device

# Logging
Log: TypeAlias = dict[str, Any]
ModuleNamePair: TypeAlias = tuple[nn.Module, str]
ModuleNamePairs: TypeAlias = list[ModuleNamePair]
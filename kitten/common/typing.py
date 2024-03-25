from typing import Any, TypeAlias

import torch.nn as nn

# General
Size : TypeAlias = tuple[int, ...]

# Logging
Log: TypeAlias = dict[str, Any]
ModuleNamePair: TypeAlias = tuple[nn.Module, str]
ModuleNamePairs: TypeAlias = list[ModuleNamePair]
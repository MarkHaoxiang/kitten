from collections.abc import Sequence
import math
from typing import TypeVar

import numpy as np
from numpy.random import default_rng, Generator

T = TypeVar('T')
def generate_minibatches(batch: T,
                         mb_size: int,
                         rng: Generator  | None = None
) -> list[tuple[list[int], T]]:
    # RNG 
    if rng is not None:
        _rng = rng
    else:
        _rng = default_rng()
    # Sampled indices
    n = len(batch)
    indices = np.arange(n)
    _rng.shuffle(indices)
    m = math.ceil(n / mb_size)
    return [(
        indices[i*mb_size:min(n, (i+1)*mb_size)],
        batch[indices[i*mb_size:min(n, (i+1)*mb_size)]]        
    ) for i in range(m)]


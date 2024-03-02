import importlib

from .interface import Policy
from .e_greedy import EpsilonGreedyPolicy

is_pink_installed = importlib.util.find_spec("pink") is not None
if is_pink_installed:
    from .colored_noise import ColoredNoisePolicy
else:

    class ColoredNoisePolicy:
        def __init__(self) -> None:
            raise NotImplementedError("Dependency pink-noise-rl required")

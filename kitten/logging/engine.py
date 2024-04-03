from os.path import join
from abc import ABC, abstractmethod
import importlib.util
from typing import Any
import logging
import pickle as pkl

from torch import nn
from kitten.common.typing import Log

has_wandb = importlib.util.find_spec("wandb") is not None
if has_wandb:
    import wandb

logger = logging.Logger(name="KittenLogger")
log = logger.log


class LogEngine(ABC):
    """Provides a hook to perform IO"""

    def __init__(
        self,
        cfg: dict[str, Any],
        path: str,
        name: str,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._path = path
        self._name = name

    @abstractmethod
    def log(self, log: Log) -> None:
        """Log log.
        Args:
            log (Log): log.
        """
        raise NotImplementedError

    def register_model(
        self,
        model: nn.Module,
        idx: int,
        watch: bool = True,
        watch_frequency: int = 1000,
    ) -> None:
        pass

    def close(self) -> None:
        """Release resources"""
        pass


class WandBEngine(LogEngine):
    def __init__(
        self,
        cfg: dict[str, Any],
        path: str,
        name: str,
        *args,
        online: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(cfg, path, name)
        if not has_wandb:
            raise ImportError("WandB not available")

        self._online = online
        wandb.init(
            *args,
            name=self._name,
            config=self._cfg,
            dir=self._path,
            mode="online" if online else "offline",
            **kwargs,
        )

    def log(self, log: Log) -> None:
        wandb.log(log)

    def register_model(
        self,
        model: nn.Module,
        idx: int,
        watch: bool = True,
        watch_frequency: int = 1000,
    ) -> None:
        if watch:
            wandb.watch(model, log_freq=watch_frequency, idx=idx)

    def close(self) -> None:
        wandb.finish()

class DictEngine(LogEngine):
    """ Quick and simple -  saves to a list of dictionaries
    """
    def __init__(self, cfg: Log[str, Any], path: str, name: str) -> None:
        super().__init__(cfg, path, name)
        self.results = []

    def log(self, log: Log[str, Any]) -> None:
        self.results.append(log)

    def close(self) -> None:
        # Store Log
        pkl.dump(self.results, open(join(self._path, "log.pkl"), "wb"))
        return super().close()

engine_registry = {
    "wandb": WandBEngine,
    "dict": DictEngine
}
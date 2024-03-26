from abc import ABC
from kitten.common.typing import Log, ModuleNamePairs


class Loggable(ABC):
    """Interface signalling a module which can be logged"""

    def get_log(self) -> Log:
        """Collect logs for publishing

        Returns:
            Dict: Dictionary of key-value logging pairs
        """
        return {}

    def get_models(self) -> ModuleNamePairs:
        """List of publishable networks with names

        Returns:
            List[Tuple[nn.Module, str]]: list.
        """
        return []

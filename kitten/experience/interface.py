from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar, Generic

from torch import Tensor
from jaxtyping import Bool, Float, Shaped, Integer, jaxtyped
from typeguard import typechecked as typechecker

from kitten.logging import Loggable
from kitten.common.typing import Shape, shape_annotation

InDataType = TypeVar("InDataType")
OutDataType = TypeVar("OutDataType")


class Memory(Loggable, Generic[InDataType, OutDataType], ABC):
    """Abstract storage object storing history"""

    @abstractmethod
    def append(self, data: InDataType, **kwargs) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, **kwargs) -> OutDataType:
        raise NotImplementedError


class Transitions:
    """Represents base Markov Chain Transitions gathered as experience

    # TODO: Support dictionary based observations and actions
    #   - We probably want to generalise out the interface for transitions to allow objects
    #   - And have the current class be a subclass
    # TODO: Write tests
    """

    @jaxtyped(typechecker=typechecker)  # type: ignore[misc]
    def __init__(
        self,
        s_0: Float[Tensor, "..."],
        a: Shaped[Tensor, "..."],
        r: Float[Tensor, "*batch"],
        s_1: Float[Tensor, "..."],
        d: Bool[Tensor, "*batch"],
        t: Bool[Tensor, "*batch"] | None = None,
    ) -> None:
        """Constructs a

        Args:
            s_0 (Float[Tensor, ): Observation before the action
            a (Shaped[Tensor, ): Action
            r (Float[Tensor, ): Reward.
            s_1 (Float[Tensor, ): Observation after the action
            d (Bool[Tensor, ): Termination.
            t (Optional, Bool[Tensor, ): Truncation.
        """
        self._s_0 = s_0
        self._a = a
        self._r = r
        self._s_1 = s_1
        self._d = d
        self._t = t
        self._batch_shape = self._d.shape
        self._batch_dimensions = len(self._batch_shape)
        # Manual validation of observation and action shapes
        assert (
            self._s_0.shape == self._s_1.shape
        ), f"Observation shapes {self._s_0.shape} != {self._s_1.shape}"
        assert (
            self._s_0.shape[: self._batch_dimensions] == self._batch_shape
        ), f"Batch shape of observation is inconsistent"
        self._obs_shape = self._s_0.shape[self._batch_dimensions :]
        assert (
            self._a.shape[: self._batch_dimensions] == self._batch_shape
        ), f"Batch shape of action is inconsistent"
        self._action_shape = self._a.shape[self._batch_dimensions :]
        # Annotations
        self._batch_annotation = shape_annotation(self._batch_shape)
        self._obs_annotation = shape_annotation(self._obs_shape)
        self._action_annotation = shape_annotation(self._action_shape)

    @property
    def shape(self) -> Shape:
        return self._batch_shape

    @property
    def s_0(self) -> Float[Tensor, "{self._batch_annotation} {self._obs_annotation}"]:
        return self._s_0
    @jaxtyped(typechecker=typechecker) # type: ignore[misc]
    @s_0.setter
    def s_0(self, value: Float[Tensor, "{self._batch_annotation} {self._obs_annotation}"]):
        self._s_0 = value

    @property
    def a(self) -> Float[Tensor, "{self._batch_annotation} {self._action_annotation}"]:
        return self._a
    @jaxtyped(typechecker=typechecker) # type: ignore[misc]
    @a.setter
    def a(self, value: Float[Tensor, "{self._batch_annotation} {self._action_annotation}"]):
        self._a = value

    @property
    def s_1(self) -> Float[Tensor, "{self._batch_annotation} {self._obs_annotation}"]:
        return self._s_1
    @jaxtyped(typechecker=typechecker) # type: ignore[misc]
    @s_1.setter
    def s_1(self, value: Float[Tensor, "{self._batch_annotation} {self._obs_annotation}"]):
        self._s_1 = value

    @property
    def r(self) -> Float[Tensor, "{self._batch_annotation}"]:
        return self._r
    @jaxtyped(typechecker=typechecker) # type: ignore[misc]
    @r.setter
    def r(self, value: Float[Tensor, "{self._batch_annotation}"]):
        self._r = value

    @property
    def d(self) -> Bool[Tensor, "{self._batch_annotation}"]:
        return self._d
    @jaxtyped(typechecker=typechecker) # type: ignore[misc]
    @d.setter
    def d(self, value: Bool[Tensor, "{self._batch_annotation}"]):
        self._d = value

    @property
    def t(self) -> Bool[Tensor, "{self._batch_annotation}"]:
        if self._t is None:
            raise ValueError("Truncation has not been provided")
        return self._t
    @jaxtyped(typechecker=typechecker) # type: ignore[misc]
    @t.setter
    def t(self, value: Bool[Tensor, "{self._batch_annotation}"]):
        self._t = value

    def __iter__(self) -> Iterator[Tensor]:
        return iter((self.s_0, self.a, self.r, self.s_1, self.t))


@jaxtyped(typechecker=typechecker)
@dataclass()
class AuxiliaryMemoryData:
    weights: Float[Tensor, "*batch"]
    random: Float[Tensor, "*batch"]
    indices: Integer[Tensor, "*batch"]

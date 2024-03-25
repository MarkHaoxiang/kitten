from collections import namedtuple
from collections.abc import Iterator
from dataclasses import dataclass

from torch import Tensor
from jaxtyping import Bool, Float, Shaped, Integer, jaxtyped
from typeguard import typechecked as typechecker

from kitten.common.typing import Size


def shape_annotation(shape: Size) -> str:
    return " ".join(map(str, shape))


class Transitions:
    """Represents base Markov Chain Transitions gathered as experinece

    # TODO: Support dictionary based observations and actions
    #   - We probably want to generalise out the interface for transitions to allow objects
    #   - And have the current class be a subclass
    # TODO: Write tests
    """

    @jaxtyped(typechecker=typechecker)
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
    def shape(self) -> Size:
        return self._batch_shape

    @property
    def s_0(self) -> Float[Tensor, "{self._batch_annotation} {self._obs_annotation}"]:
        return self._s_0

    @property
    def a(self) -> Float[Tensor, "{self._batch_annotation} {self._action_annotation}"]:
        return self._a

    @property
    def s_1(self) -> Float[Tensor, "{self._batch_annotation} {self._obs_annotation}"]:
        return self._s_1

    @property
    def r(self) -> Float[Tensor, "{self._batch_annotation}"]:
        return self._r

    @property
    def d(self) -> Float[Tensor, "{self._batch_annotation}"]:
        return self._d

    @property
    def t(self) -> Float[Tensor, "{self._batch_annotation}"]:
        return self._d

    def __iter__(self) -> Iterator[Tensor]:
        return iter((self.s_0, self.a, self.r, self.s_1, self.t))


@jaxtyped(typechecker=typechecker)
@dataclass()
class AuxiliaryMemoryData:
    weights: Float[Tensor, "*batch"] | float
    random: Float[Tensor, "*batch"]
    indices: Integer[Tensor, "*batch"]

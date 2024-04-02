from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Generic
import types

IN = TypeVar("IN")
OUT = TypeVar("OUT")
T = TypeVar("T")


class Transform(Generic[IN, OUT], ABC):
    """General data transformations

    A wrapper around a function with utilities to bind other functions and methods

    ... class ExampleTransform(Transform):
    ...     def __call__(self, data: Any) -> Any:
    ...     return data + "_example_transform"
    ...
    ... class ExampleClass:
    ...     def example(self):
    ...         return ("example_class")
    ...
    ... transform = ExampleTransform()
    ... cls = ExampleClass()
    ... transform.prepend(cls.example)
    ... cls.example()

    Prints out 'example_class_example_transform'

    #TODO: Add ability to skip transformations after composition
    """

    def __init__(self) -> None:
        super().__init__()
        self._enabled = True

    def __call__(self, data: IN, *args, **kwargs) -> IN | OUT:
        if not self._enabled:
            return data
        else:
            return self.transform(data, *args, **kwargs)

    @abstractmethod
    def transform(self, data: IN, *args, **kwargs) -> OUT:
        """Applies a transformation on data

        Args:
            data (Any): Input data to be transformed.
        """
        raise NotImplementedError

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def prepend(
        self,
        f: Callable[[T], IN],
        bind_method_type: bool = True,
    ) -> Callable[[T], OUT]:
        """A function continuation"""
        if isinstance(f, types.MethodType) and bind_method_type:

            def _g(_, *args, **kwargs):
                return self(f(*args, **kwargs))

            f.__self__.__setattr__(f.__name__, types.MethodType(_g, f.__self__))
            return f.__self__.__getattribute__(f.__name__)
        elif isinstance(f, types.FunctionType) or isinstance(f, types.MethodType):

            def g(*args, **kwargs):
                return self(f(*args, **kwargs))

            return g
        else:
            raise ValueError(f"{f} is not a valid function target")

    def append(
        self,
        f: Callable[[OUT], T],
        bind_method_type: bool = True,
    ) -> Callable[[IN], T]:
        """A function continuation"""
        if isinstance(f, types.MethodType) and bind_method_type:

            def _g(_, data: IN, *args, **kwargs) -> T:
                return f(self(data, *args, **kwargs))

            f.__self__.__setattr__(f.__name__, types.MethodType(_g, f.__self__))
            return f.__self__.__getattribute__(f.__name__)
        elif isinstance(f, types.FunctionType) or isinstance(f, types.MethodType):

            def g(*args, **kwargs):
                return f(self(*args, **kwargs))

            return g
        else:
            raise ValueError(f"{f} is not a valid function target")


class Identity(Generic[T], Transform[T, T]):
    def transform(self, data: T, *args, **kwargs) -> T:
        return data

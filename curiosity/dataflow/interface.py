from abc import ABC, abstractmethod
from typing import Any, Callable, Union
import types

from curiosity.experience import AuxiliaryMemoryData, Transition

class Transform(Callable, ABC):
    """ General data transformations

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

    def __init__(self):
        self._enabled = True

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if not self._enabled:
            return data
        else:
            return self.transform(data, *args, **kwargs)

    @abstractmethod
    def transform(self, data: Any, *args, **kwargs) -> Any:
        """ Applies a transformation on data

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

    def prepend(self,
               f: Union[types.FunctionType, types.MethodType],
               bind_method_type: bool=True):
        """ A function continuation
        """
        if isinstance(f, types.MethodType) and bind_method_type:
            def g(_, *args, **kwargs):
                return self(f(*args, **kwargs))
            f.__self__.__setattr__(f.__name__, types.MethodType(g, f.__self__))
        elif isinstance(f, types.FunctionType) or isinstance(f, types.MethodType):
            def g(*args, **kwargs):
                return self(f(*args, **kwargs))
            return g
        else:
            raise ValueError(f"{f} is not a valid function target")

    def append(self,
                f: Union[types.FunctionType, types.MethodType],
                bind_method_type: bool=True):
        """ A function continuation
        """
        if isinstance(f, types.MethodType) and bind_method_type:
            def g(_, *args, **kwargs):
                return f(self(*args, **kwargs))
            f.__self__.__setattr__(f.__name__, types.MethodType(g, f.__self__))
        elif isinstance(f, types.FunctionType) or isinstance(f, types.MethodType):
            def g(*args, **kwargs):
                return f(self(*args, **kwargs))
            return g
        else:
            raise ValueError(f"{f} is not a valid function target")

class Identity(Transform):
    def transform(self, data: Any, *args, **kwargs) -> Any:
        return data

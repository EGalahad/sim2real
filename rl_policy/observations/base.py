import inspect
import numpy as np
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from rl_policy.base_policy import BasePolicy

class _RegistryMixin:
    
    def __init_subclass__(cls) -> None:
        """Put the subclass in the global registry"""
        if not hasattr(cls, 'registry'):
            cls.registry = {}
            
        cls_name = cls.__name__
        cls._file = inspect.getfile(cls)
        cls._line = inspect.getsourcelines(cls)[1]
        if cls_name.startswith("_"):
            return
        if cls_name not in cls.registry:
            cls.registry[cls_name] = cls
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")


class Observation(_RegistryMixin):
    def __init__(self, env: "BasePolicy", **kwargs):
        self.env = env
        self.state_processor = env.state_processor
    
    def reset(self):
        pass

    def update(self, data: Dict[str, Any]) -> None:
        pass

    def compute(self) -> np.ndarray:
        raise NotImplementedError

class ObsGroup:
    def __init__(
        self,
        name: str,
        funcs: Dict[str, Observation],
    ):
        self.name = name
        self.funcs = funcs

    def compute(self) -> np.ndarray:
        # torch.compiler.cudagraph_mark_step_begin()
        output = self._compute()
        return output
    
    def _compute(self) -> np.ndarray:
        # update only if outdated
        tensors = [func.compute() for func in self.funcs.values()]
        # for func, tensor in zip(self.funcs.values(), tensors):
        #     print(func.__class__.__name__, tensor.shape)
        return np.concatenate(tensors, axis=-1)


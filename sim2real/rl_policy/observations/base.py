import inspect
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from sim2real.rl_policy.base_policy import BasePolicy

class _RegistryMixin:
    namespace: str | Sequence[str] | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Put the subclass in the global registry"""
        if not hasattr(cls, 'registry'):
            cls.registry = {}

        namespace = kwargs.pop("namespace", None)
        if namespace is None:
            namespace = getattr(cls, "namespace", None)

        cls_name = cls.__name__
        cls._file = inspect.getfile(cls)
        cls._line = inspect.getsourcelines(cls)[1]
        if cls_name.startswith("_"):
            return

        if namespace is None:
            registry_keys = [cls_name]
        else:
            namespaces = [namespace] if isinstance(namespace, str) else list(namespace)
            registry_keys = [f"{namespace}.{cls_name}" for namespace in namespaces]

        for registry_key in registry_keys:
            if registry_key not in cls.registry:
                cls.registry[registry_key] = cls
            else:
                conflicting_cls = cls.registry[registry_key]
                location = f"{conflicting_cls._file}:{conflicting_cls._line}"
                raise ValueError(f"Term {registry_key} already registered in {location}")


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

    @classmethod
    def resolve(cls, obs_key: str) -> type["Observation"]:
        if obs_key in cls.registry:
            return cls.registry[obs_key]

        if "." not in obs_key:
            matches: Dict[type["Observation"], list[str]] = {}
            for registered_key, registered_cls in cls.registry.items():
                if registered_key.split(".")[-1] == obs_key:
                    matches.setdefault(registered_cls, []).append(registered_key)
            if len(matches) == 1:
                return next(iter(matches))
            if len(matches) > 1:
                candidates = sorted(key for keys in matches.values() for key in keys)
                raise ValueError(
                    f"Observation target {obs_key!r} is ambiguous; use one of {candidates}"
                )

        available = sorted(cls.registry)
        raise ValueError(f"Observation target {obs_key!r} not found. Available targets: {available}")

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

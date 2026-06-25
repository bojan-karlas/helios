"""Model base classes + a small registry.

Every trainable/inference model in HELIOS implements the same tiny contract so
the pipeline can fit, persist, load, and apply it uniformly:

    model.fit(inputs, **params) -> self
    model.predict(inputs) -> outputs
    model.save(path)        (pickle by default)
    Model.load(path)        (classmethod)

Register concrete models with ``@register_model("name")`` so they can be looked
up by id (e.g. from config).
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", bound="Model")


class Model(ABC):
    """Minimal fit/predict/save/load contract shared by all HELIOS models."""

    #: Set by ``@register_model``; the registry key for this class.
    name: str = ""

    @abstractmethod
    def fit(self, inputs: Any, **params: Any) -> Model:
        """Fit on training inputs; return ``self``."""

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Produce predictions for the given inputs."""

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: {type(obj).__name__}")
        return obj


_REGISTRY: dict[str, type[Model]] = {}


def register_model(name: str) -> Callable[[type[Model]], type[Model]]:
    def deco(klass: type[Model]) -> type[Model]:
        klass.name = name
        _REGISTRY[name] = klass
        return klass

    return deco


def get_model(name: str) -> type[Model]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(_REGISTRY)}") from exc


def registered_models() -> list[str]:
    return sorted(_REGISTRY)

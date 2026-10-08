"""Name -> factory registries behind the pluggable parts of the package.

Every swappable component family (models, trackers, team embedders) keeps one
``Registry``. Backends register themselves with a decorator, and callers pick a
backend by name, which is also how YAML configs refer to them::

    @MODELS.register("yolo")
    class YOLODetector(Model): ...

    model = MODELS.create("yolo", weights="yolo11n.pt")
"""

from collections.abc import Callable
from typing import Any


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register[F: Callable[..., Any]](self, name: str) -> Callable[[F], F]:
        def decorator(factory: F) -> F:
            if name in self._factories:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._factories[name]
        except KeyError:
            raise ValueError(
                f"Unknown {self.kind} {name!r}. Available: {', '.join(self.names())}"
            ) from None

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"

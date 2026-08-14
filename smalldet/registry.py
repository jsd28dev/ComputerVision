"""A minimal name-to-factory registry.

Every extension point in the project (transforms, detector families,
optimizers, schedulers, finetuning strategies, callbacks) is a `Registry`. That
keeps the YAML surface honest: a config value like ``optimizer.name: sgd`` is
only ever a key lookup, so an unknown name fails immediately with the list of
valid names instead of failing three layers down with an AttributeError.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised when a registry lookup or registration fails."""

    def __str__(self) -> str:  # KeyError repr adds quotes; this reads better.
        return self.args[0] if self.args else ""


class Registry(Generic[T]):
    """Maps a lowercase string key to a factory callable."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Callable[..., T]] = {}

    def register(
        self, key: str, factory: Optional[Callable[..., T]] = None
    ) -> Callable[..., T]:
        """Register ``factory`` under ``key``, or return a decorator that does.

        Usable both as ``registry.register("sgd", build_sgd)`` and as a
        ``@registry.register("sgd")`` decorator.
        """
        normalized = self._normalize(key)

        def _register(func: Callable[..., T]) -> Callable[..., T]:
            if normalized in self._entries:
                raise RegistryError(
                    f"{self.kind} {normalized!r} is already registered"
                )
            self._entries[normalized] = func
            return func

        if factory is None:
            return _register
        return _register(factory)

    def get(self, key: str) -> Callable[..., T]:
        normalized = self._normalize(key)
        try:
            return self._entries[normalized]
        except KeyError:
            raise RegistryError(
                f"unknown {self.kind} {key!r}; available: {', '.join(self.names())}"
            ) from None

    def build(self, key: str, *args: object, **kwargs: object) -> T:
        return self.get(key)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._normalize(key) in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self._entries)} entries)"

    @staticmethod
    def _normalize(key: str) -> str:
        if not isinstance(key, str) or not key.strip():
            raise RegistryError(f"registry keys must be non-empty strings, got {key!r}")
        return key.strip().lower()

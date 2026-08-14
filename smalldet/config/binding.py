"""Bind plain dictionaries (parsed YAML) onto typed dataclasses.

This is the only place that turns untrusted input into config objects, so it is
also the only place that has to be strict. Two rules earn their keep:

* **Unknown keys are errors.** A silently-ignored ``bacth_size`` typo means a
  training run that quietly uses the default and wastes a GPU-day.
* **Errors name their path.** ``model.anchors.base_sizes[2]`` is actionable;
  ``invalid literal for int()`` is not.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Mapping, Sequence, TypeVar, Union

T = TypeVar("T")

_NONE_TYPE = type(None)
_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


class ConfigError(ValueError):
    """Raised when a config document does not match the schema."""


def from_dict(cls: type, data: Any, *, path: str = "config") -> Any:
    """Recursively build an instance of dataclass ``cls`` from ``data``."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path}: expected a mapping for {cls.__name__}, got {_describe(data)}"
        )

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Valid keys: {', '.join(sorted(fields))}"
        )

    kwargs = {
        name: _coerce(hints[name], value, f"{path}.{name}")
        for name, value in data.items()
        if name in fields
    }
    try:
        return cls(**kwargs)
    except TypeError as exc:  # a required field was missing
        raise ConfigError(f"{path}: {exc}") from exc


def to_dict(obj: Any) -> Any:
    """Inverse of :func:`from_dict`, for round-tripping a config back to YAML."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {key: to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    return obj


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(annotation)

    if annotation is Any:
        return value

    if origin is Union:  # includes Optional[X]
        args = typing.get_args(annotation)
        if value is None and _NONE_TYPE in args:
            return None
        # Try each member in declaration order; the first that binds wins.
        errors = []
        for arg in args:
            if arg is _NONE_TYPE:
                continue
            try:
                return _coerce(arg, value, path)
            except (ConfigError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise ConfigError(
            f"{path}: {_describe(value)} does not match {_name(annotation)}"
            + (f" ({errors[0]})" if errors else "")
        )

    if origin in (list, Sequence):
        (item_type,) = typing.get_args(annotation) or (Any,)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {_describe(value)}")
        return [_coerce(item_type, item, f"{path}[{i}]") for i, item in enumerate(value)]

    if origin is tuple:
        args = typing.get_args(annotation)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {_describe(value)}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _coerce(args[0], item, f"{path}[{i}]") for i, item in enumerate(value)
            )
        if len(args) != len(value):
            raise ConfigError(
                f"{path}: expected {len(args)} item(s), got {len(value)}"
            )
        return tuple(
            _coerce(arg, item, f"{path}[{i}]")
            for i, (arg, item) in enumerate(zip(args, value))
        )

    if origin in (dict, Mapping):
        key_type, value_type = typing.get_args(annotation) or (Any, Any)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {_describe(value)}")
        return {
            _coerce(key_type, key, f"{path}.<key>"): _coerce(
                value_type, item, f"{path}.{key}"
            )
            for key, item in value.items()
        }

    if dataclasses.is_dataclass(annotation):
        return from_dict(annotation, value, path=path)

    return _coerce_scalar(annotation, value, path)


def _coerce_scalar(annotation: Any, value: Any, path: str) -> Any:
    if annotation is bool:
        if isinstance(value, bool):
            return value
        # CLI overrides arrive as strings; accept the usual spellings.
        if isinstance(value, str):
            if value.strip().lower() in _TRUE:
                return True
            if value.strip().lower() in _FALSE:
                return False
        raise ConfigError(f"{path}: expected a boolean, got {_describe(value)}")

    if annotation is int:
        # bool is a subclass of int, but `epochs: true` is always a mistake.
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected an int, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"{path}: expected an int, got {_describe(value)}")

    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a float, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # YAML 1.1 parses `1e10` as a string, so floats often arrive as text.
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"{path}: expected a float, got {_describe(value)}")

    if annotation is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected a string, got {_describe(value)}")

    if isinstance(annotation, type) and isinstance(value, annotation):
        return value

    raise ConfigError(
        f"{path}: cannot bind {_describe(value)} to {_name(annotation)}"
    )


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _describe(value: Any) -> str:
    return f"{type(value).__name__} {value!r}"

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

class SilentDegradationError(RuntimeError):
    pass

def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value

class UnsLog:
    def __init__(self, allow_degradation: bool = True) -> None:
        self._data: dict[str, Any] = {}
        self._superseded: dict[str, list[Any]] = {}
        self._allow_degradation = allow_degradation

    def record(
        self,
        key: str,
        value: Any,
        section: str | None = None,
        overwrite: bool = False,
    ) -> None:
        target = self._data
        if section is not None:
            target = target.setdefault(section, {})
        if key in target and not overwrite:
            raise SilentDegradationError(
                f"{key!r} was already recorded as {target[key]!r}; pass "
                f"overwrite=True if replacing it is intended"
            )
        if key in target:
            self._superseded.setdefault(key, []).append(target[key])
        target[key] = _jsonify(value)

    def degrade(self, what: str, reason: str, fallback: str) -> None:
        if not self._allow_degradation:
            raise SilentDegradationError(
                f"refusing to degrade {what!r}: {reason}. The fallback would "
                f"have been {fallback!r}; enable it explicitly to proceed."
            )
        section = self._data.setdefault("degradations", {})
        section[what] = {"reason": reason, "fallback": fallback}

    def copy(self) -> "UnsLog":
        import copy as _copy

        clone = UnsLog(allow_degradation=self._allow_degradation)
        clone._data = _copy.deepcopy(self._data)
        clone._superseded = _copy.deepcopy(self._superseded)
        return clone

    def as_dict(self) -> dict[str, Any]:
        out = dict(self._data)
        if self._superseded:
            out["_superseded"] = {k: _jsonify(v) for k, v in self._superseded.items()}
        return out

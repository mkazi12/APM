"""Small contract between validated home tools and local device protocols."""
from dataclasses import dataclass
from typing import Any, Protocol


class BackendError(RuntimeError):
    """A device operation failed; messages must not contain credentials."""


@dataclass(frozen=True)
class DeviceState:
    state: str
    # Write acknowledgement is distinct from observing the requested final state.
    accepted: bool = False


class DeviceAdapter(Protocol):
    def validate_target(self, kind: str, target: dict[str, Any]) -> None: ...
    def discover(self) -> list[dict[str, Any]]: ...
    def get_state(self, kind: str, target: dict[str, Any]) -> DeviceState: ...
    def set_state(self, kind: str, target: dict[str, Any], state: str) -> DeviceState: ...
    def close(self) -> None: ...

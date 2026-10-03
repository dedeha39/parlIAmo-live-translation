"""Audio capture, playback and device discovery.

On Windows the same physical device is exposed once per host API (MME,
DirectSound, WASAPI, WDM-KS) with wildly different latency characteristics, so
picking a device is really picking a *(device, host API)* pair. Everything in
this package treats that pair as the unit of selection and prefers WASAPI,
which is the only one of the four that gets us into the sub-30 ms range.
"""

from .devices import (
    AudioDevice,
    DeviceResolutionError,
    candidates,
    describe_devices,
    list_devices,
    open_with_fallback,
    preferred_hostapi,
    resolve_device,
)

__all__ = [
    "AudioDevice",
    "DeviceResolutionError",
    "describe_devices",
    "list_devices",
    "open_with_fallback",
    "preferred_hostapi",
    "candidates",
    "resolve_device",
]

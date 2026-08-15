"""
Shared input/output device resolution for voice.py and the ha-addon's
wake_word.py.

Exists because relying on sounddevice/PortAudio's OS default device
doesn't work inside the ha-addon's Docker container: PortAudio raises
"Error querying device -1" (no default device) there even with /dev/snd
passed through and the USB audio hardware confirmed present via lsusb.
This affects the default *output* device exactly the same way it affects
the default input device, since it's the same "no default device index"
problem - both directions resolve a device explicitly by name instead of
trusting the default.
"""

import sounddevice as sd

DEFAULT_DEVICE_NAME_SUBSTRING = "USB Audio"


def _resolve_device(name_substring: str, channel_key: str) -> int:
    devices = sd.query_devices()
    needle = name_substring.lower()

    for index, device in enumerate(devices):
        if device[channel_key] > 0 and needle in device["name"].lower():
            return index

    available = "\n".join(
        f"  [{i}] {d['name']} ({channel_key}={d[channel_key]})"
        for i, d in enumerate(devices)
    )
    direction = "input" if channel_key == "max_input_channels" else "output"
    raise RuntimeError(
        f"No {direction} device found with '{name_substring}' in its name. "
        f"Available devices:\n{available}"
    )


def resolve_input_device(name_substring: str = DEFAULT_DEVICE_NAME_SUBSTRING) -> int:
    """
    Returns the index of the first input-capable device whose name
    contains name_substring (case-insensitive). Raises RuntimeError,
    listing every available device's name, if nothing matches.
    """
    return _resolve_device(name_substring, "max_input_channels")


def resolve_output_device(name_substring: str = DEFAULT_DEVICE_NAME_SUBSTRING) -> int:
    """
    Returns the index of the first output-capable device whose name
    contains name_substring (case-insensitive). Raises RuntimeError,
    listing every available device's name, if nothing matches.
    """
    return _resolve_device(name_substring, "max_output_channels")

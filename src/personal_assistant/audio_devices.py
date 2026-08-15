"""
Shared input-device resolution for voice.py and the ha-addon's
wake_word.py.

Exists because relying on sounddevice/PortAudio's OS default input device
doesn't work inside the ha-addon's Docker container: PortAudio raises
"Error querying device -1" (no default device) there even with /dev/snd
passed through and the USB audio hardware confirmed present via lsusb.
Both call sites resolve a device explicitly by name instead of trusting
the default.
"""

import sounddevice as sd

DEFAULT_DEVICE_NAME_SUBSTRING = "USB Audio"


def resolve_input_device(name_substring: str = DEFAULT_DEVICE_NAME_SUBSTRING) -> int:
    """
    Returns the index of the first input-capable device whose name
    contains name_substring (case-insensitive). Raises RuntimeError,
    listing every available device's name, if nothing matches.
    """
    devices = sd.query_devices()
    needle = name_substring.lower()

    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0 and needle in device["name"].lower():
            return index

    available = "\n".join(
        f"  [{i}] {d['name']} (max_input_channels={d['max_input_channels']})"
        for i, d in enumerate(devices)
    )
    raise RuntimeError(
        f"No input device found with '{name_substring}' in its name. "
        f"Available devices:\n{available}"
    )

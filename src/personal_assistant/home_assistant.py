"""
Thin wrapper around Home Assistant's REST API.

Every function here does one HTTP call and returns a plain string —
that string is what gets fed back to Claude as a tool_result, so it
needs to be something a language model can read directly, not a raw
dict or a Response object.

Docs: https://developers.home-assistant.io/docs/api/rest/
"""

import os
import re
import requests
from dotenv import load_dotenv

# Load .env here rather than relying on whoever imports this module to have
# already called load_dotenv() first. HA_URL/HA_TOKEN are read immediately
# below, at import time - if the .env file hasn't been loaded yet by then,
# HA_URL silently falls back to its hardcoded default instead of picking up
# your actual configuration. load_dotenv() is safe to call more than once
# (e.g. if assistant.py also calls it), so this doesn't conflict with that.
load_dotenv()

HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

# Every HA request goes through this. Keeping requests.exceptions handling
# in one place means turn_on_light/turn_off_light/get_entity_state/list_entities
# don't each need their own try/except block.
def _request(method: str, path: str, json: dict | None = None) -> tuple[bool, object]:
    """
    Makes an HTTP request to the Home Assistant API.
    Returns (success, data). On failure, data is a human-readable error string.
    """
    url = f"{HA_URL}/api{path}"
    try:
        response = requests.request(method, url, headers=HEADERS, json=json, timeout=10)
    except requests.exceptions.ConnectTimeout:
        return False, f"Timed out connecting to Home Assistant at {HA_URL}. Is it running and reachable?"
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to Home Assistant at {HA_URL}. Check HA_URL and that Home Assistant is running."
    except requests.exceptions.RequestException as e:
        return False, f"Request to Home Assistant failed: {e}"

    if response.status_code == 401:
        return False, "Home Assistant rejected the request: invalid or expired HA_TOKEN."
    if response.status_code == 404:
        return False, f"Home Assistant returned 404 for {path}. Check the entity_id is correct."
    if not response.ok:
        return False, f"Home Assistant returned HTTP {response.status_code}: {response.text[:300]}"

    try:
        return True, response.json()
    except ValueError:
        return True, response.text


def list_entities() -> str:
    """
    Calls GET /api/states, which returns every entity Home Assistant knows
    about along with its current state and attributes. This is what lets
    Claude figure out "bedroom light" -> light.bedroom_lamp without you
    ever having to hardcode entity IDs.
    """
    ok, data = _request("GET", "/states")
    if not ok:
        return f"Error listing entities: {data}"

    entities = []
    for entity in data:
        entity_id = entity.get("entity_id", "")
        domain = entity_id.split(".")[0] if "." in entity_id else "unknown"
        friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)
        state = entity.get("state", "unknown")
        entities.append(f"{entity_id} | {friendly_name} | domain={domain} | state={state}")

    if not entities:
        return "Home Assistant returned no entities."

    return "\n".join(entities)


def get_entity_state(entity_id: str) -> str:
    """
    Calls GET /api/states/<entity_id> to check the current state of any
    single entity (a light, a sensor, a switch, whatever).
    """
    ok, data = _request("GET", f"/states/{entity_id}")
    if not ok:
        return f"Error getting state for '{entity_id}': {data}"

    friendly_name = data.get("attributes", {}).get("friendly_name", entity_id)
    state = data.get("state", "unknown")
    attributes = data.get("attributes", {})
    return f"{friendly_name} ({entity_id}) is currently '{state}'. Attributes: {attributes}"


def turn_on_light(entity_id: str) -> str:
    """
    Calls POST /api/services/light/turn_on with the entity_id in the body.
    This is a "service call" - HA's mechanism for actually doing something,
    as opposed to /states which is read-only.
    """
    ok, data = _request("POST", "/services/light/turn_on", json={"entity_id": entity_id})
    if not ok:
        return f"Error turning on '{entity_id}': {data}"
    return f"Turned on {entity_id}."


def turn_off_light(entity_id: str) -> str:
    """
    Calls POST /api/services/light/turn_off - the mirror image of
    turn_on_light above.
    """
    ok, data = _request("POST", "/services/light/turn_off", json={"entity_id": entity_id})
    if not ok:
        return f"Error turning off '{entity_id}': {data}"
    return f"Turned off {entity_id}."


# ---------------------------------------------------------------------------
# Battery monitor support (see battery_monitor.py)
# ---------------------------------------------------------------------------
_BATTERY_DEVICE_CLASS = "battery"
_CHARGING_DEVICE_CLASS = "battery_charging"


def discover_battery_entities() -> tuple[str | None, str | None]:
    """
    Auto-discovers the phone's battery-level sensor (a `sensor.*` entity
    with attributes.device_class == "battery") and, if present, a matching
    `binary_sensor.*` for charging state (device_class == "battery_charging")
    on the same device - this is what Home Assistant's mobile app
    integration typically exposes, and we never want to hardcode an
    entity_id that could differ per install.

    BATTERY_ENTITY_ID / BATTERY_CHARGING_ENTITY_ID in .env override either
    half of auto-detection, for when none is found or the wrong one is
    picked (multiple devices reporting battery, unusual naming, etc.) -
    whatever this function finds along the way is printed so you know
    exactly what to paste into .env.

    Returns (battery_entity_id, charging_entity_id). Either may be None:
    no charging entity isn't fatal to the caller (charging state is just
    treated as unknown), but no battery entity means there's nothing to
    monitor at all.
    """
    battery_override = os.getenv("BATTERY_ENTITY_ID") or None
    charging_override = os.getenv("BATTERY_CHARGING_ENTITY_ID") or None

    ok, data = _request("GET", "/states")
    if not ok:
        print(f"[battery discovery] Could not list entities: {data}")
        return battery_override, charging_override

    battery_sensors = []    # [(entity_id, friendly_name), ...]
    charging_sensors = []
    for entity in data:
        entity_id = entity.get("entity_id", "")
        domain = entity_id.split(".")[0] if "." in entity_id else ""
        attributes = entity.get("attributes", {})
        device_class = attributes.get("device_class")
        friendly_name = attributes.get("friendly_name", entity_id)

        if domain == "sensor" and device_class == _BATTERY_DEVICE_CLASS:
            battery_sensors.append((entity_id, friendly_name))
        elif domain == "binary_sensor" and device_class == _CHARGING_DEVICE_CLASS:
            charging_sensors.append((entity_id, friendly_name))

    battery_entity_id = battery_override
    if not battery_entity_id:
        if len(battery_sensors) == 1:
            battery_entity_id = battery_sensors[0][0]
        elif not battery_sensors:
            print("[battery discovery] No battery-level sensor (device_class: battery) found.")
        else:
            print(
                "[battery discovery] Multiple battery sensors found - set "
                "BATTERY_ENTITY_ID in .env to pick one:"
            )
            for entity_id, friendly_name in battery_sensors:
                print(f"  {entity_id}  ({friendly_name})")

    charging_entity_id = charging_override
    if not charging_entity_id and battery_entity_id and charging_sensors:
        # Match by shared device name, e.g. "Pixel 7 Battery" (sensor) and
        # "Pixel 7 Battery Charging" (binary_sensor) - strip the battery
        # sensor's friendly name down to the part before "battery" and look
        # for a charging sensor whose name starts with it.
        battery_friendly = next(
            (name for eid, name in battery_sensors if eid == battery_entity_id), ""
        )
        device_name = re.sub(r"\s*battery.*$", "", battery_friendly, flags=re.IGNORECASE).strip()
        matches = [
            eid for eid, name in charging_sensors
            if device_name and name.lower().startswith(device_name.lower())
        ]
        if len(matches) == 1:
            charging_entity_id = matches[0]
        elif len(charging_sensors) == 1:
            # Only one charging sensor exists at all - reasonable to assume
            # it belongs to this device even if the naming didn't line up.
            charging_entity_id = charging_sensors[0][0]
        else:
            print(
                "[battery discovery] Multiple charging sensors found - set "
                "BATTERY_CHARGING_ENTITY_ID in .env to pick one (optional - "
                "charging state just won't be considered if left unset):"
            )
            for entity_id, friendly_name in charging_sensors:
                print(f"  {entity_id}  ({friendly_name})")

    return battery_entity_id, charging_entity_id


def get_battery_status(
    battery_entity_id: str, charging_entity_id: str | None
) -> tuple[float | None, bool | None]:
    """
    Reads the current battery percentage and, if a charging entity was
    found, whether it's currently charging. Returns (level, is_charging) -
    either may be None if that entity's state is missing or unreadable
    (e.g. HA briefly reports "unavailable"), which callers should treat as
    "try again next poll" rather than an error.
    """
    ok, data = _request("GET", f"/states/{battery_entity_id}")
    level = None
    if ok:
        try:
            level = float(data.get("state"))
        except (TypeError, ValueError):
            pass
    else:
        print(f"[battery discovery] Could not read battery level: {data}")

    is_charging = None
    if charging_entity_id:
        ok, data = _request("GET", f"/states/{charging_entity_id}")
        if ok:
            state = str(data.get("state", "")).lower()
            if state in ("on", "off"):
                is_charging = state == "on"
        else:
            print(f"[battery discovery] Could not read charging state: {data}")

    return level, is_charging

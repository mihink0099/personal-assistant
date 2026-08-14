"""
Thin wrapper around Home Assistant's REST API.

Every function here does one HTTP call and returns a plain string —
that string is what gets fed back to Claude as a tool_result, so it
needs to be something a language model can read directly, not a raw
dict or a Response object.

Docs: https://developers.home-assistant.io/docs/api/rest/
"""

import os
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

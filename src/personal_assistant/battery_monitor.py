"""
Background phone-battery monitor.

Polls a Home Assistant battery sensor (auto-discovered via
home_assistant.discover_battery_entities()) on its own daemon thread and
speaks a short, unprompted announcement on meaningful transitions - dropping
below 20% while not charging, or reaching 100% - rather than on every poll.
Started from assistant.main(); announcements go through voice.speak(), which
takes voice.PLAYBACK_LOCK around actual playback, so this never talks over
the main loop's own speech - it just waits its turn.
"""

import threading
import time

from . import home_assistant as ha
from . import voice

POLL_INTERVAL_SECONDS = 5 * 60
LOW_BATTERY_THRESHOLD = 20.0
FULL_BATTERY_LEVEL = 100.0

# "low", "full", or None (neither) - which announcement, if any, is
# currently active. Only a *change* in this gets spoken, so the same
# announcement isn't repeated on every poll while the phone sits below 20%
# or at 100%.
_last_announced_state: str | None = None


def _announce(text: str) -> None:
    print(f"[battery monitor] {text}")
    try:
        voice.speak(text)
    except Exception as e:
        print(f"[battery monitor] Voice announcement failed: {e}")


def _check_once(battery_entity_id: str, charging_entity_id: str | None) -> None:
    global _last_announced_state

    level, is_charging = ha.get_battery_status(battery_entity_id, charging_entity_id)
    if level is None:
        return  # unavailable/unreadable this poll - just try again next time

    if level < LOW_BATTERY_THRESHOLD and not is_charging:
        current_state = "low"
    elif level >= FULL_BATTERY_LEVEL:
        current_state = "full"
    else:
        current_state = None

    if current_state == _last_announced_state:
        return

    if current_state == "low":
        _announce("Heads up, your phone's under 20%.")
    elif current_state == "full":
        _announce("Your phone's fully charged.")

    _last_announced_state = current_state


def _run(battery_entity_id: str, charging_entity_id: str | None) -> None:
    while True:
        try:
            _check_once(battery_entity_id, charging_entity_id)
        except Exception as e:
            print(f"[battery monitor] Poll failed: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


def start() -> threading.Thread | None:
    """
    Discovers the battery entity (and its charging-state pair, if any) and
    starts the polling thread. Returns the Thread, or None if no battery
    entity could be found - discover_battery_entities() already printed
    what it did/didn't find so you know what to hardcode in .env.
    """
    battery_entity_id, charging_entity_id = ha.discover_battery_entities()
    if not battery_entity_id:
        print("[battery monitor] Not starting - no battery entity to monitor.")
        return None

    charging_note = f" (charging sensor: {charging_entity_id})" if charging_entity_id else " (no charging sensor found)"
    print(
        f"[battery monitor] Watching {battery_entity_id}{charging_note}, "
        f"checking every {POLL_INTERVAL_SECONDS // 60} min."
    )

    thread = threading.Thread(
        target=_run, args=(battery_entity_id, charging_entity_id), daemon=True
    )
    thread.start()
    return thread

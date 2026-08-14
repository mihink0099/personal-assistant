# Personal Assistant

A voice- and text-based personal assistant powered by Claude, with smart-home
control via Home Assistant and fully offline speech-to-text/text-to-speech.

## Features

- **Conversational loop** with Claude (tool use, multi-turn memory)
- **Web search** for time-sensitive questions (current events, prices, etc.)
- **Home Assistant control**: list entities, check entity state, turn lights
  on/off - Claude discovers entity IDs dynamically instead of them being
  hardcoded anywhere
- **Push-to-talk voice input**: local, offline speech-to-text via
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Spoken responses**: local, offline text-to-speech via
  [Piper](https://github.com/OHF-Voice/piper1-gpl)
- **Barge-in interruption**: talk over the assistant mid-reply to cancel it
  ("stop", "cancel", "never mind", ...) or redirect it with a new question,
  the way Google Home/Alexa handle interruption, with acoustic echo
  cancellation so the assistant's own voice bleeding into the mic doesn't
  mask or falsely trigger interruptions
- **HA connectivity diagnostics**: a standalone script for troubleshooting
  intermittent connection issues to your Home Assistant instance
- **Background battery monitor**: auto-discovers your phone's battery
  sensor in Home Assistant and speaks an unprompted heads-up when it drops
  below 20% or reaches 100% - no need to ask

## Project structure

```
personal-assistant/
├── src/
│   └── personal_assistant/
│       ├── __init__.py
│       ├── assistant.py       # conversation loop, tool definitions, entry point
│       ├── home_assistant.py  # Home Assistant REST API wrapper
│       ├── voice.py           # push-to-talk STT, TTS, barge-in interruption
│       ├── aec.py             # acoustic echo cancellation for barge-in
│       └── battery_monitor.py # background thread: unprompted battery announcements
├── scripts/
│   └── diagnose_ha.py         # standalone Home Assistant connectivity diagnostic
├── .env.example                # copy to .env and fill in your own values
├── pyproject.toml
└── requirements.txt
```

## Setup

1. **Clone and install**

   ```bash
   git clone https://github.com/mihink0099/personal-assistant.git
   cd personal-assistant
   pip install -e .
   ```

   Or, without installing the package:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**

   ```bash
   cp .env.example .env
   ```

   Then fill in `.env`:

   | Variable            | Description                                                                                                          |
   | ------------------- | --------------------------------------------------------------------------------------------------------------------- |
   | `ANTHROPIC_API_KEY` | Your Anthropic API key                                                                                                |
   | `HA_TOKEN`          | A Home Assistant [Long-Lived Access Token](https://www.home-assistant.io/docs/authentication/#your-account-profile) |
   | `HA_URL`            | Base URL of your Home Assistant instance (e.g. `http://homeassistant.local`)                                          |
   | `BATTERY_ENTITY_ID` *(optional)* | Overrides auto-discovery of the phone battery sensor - only needed if it picks the wrong one or finds none |
   | `BATTERY_CHARGING_ENTITY_ID` *(optional)* | Overrides auto-discovery of the matching charging sensor |

3. **Run**

   ```bash
   personal-assistant
   ```

   (or `python -m personal_assistant.assistant` if you didn't install with
   `pip install -e .`)

   Voice models (a ~60MB Piper voice, and a local Whisper model) download
   automatically the first time you use voice input/output, then run fully
   offline after that. Neither is committed to the repo (see `.gitignore`).

## Usage

- Type a message and press Enter, or press Enter with nothing typed to
  talk (push-to-talk) - it records until you go quiet.
- While the assistant is speaking, you can talk over it: say "stop",
  "cancel", or "never mind" to cut it off, or ask a new question to
  redirect it immediately - no need to wait for it to finish.
- Type `exit` or `quit` to end the session.

## Diagnostics

If you're having trouble reaching Home Assistant, `scripts/diagnose_ha.py`
probes it every 2 seconds for 5 minutes (raw TCP + HTTP), logging every
attempt to `ha_diagnostic_log.txt` and summarizing the longest continuous
failure run at the end. Edit `HOST`/`PORT` at the top of the script to match
your setup.

```bash
python scripts/diagnose_ha.py
```

## Battery monitor

On startup, a background thread auto-discovers a battery-level sensor
(`device_class: battery`) in Home Assistant - and a matching charging
sensor, if one exists - by scanning `/api/states`; what it finds (or
doesn't) is printed so you can hardcode `BATTERY_ENTITY_ID` /
`BATTERY_CHARGING_ENTITY_ID` in `.env` if it guesses wrong or your setup
has more than one. It then checks every 5 minutes and speaks a short,
unprompted announcement the first time the level drops below 20% (while
not charging) or reaches 100% - not on every poll while it stays in that
state.

## Echo cancellation

Barge-in interrupt-watch runs the mic signal through acoustic echo
cancellation (`src/personal_assistant/aec.py`) before doing speech
detection on it, so the assistant's own voice coming out of the speakers
doesn't get picked up by the mic and mistaken for (or drowned out by) a
real interruption. Two backends, picked automatically at startup (and
printed to the log either way):

- **Real WebRTC AEC3**, via [aec-audio-processing](https://pypi.org/project/aec-audio-processing/)
  - not installed by default (prebuilt wheels only cover CPython 3.11-3.13
    on Windows as of writing). Install it manually on a matching
    interpreter with `pip install -e .[aec]`, or `pip install
    aec-audio-processing` directly.
- **A pure-numpy NLMS adaptive filter** - the automatic fallback whenever
  the above isn't installed (this project's own Python 3.14 environment
  included). Much simpler than a real AEC3 stack, but needs nothing beyond
  numpy.

**This can't be meaningfully validated on headset audio** - a headset mic
is acoustically isolated from its own earcups, so there's essentially no
echo path for either backend to cancel; barge-in will look identical with
AEC on or off in that setup, which isn't the same as it working. To
actually test it, use real external speaker + separate mic hardware, or
temporarily switch Windows' default playback device to the laptop's
built-in speakers (Settings > System > Sound) while keeping a separate mic
as input.

`DEBUG_AEC = True` in `voice.py` prints raw vs. post-AEC RMS side by side
on every interrupt-watch block - off by default (noisy), turn on
specifically while validating/tuning AEC against real hardware.

## Notes

- `DEBUG = True` in `voice.py` prints detailed voice-activity-detection info
  (input device, calibration, live RMS) - useful when tuning thresholds for
  your mic/room; set it to `False` once dialed in.
- The barge-in interrupt sensitivity (`INTERRUPT_WINDOW_BLOCKS`,
  `INTERRUPT_WINDOW_TRIGGER`, `INTERRUPT_WARMUP_SECONDS`) and voice-activity
  thresholds (`SPEECH_RMS_FLOOR`, `SPEECH_RMS_MULTIPLIER`) in `voice.py` may
  need tuning for your specific headset and room noise.
- The NLMS echo-cancellation fallback's `NLMS_FILTER_TAPS`/`NLMS_STEP_SIZE`,
  and the WebRTC backend's `WEBRTC_STREAM_DELAY_MS`, in `aec.py` may need
  tuning once you have real speaker hardware to test against (see "Echo
  cancellation" above).

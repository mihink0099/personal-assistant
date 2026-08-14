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
  the way Google Home/Alexa handle interruption
- **HA connectivity diagnostics**: a standalone script for troubleshooting
  intermittent connection issues to your Home Assistant instance

## Project structure

```
personal-assistant/
├── src/
│   └── personal_assistant/
│       ├── __init__.py
│       ├── assistant.py       # conversation loop, tool definitions, entry point
│       ├── home_assistant.py  # Home Assistant REST API wrapper
│       └── voice.py           # push-to-talk STT, TTS, barge-in interruption
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

## Notes

- `DEBUG = True` in `voice.py` prints detailed voice-activity-detection info
  (input device, calibration, live RMS) - useful when tuning thresholds for
  your mic/room; set it to `False` once dialed in.
- The barge-in interrupt sensitivity (`INTERRUPT_CONSECUTIVE_BLOCKS`,
  `INTERRUPT_WARMUP_SECONDS`) and voice-activity thresholds
  (`SPEECH_RMS_FLOOR`, `SPEECH_RMS_MULTIPLIER`) in `voice.py` may need
  tuning for your specific headset and room noise.

#!/usr/bin/env bash
# Container entrypoint. Translates Home Assistant add-on options
# (/data/options.json, the Supervisor's standard config file for every
# add-on) into the plain environment variables assistant.py, voice.py, and
# home_assistant.py already read via os.getenv()/load_dotenv() - none of
# that code knows or needs to know it's running inside an add-on.
set -e

CONFIG_PATH=/data/options.json

export ANTHROPIC_API_KEY="$(jq --raw-output '.anthropic_api_key' "$CONFIG_PATH")"

# HA_URL/HA_TOKEN default to the Supervisor's own authenticated proxy to
# Home Assistant core (SUPERVISOR_TOKEN is injected into every add-on
# container automatically) so this works with zero manual configuration -
# no long-lived access token to create and paste in. The ha_url/ha_token
# options exist only as a manual override (pointing at a different HA
# instance, or troubleshooting) - see DEPLOY.md.
HA_URL_OPTION="$(jq --raw-output '.ha_url // empty' "$CONFIG_PATH")"
HA_TOKEN_OPTION="$(jq --raw-output '.ha_token // empty' "$CONFIG_PATH")"
export HA_URL="${HA_URL_OPTION:-http://supervisor/core}"
export HA_TOKEN="${HA_TOKEN_OPTION:-$SUPERVISOR_TOKEN}"

export ENABLE_VOICE="$(jq --raw-output '.enable_voice' "$CONFIG_PATH")"

# wake_word.py's own default (models/hey_jarvis_custom.onnx, baked into
# the image) is used when WAKE_WORD_MODEL_PATH is unset entirely - so only
# export it here if the user actually set the option, rather than
# exporting an empty string that would override wake_word.py's default
# with nothing.
WAKE_WORD_MODEL_PATH_OPTION="$(jq --raw-output '.wake_word_model_path // empty' "$CONFIG_PATH")"
if [ -n "$WAKE_WORD_MODEL_PATH_OPTION" ]; then
    export WAKE_WORD_MODEL_PATH="$WAKE_WORD_MODEL_PATH_OPTION"
fi
export WAKE_WORD_THRESHOLD="$(jq --raw-output '.wake_word_threshold' "$CONFIG_PATH")"
export DEBUG_WAKE_WORD="$(jq --raw-output '.debug_wake_word' "$CONFIG_PATH")"

echo "[run.sh] Starting personal-assistant add-on (HA_URL=${HA_URL}, ENABLE_VOICE=${ENABLE_VOICE}, WAKE_WORD_THRESHOLD=${WAKE_WORD_THRESHOLD})"
exec python3 /app/pi_server.py

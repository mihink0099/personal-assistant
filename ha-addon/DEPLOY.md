# Deploying as a Home Assistant local add-on

This folder (`ha-addon/`) packages the assistant as a Home Assistant
**local add-on**: a FastAPI `POST /chat` endpoint plus an always-on,
wake-word-gated background voice loop, both running on the same device as
Home Assistant (typically the Raspberry Pi HAOS is installed on) instead
of your dev machine.

## What's shared vs. what's Pi-specific

This matters for "how do I update it later" - the two categories are
updated completely differently.

**Shared (lives in the main repo, reused unchanged)**

```
src/personal_assistant/
├── assistant.py       # conversation loop, tool definitions - run_turn() is called as-is
├── voice.py            # listen_for_speech(), speak_interruptible() - called as-is
├── home_assistant.py   # HA REST wrapper - untouched
├── aec.py               # echo cancellation, used internally by voice.py
└── battery_monitor.py   # not wired into pi_server.py by default - see "Ideas" below
pyproject.toml
requirements.txt
```

None of this is copied into `ha-addon/`. The add-on's `Dockerfile` pulls
it straight from **this same GitHub repo** (`main` branch by default - see
`PERSONAL_ASSISTANT_REPO`/`PERSONAL_ASSISTANT_REF` build args in the
Dockerfile) at build time and `pip install`s it. There is exactly one copy
of this code, ever - the one in `src/`.

**To update shared logic**: edit it in `src/` like normal, commit, push to
`main`. Then in Home Assistant: Settings → Add-ons → Personal Assistant →
**Rebuild** (not just Restart - Restart reuses the existing image and
won't see your changes; Rebuild re-runs the Dockerfile, which re-clones
`main`). The Dockerfile deliberately cache-busts the clone step on every
build so this always picks up the latest commit, not a stale cached one.

**Pi-specific (lives here, in `ha-addon/`, committed to the same repo)**

```
ha-addon/
├── config.yaml       # add-on manifest: ports, devices, user-configurable options
├── Dockerfile         # builds the container image
├── run.sh              # entrypoint: options.json -> environment variables
├── pi_server.py        # FastAPI /chat + the continuous-listening voice loop
├── wake_word.py         # openwakeword gating for the voice loop (see "Wake word" below)
├── models/
│   └── hey_jarvis_custom.onnx  # your wake word model - baked into the image, see Dockerfile
├── requirements.txt    # pi_server.py's own deps (fastapi, uvicorn, openwakeword) - separate from the repo root's
└── DEPLOY.md            # this file
```

`pi_server.py` and `wake_word.py` are new code, specific to running
unattended on the Pi - they don't exist anywhere else (the CLI is
push-to-talk and has no wake-word concept). Everything else here is
standard Home Assistant add-on packaging.

**To update Pi-specific files**: edit them here, commit/push (optional,
but keeps history), then re-copy this folder to the Pi (see below) and
Rebuild. Swapping the wake word model is the same process: replace
`models/hey_jarvis_custom.onnx` (or point `wake_word_model_path` at a
different filename), re-copy, Rebuild.

## One-time setup

1. Make sure your changes are pushed to GitHub - the Docker build clones
   from there, not from your local working copy.

2. Copy this `ha-addon/` folder onto the Pi, at
   `/addons/local/personal_assistant/` (the standard location Home
   Assistant's Supervisor scans for local add-ons). From your Windows
   machine, using the OpenSSH client built into Windows 10/11:

   ```powershell
   ssh pi@<pi-host> "mkdir -p /addons/local/personal_assistant"
   scp -r ha-addon\* pi@<pi-host>:/addons/local/personal_assistant/
   ```

   (If you're running HAOS directly rather than a separate Pi with SSH
   access to the filesystem, use the Samba share add-on or the SSH & Web
   Terminal add-on to reach `/addons/local/` instead.)

3. In the Home Assistant UI: Settings → Add-ons → Add-on Store → ⋮ (top
   right) → **Check for updates**. "Personal Assistant" should appear
   under **Local add-ons**.

4. Click it → **Install**. This triggers the Docker build described above
   (expect it to take a few minutes the first time - it's compiling
   PortAudio bindings and pulling faster-whisper/Piper's dependencies).

5. Go to the **Configuration** tab and set:
   - `anthropic_api_key` - required.
   - `ha_url` / `ha_token` - leave blank. By default `run.sh` points the
     assistant at the Supervisor's own authenticated proxy
     (`http://supervisor/core`, using the `SUPERVISOR_TOKEN` every add-on
     container gets automatically) so it talks to *this* Home Assistant
     instance with zero manual token setup. Only fill these in if you
     specifically want to point it at a different HA instance.
   - `enable_voice` - `true` for the continuous mic listening loop, `false`
     for `/chat`-only (no mic hardware attached, or you just don't want an
     always-listening mic right now).
   - `wake_word_model_path` - leave blank to use the model baked into the
     image at `models/hey_jarvis_custom.onnx`. Only set this if you've
     added a different model file (see "Wake word" below).
   - `wake_word_threshold` - `0.5` is a reasonable starting point; see
     "Wake word" below for how to tune it against your actual room.
   - `debug_wake_word` - leave `false` normally; flip to `true` temporarily
     while tuning the threshold (see below).

6. Start the add-on. Check the **Log** tab - you should see
   `[run.sh] Starting personal-assistant add-on ...` followed by either
   `[wake_word] Loading '...' (threshold=0.5)` then `[pi_server] Continuous
   voice listening started (wake-word gated).`, or the "ENABLE_VOICE is
   off" message if you disabled voice.

## Wake word

The voice loop no longer treats "any sufficiently loud sound" as a
command - it idles on a dedicated wake-word model (via
[openwakeword](https://github.com/dscripka/openWakeWord)) until that
specific word/phrase is heard, and only *then* opens a real
`voice.listen_for_speech()` capture for the actual command. After the
resulting turn finishes, it goes back to idling on the wake word - not
straight back to listening for anything loud.

- **Model file**: `wake_word.py` loads `WAKE_WORD_MODEL_PATH` if set,
  otherwise `models/hey_jarvis_custom.onnx` (baked into the image by the
  Dockerfile's `COPY models ./models`). **If neither exists, the add-on
  refuses to start** with a clear error rather than silently falling back
  to one of openwakeword's built-in wake words - a silent fallback would
  mean it responds to a different phrase than the one you think you
  configured, which is a much worse failure mode than a startup crash you
  can actually see in the log.
- **Tuning the threshold**: set `debug_wake_word: true` in Configuration,
  restart the add-on, and watch the log - every single mic block prints
  its live confidence score (`[debug:wake-word] score=... threshold=...`).
  Say the wake word a few times and see what scores it actually produces
  in your room versus background noise/normal speech, then set
  `wake_word_threshold` accordingly (higher = fewer false triggers but
  more missed wake words; lower = the opposite). Turn `debug_wake_word`
  back off once you're done - it's noisy.

## Testing

Text:

```bash
curl -X POST http://<pi-host>:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what time is it", "speak": false}'
```

`"speak": false` skips playing the reply through the Pi's speaker (default
is `true` - see pi_server.py's `ChatRequest.speak`) - useful for testing
over SSH/curl without triggering audio every time.

Voice: say the wake word first (e.g. "Hey Jarvis", depending on which
model is loaded), wait for `[wake_word] Wake word detected` in the log,
then speak your command - same push-to-talk-style silence detection as
the CLI (`voice.record_until_silence()`) from there, just triggered by the
wake word instead of Enter.

Health check: `curl http://<pi-host>:8000/health`

## Known trade-offs

- **`/chat` can be slow to respond if it lands while the voice loop is
  actively handling a command** (i.e. after the wake word has already
  triggered, from `voice.listen_for_speech()` through the end of that
  turn). Both entry points share one `threading.Lock`
  (`conversation_lock` in `pi_server.py`) held for that whole span - not
  just the Claude call and speaking. This is deliberate: `voice.py`'s
  `listen_for_speech()` and `speak_interruptible()` each open their own
  `sd.InputStream`, and running two of those on the same audio device at
  once isn't something `voice.py` was written to support (it was built
  for a single-threaded CLI). Serializing "who has the mic" avoids that
  entirely, at the cost of `/chat` occasionally waiting a few seconds -
  though wake-word gating means this window is now much narrower than
  before (it's only from wake-word-detected to turn-finished, not the
  entire idle-listening loop). This wasn't validated against real
  concurrent load - if it matters for your use case, watch the add-on log
  for how often it actually happens.
- **Wake-word listening itself is deliberately *not* covered by
  `conversation_lock`** (see `voice_loop()`'s comment in `pi_server.py`) -
  it can block for minutes or hours between wake words, and holding a
  shared lock across that would starve `/chat` almost permanently. The
  accepted residual risk: `wait_for_wake_word()` keeps listening even
  while a `/chat`-triggered reply is being spoken elsewhere, so in theory
  two `sd.InputStream`s could be open briefly at once in that narrow
  window. Not validated against real hardware - if you see audio glitches
  specifically when a `/chat` reply is spoken while voice is also active,
  this is the first place to look.
- **Echo cancellation (`aec.py`) still needs real speaker hardware to
  validate** - see its module docstring. This applies here exactly as it
  does to the CLI; running as an add-on doesn't change it.
- **The add-on runs the pure-numpy NLMS echo-cancellation fallback, not
  real WebRTC AEC3** - `aec-audio-processing`'s prebuilt wheels are
  Windows-only as of writing (see `aec.py`'s module docstring), and this
  container is Linux, so the Dockerfile doesn't even attempt to install
  it (a source build would need Meson/CMake/SWIG and a C++ toolchain not
  currently wired up here). Functionally this is the same fallback the
  dev machine uses, just for a different reason. Check the add-on log for
  `[AEC] Using ...` to confirm.

## Ideas not implemented here (out of scope for this pass)

- `battery_monitor.start()` isn't called from `pi_server.py` - it's a
  three-line addition (`from personal_assistant import battery_monitor`
  + a `battery_monitor.start()` call near the top of `pi_server.py`'s
  `__main__` block) if you want phone-battery announcements from the
  add-on too, exactly like the CLI has.
- No addon-side audio device selection UI - if the Pi has multiple audio
  devices, `sounddevice`'s default-device selection applies exactly as
  it does on the CLI (see `voice.py`); set it via `asound.conf`/`ALSA`
  defaults on the Pi's host OS if you need a specific device chosen.

"""
Wake-word gating for pi_server.py's continuous-listening voice loop, using
openwakeword. Lives here in ha-addon/ (not src/personal_assistant/)
because it's specific to the always-on Pi deployment - the CLI is
push-to-talk (Enter key) and has no equivalent "is anyone even talking to
me" problem to solve.

Replaces the earlier naive behavior (any RMS-loud sound handed straight to
voice.listen_for_speech()) with: stay quiet, watching the mic for one
specific wake word, and only start a real listen_for_speech() capture once
that's actually been heard.

Model file: loaded from WAKE_WORD_MODEL_PATH if set, otherwise
models/hey_jarvis_custom.onnx alongside this file. If neither exists, this
module refuses to import (see _load_model()) rather than silently falling
back to one of openwakeword's bundled built-in models - a silent fallback
would mean the assistant responds to a *different* wake word than the one
you thought you configured, which is a much more confusing failure mode
than a clear startup error.
"""

import os
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz - openwakeword's own recommended frame size

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hey_jarvis_custom.onnx"
MODEL_PATH = Path(os.getenv("WAKE_WORD_MODEL_PATH") or DEFAULT_MODEL_PATH)

# TUNE THIS against your actual room/mic - see DEBUG_WAKE_WORD below for
# the live scores to tune it with. Higher = fewer false triggers but more
# missed wake words; lower = the opposite.
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))

# Prints every live confidence score, not just ones that cross the
# threshold - noisy by design, turn on specifically while tuning
# WAKE_WORD_THRESHOLD against your Pi's actual room.
DEBUG_WAKE_WORD = os.getenv("DEBUG_WAKE_WORD", "false").strip().lower() in ("1", "true", "yes")


def _load_model() -> Model:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Wake word model not found at '{MODEL_PATH}'. Set WAKE_WORD_MODEL_PATH "
            "in the add-on's Configuration tab, or place your .onnx model at "
            f"'{DEFAULT_MODEL_PATH}' - see DEPLOY.md. Refusing to silently fall back "
            "to one of openwakeword's built-in models instead."
        )
    print(f"[wake_word] Loading '{MODEL_PATH}' (threshold={WAKE_WORD_THRESHOLD})")
    # inference_framework="onnx": our model file is .onnx, not .tflite -
    # avoids openwakeword also requiring tflite-runtime for a format we're
    # not using.
    return Model(wakeword_models=[str(MODEL_PATH)], inference_framework="onnx")


_model = _load_model()


def wait_for_wake_word() -> None:
    """
    Blocks until the configured wake word is heard, then returns. Opens
    its own sd.InputStream for the duration - by design this can block
    indefinitely (could be minutes between wake words), so callers should
    NOT hold any lock shared with other mic/speaker users across this call
    - see pi_server.py's voice_loop() for why conversation_lock is
    deliberately not held here.
    """
    _model.reset()

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        while True:
            chunk = audio_queue.get()
            prediction = _model.predict(chunk.flatten())
            # Exactly one wakeword model is loaded (see _load_model()), so
            # prediction has exactly one entry - read the score straight
            # out of predict()'s own return value rather than assuming
            # what openwakeword internally names it (its key naming is an
            # implementation detail we don't need to depend on here).
            score = float(max(prediction.values(), default=0.0))

            if DEBUG_WAKE_WORD:
                print(f"[debug:wake-word] score={score:.3f}  threshold={WAKE_WORD_THRESHOLD:.3f}")

            if score >= WAKE_WORD_THRESHOLD:
                print(f"[wake_word] Wake word detected (score={score:.3f}).")
                # Clears openwakeword's internal rolling buffer so trailing
                # audio from this detection doesn't immediately re-trigger
                # (or suppress) the next wait_for_wake_word() call.
                _model.reset()
                return

"""
Push-to-talk voice I/O for the smart home assistant.

Everything here is local/offline once the model files have been downloaded
once:
  - Speech-to-text: faster-whisper (a CTranslate2 reimplementation of
    OpenAI's Whisper) running on CPU. The model weights are downloaded
    from Hugging Face the first time WhisperModel() is constructed and
    then cached in ~/.cache/huggingface - no network needed after that.
  - Text-to-speech: Piper, a small local neural TTS engine. Its voice
    model (.onnx) and config (.onnx.json) are NOT bundled with Piper or
    this repo - we download them once into ./voices/ the first time
    speak() is used, then reuse them offline from then on.

Both directions talk to the microphone/speakers via `sounddevice`, which
plays and records plain numpy float32 arrays.
"""

import concurrent.futures
import queue
import re
import string
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

from . import aec, audio_devices

# Set to False once voice activity detection is dialed in and you don't
# need to see mic/threshold/RMS details on every recording anymore.
DEBUG = True

# Separate from DEBUG - prints raw vs. post-echo-cancellation RMS side by
# side on every interrupt-watch block. Off by default since it's noisy;
# turn on specifically while tuning/validating AEC (see aec.py's module
# docstring for why that requires real speaker hardware, not a headset).
DEBUG_AEC = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Whisper's feature extractor expects 16kHz mono audio - recording at that
# rate directly means we never have to resample before transcribing.
SAMPLE_RATE = 16_000
BLOCK_SECONDS = 0.1  # size of each chunk we read from the mic while recording

# Energy-based (RMS) voice activity detection - no extra dependency beyond
# what we already need. A short burst of ambient noise seeds a starting
# estimate once, the first time the process records anything; after that
# the estimate keeps adapting for the rest of the session (see
# _NoiseFloorEstimator below) instead of resetting to a fresh calibration
# on every recording.
CALIBRATION_SECONDS = 0.3          # length of the one-time startup calibration burst
NOISE_FLOOR_EMA_ALPHA = 0.08       # how fast the adaptive estimate reacts to new silence (0-1, higher = faster)
SPEECH_RMS_FLOOR = 0.01            # minimum RMS to ever count as speech
SPEECH_RMS_MULTIPLIER = 3.0        # speech must be this many x louder than the adaptive noise floor
SILENCE_HOLD_SECONDS = 1.2         # stop this long after speech trails off
MAX_RECORD_SECONDS = 15.0          # hard cap so a stuck mic can't record forever
NO_SPEECH_TIMEOUT_SECONDS = 5.0    # give up if nothing is heard at all

# "base.en" is a good speed/accuracy balance for push-to-talk on CPU.
# English-only ("*.en") models are smaller and a bit faster than the
# multilingual ones of the same size.
WHISPER_MODEL_SIZE = "base.en"

# Piper voice. "medium" quality is a reasonable default; the files are
# ~60MB combined and are downloaded once into VOICES_DIR (see
# ensure_voice_downloaded below) - they are not committed to git.
PIPER_VOICE_NAME = "en_US-amy-medium"
VOICES_DIR = Path(__file__).parent / "voices"

# Piper's default pace reads a little slow/deliberate for a voice assistant.
# length_scale < 1 speeds up delivery (0.85 = ~15% faster) without changing
# pitch, unlike naively speeding up the raw audio.
SPEECH_SYNTHESIS_CONFIG = SynthesisConfig(length_scale=0.85)

# Barge-in interruption (speak_interruptible, below) - how eagerly playback
# treats mic input as the user talking over it, versus ignoring it.
# TUNE THESE against your actual headset/room - see notes at the bottom.
INTERRUPT_WARMUP_SECONDS = 0.25      # ignore the mic for this long after each chunk starts
# Rolling window instead of a strict consecutive-block streak, so a natural
# micro-pause between words (e.g. "wait... stop") doesn't reset progress
# toward detecting a real interruption.
INTERRUPT_WINDOW_BLOCKS = 5          # look at the last this-many blocks (~0.5s)
INTERRUPT_WINDOW_TRIGGER = 3         # ...and trigger once at least this many are over threshold

# Checked after normalizing (lowercase, punctuation stripped). Matched
# either as a whole utterance, or as a keyword inside a short (few-word)
# utterance - see _is_stop_command(). These cancel outright; anything else
# transcribed during an interruption is treated as a redirect - the next
# user turn - instead.
STOP_PHRASES = {"stop", "shut up", "cancel", "quiet", "enough", "never mind"}
STOP_PHRASE_MAX_WORDS = 5            # utterances this short or shorter are checked for a contained keyword too

# After a stop command, how long to keep listening for an immediate
# follow-up ("stop... actually, turn off the lights") before giving up and
# returning to the idle prompt.
STOP_GRACE_LISTEN_SECONDS = 3.5

# Model objects are slow to construct (loading weights), so we build each
# one at most once per process and reuse it across turns.
_whisper_model: WhisperModel | None = None
_piper_voice: PiperVoice | None = None

# Resolved once per process, then reused - see audio_devices.py for why we
# can't rely on sounddevice's OS default input/output device.
_input_device_index: int | None = None
_output_device_index: int | None = None


def _get_input_device_index() -> int:
    global _input_device_index
    if _input_device_index is None:
        _input_device_index = audio_devices.resolve_input_device()
        print(
            f"[voice] Resolved input device index {_input_device_index}: "
            f"{sd.query_devices(_input_device_index)['name']}"
        )
    return _input_device_index


def _get_output_device_index() -> int:
    global _output_device_index
    if _output_device_index is None:
        _output_device_index = audio_devices.resolve_output_device()
        print(
            f"[voice] Resolved output device index {_output_device_index}: "
            f"{sd.query_devices(_output_device_index)['name']}"
        )
    return _output_device_index

# Guards actual audio *playback* (the sd.play()/sd.wait() calls in speak()
# and speak_interruptible()) so an unprompted background announcement (see
# battery_monitor.py) can never start talking over the main loop mid-reply,
# or vice versa - whichever gets here second just blocks on this lock until
# the other one's done, then plays normally. Synthesis (Piper, which is
# CPU-bound) deliberately happens *outside* the lock so it isn't blocked.
PLAYBACK_LOCK = threading.Lock()


class _NoiseFloorEstimator:
    """
    Session-persistent, continuously-adapting ambient-noise estimate - an
    exponential moving average (EMA) of RMS, updated only from blocks
    confirmed as silence (never from speech - see each call site's
    `.update()`). This lets the live speech threshold track slow drift in
    room/mic conditions (an AC turning on, a door closing) across the
    whole running process, instead of resetting to a fresh from-scratch
    calibration every time record_until_silence() or speak_interruptible()
    runs.
    """

    def __init__(self, alpha: float = NOISE_FLOOR_EMA_ALPHA):
        self.alpha = alpha
        self.ema: float | None = None

    def seed(self, initial_value: float) -> None:
        """One-time startup calibration burst result - a no-op if already seeded."""
        if self.ema is None:
            self.ema = initial_value

    def update(self, rms: float) -> None:
        """Folds one more confirmed-silence RMS reading into the running estimate."""
        if self.ema is None:
            self.ema = rms
        else:
            self.ema = self.alpha * rms + (1 - self.alpha) * self.ema

    @property
    def threshold(self) -> float:
        ema = self.ema if self.ema is not None else 0.0
        return max(SPEECH_RMS_FLOOR, ema * SPEECH_RMS_MULTIPLIER)


# One estimator, shared for the process's lifetime by every recording path
# (push-to-talk and barge-in interrupt-watch alike) - see _seed_noise_floor()
# below and each classification site's _noise_floor.update() call.
_noise_floor = _NoiseFloorEstimator()


# ---------------------------------------------------------------------------
# Recording + voice activity detection
# ---------------------------------------------------------------------------
def _seed_noise_floor(audio_queue: queue.Queue, debug_label: str = "") -> None:
    """
    Runs once per process, whichever entry point (record_until_silence() or
    speak_interruptible()) gets there first: samples CALIBRATION_SECONDS of
    ambient noise to seed _noise_floor's starting EMA value. Every call
    after that is a no-op - from then on the estimate stays current on its
    own via _noise_floor.update(), called on every block classified as
    silence during capture/interrupt-watch.
    """
    if _noise_floor.ema is not None:
        return

    noise_samples = []
    elapsed = 0.0
    while elapsed < CALIBRATION_SECONDS:
        block = audio_queue.get()
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        noise_samples.append(rms)
        elapsed += BLOCK_SECONDS

    _noise_floor.seed(max(noise_samples))
    if DEBUG:
        label = f":{debug_label}" if debug_label else ""
        print(
            f"[debug{label}] seeded adaptive noise floor: ema={_noise_floor.ema:.5f}  "
            f"threshold={_noise_floor.threshold:.5f}"
        )


def _capture_until_silence(
    audio_queue: queue.Queue,
    initial_blocks: list[np.ndarray] | None = None,
    already_speaking: bool = False,
    debug_label: str = "",
    no_speech_timeout: float = NO_SPEECH_TIMEOUT_SECONDS,
    max_record_seconds: float = MAX_RECORD_SECONDS,
) -> np.ndarray | None:
    """
    Shared recording loop: reads blocks from `audio_queue` until the
    speaker has gone quiet for SILENCE_HOLD_SECONDS after talking, or
    max_record_seconds elapses. Used by record_until_silence() (push-to-
    talk, starting from silence), by speak_interruptible() (continuing to
    record after a barge-in has already been detected mid-playback), and
    by its post-stop-command grace-listen window (a shorter
    no_speech_timeout so it gives up quickly if there's no follow-up).

    Reads the live threshold from the shared _noise_floor estimator on
    every block (rather than a value fixed at the start of the call), and
    feeds every block classified as silence back into it via
    _noise_floor.update() - so a long recording's own trailing-silence
    blocks keep the estimate current too, not just the startup burst.

    If `already_speaking` is True, speech is assumed to have already
    started - skips the "wait for speech to begin" phase and its
    no_speech_timeout give-up, since we already know the user is talking
    (that's why this was called). `initial_blocks`, if given, are
    prepended so the leading edge of speech that triggered detection
    isn't lost.

    Returns the concatenated audio, or None if speech was never detected
    (only possible when already_speaking is False).
    """
    recorded_blocks = list(initial_blocks) if initial_blocks else []
    speech_detected = already_speaking
    silence_run = 0.0
    elapsed = len(recorded_blocks) * BLOCK_SECONDS
    block_count = len(recorded_blocks)
    # Print every few blocks instead of every single one (10/sec at
    # BLOCK_SECONDS=0.1) so debug output stays readable.
    debug_print_every_n_blocks = 3
    label = f":{debug_label}" if debug_label else ""

    while True:
        block = audio_queue.get()
        recorded_blocks.append(block)
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        threshold = _noise_floor.threshold
        elapsed += BLOCK_SECONDS
        block_count += 1

        is_speech = rms > threshold
        if is_speech:
            speech_detected = True
            silence_run = 0.0
        else:
            _noise_floor.update(rms)
            if speech_detected:
                silence_run += BLOCK_SECONDS

        if DEBUG and block_count % debug_print_every_n_blocks == 0:
            marker = "SPEECH" if is_speech else "silence"
            print(f"[debug{label}] rms={rms:.5f}  threshold={threshold:.5f}  ({marker})")

        if speech_detected and silence_run >= SILENCE_HOLD_SECONDS:
            break
        if elapsed >= max_record_seconds:
            break
        if not speech_detected and elapsed >= no_speech_timeout:
            break

    if not speech_detected:
        return None

    return np.concatenate(recorded_blocks).flatten()


def record_until_silence() -> np.ndarray | None:
    """
    Records from the default microphone until the speaker has gone quiet
    for SILENCE_HOLD_SECONDS, or MAX_RECORD_SECONDS is hit as a hard cap.

    Returns a mono float32 numpy array at SAMPLE_RATE, or None if no
    speech was detected at all within NO_SPEECH_TIMEOUT_SECONDS (so the
    caller can fall back to typed input).
    """
    block_samples = int(BLOCK_SECONDS * SAMPLE_RATE)
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            device=_get_input_device_index(),
            callback=callback,
        )
    except (sd.PortAudioError, RuntimeError) as e:
        print(f"Could not open the microphone: {e}")
        return None

    with stream:
        _seed_noise_floor(audio_queue, debug_label="ptt")
        return _capture_until_silence(audio_queue, debug_label="ptt")


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------
def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(
            f"Loading local Whisper model '{WHISPER_MODEL_SIZE}' "
            "(first run downloads it; cached and fully offline after that)..."
        )
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio: np.ndarray) -> str:
    """Transcribes a mono float32 SAMPLE_RATE array to text."""
    model = _get_whisper_model()
    segments, _info = model.transcribe(audio, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


def listen_for_speech() -> str | None:
    """
    Records one push-to-talk utterance and transcribes it. Returns the
    transcribed text, or None if no speech was detected or the
    transcription came back empty - either way, the caller should fall
    back to typed input.
    """
    print("Listening... (speak now, pause when you're done)")
    audio = record_until_silence()
    if audio is None:
        print("No speech detected.")
        return None

    print("Transcribing...")
    text = transcribe(audio)
    if not text:
        print("Transcription was empty.")
        return None

    print(f"Heard: {text}")
    return text


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------
# Split on '.'/'!'/'?' followed by whitespace, but NOT when the punctuation
# immediately follows a digit - otherwise a numbered list ("1. Do X\n2. Do
# Y") gets shredded into fragments like "1." and "Do X\n2.", which Piper
# then reads as broken half-sentences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?<!\d[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    """Splits `text` into sentence-level chunks for incremental playback."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip())]
    return [s for s in sentences if s]


# Markdown that reads naturally in a chat window but unnaturally verbatim
# out loud - "asterisk asterisk turn off the lights asterisk asterisk" is
# not what Piper should say for "**Turn off the lights**". Header/bullet/
# numbered-list markers are line-anchored (only stripped when they open a
# line); bold can appear anywhere inline.
_MARKDOWN_HEADER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_MARKDOWN_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_MARKDOWN_NUMBERED_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")


def _strip_markdown_for_speech(text: str) -> str:
    """
    Removes markdown formatting before it ever reaches Piper: bold markers,
    header hashes, and bullet/numbered list prefixes. The underlying words
    are kept (a list item still gets spoken, just without "dash" or "one
    dot" in front of it); only the formatting characters go.
    """
    text = _MARKDOWN_HEADER_RE.sub("", text)
    text = _MARKDOWN_BULLET_RE.sub("", text)
    text = _MARKDOWN_NUMBERED_RE.sub("", text)
    text = _MARKDOWN_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    return text


# Symbols Piper otherwise reads literally ("dollar sign", "asterisk",
# "percent sign") instead of the way a person would actually say them
# out loud. Currency and percent are matched before the leftover-symbol
# cleanup below so "$12.50" becomes "12 dollars and 50 cents", not
# "dollar sign 12.50".
_CURRENCY_RE = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{1,2}))?")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")

# Weather/measurement units that come up constantly in assistant replies
# (temperature, wind/speed) but that Piper reads as literal symbols/letters
# rather than words - "18°C" as "eighteen degrees C", "20 km/h" as "twenty
# k m slash h". Matched case-insensitively since Claude's output casing
# isn't guaranteed consistent.
_TEMPERATURE_RE = re.compile(r"°\s?([CF])\b", re.IGNORECASE)
_SPEED_UNIT_RE = re.compile(r"\b(km/h|kph|mph)\b", re.IGNORECASE)
_TEMPERATURE_WORDS = {"c": " degrees Celsius", "f": " degrees Fahrenheit"}
_SPEED_WORDS = {"km/h": "kilometers per hour", "kph": "kilometers per hour", "mph": "miles per hour"}


def _pluralize(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _currency_to_words(match: re.Match) -> str:
    dollars = int(match.group(1).replace(",", ""))
    cents_group = match.group(2)
    cents = int(cents_group.ljust(2, "0")) if cents_group else 0

    dollars_part = f"{dollars} {_pluralize(dollars, 'dollar', 'dollars')}"
    cents_part = f"{cents} {_pluralize(cents, 'cent', 'cents')}"

    if cents == 0:
        return dollars_part
    if dollars == 0:
        return cents_part
    return f"{dollars_part} and {cents_part}"


def _normalize_symbols_for_speech(text: str) -> str:
    """
    Rewrites symbols that Piper would otherwise spell out letter-by-letter
    ("$500" -> "dollar sign 500") into how they're actually spoken
    ("500 dollars"). Run after _strip_markdown_for_speech, which already
    removes **bold**/#headers/-bullets - this only handles what's left.
    """
    text = _CURRENCY_RE.sub(_currency_to_words, text)
    text = _PERCENT_RE.sub(lambda m: f"{m.group(1)} percent", text)
    text = _TEMPERATURE_RE.sub(lambda m: _TEMPERATURE_WORDS[m.group(1).lower()], text)
    text = _SPEED_UNIT_RE.sub(lambda m: _SPEED_WORDS[m.group(1).lower()], text)
    text = re.sub(r"#(\d+)", r"number \1", text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = text.replace("@", " at ")
    text = text.replace("*", "").replace("#", "")
    # Leftover bare degree sign (no C/F immediately after, e.g. "180°
    # angle") - the temperature-specific case above already consumed the
    # common "°C"/"°F" pattern, so anything left here isn't a duplicate.
    text = text.replace("°", " degrees ")
    text = re.sub(r" {2,}", " ", text)
    return text


# How many sentences speak_interruptible() synthesizes and plays as one
# clip. Grouping (instead of one sentence per chunk) gives Piper more
# context to work with, so the result sounds like a phrase instead of a
# string of separately-clipped sentences - at the cost of interrupt
# latency now being bounded by roughly this many sentences instead of one.
SENTENCES_PER_CHUNK = 3


def _group_sentences(sentences: list[str], group_size: int = SENTENCES_PER_CHUNK) -> list[str]:
    """
    Joins consecutive sentences into group_size-sentence chunks. Avoids
    leaving a trailing group of just one sentence (which would reintroduce
    the same choppiness this is meant to fix) by folding it into the
    previous group instead of playing it alone.
    """
    if not sentences:
        return []
    if len(sentences) <= group_size:
        return [" ".join(sentences)]

    groups = [sentences[i:i + group_size] for i in range(0, len(sentences), group_size)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())

    return [" ".join(group) for group in groups]


_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _is_stop_command(text: str) -> bool:
    """
    True if `text` should cancel playback rather than being treated as a
    redirect. Matches STOP_PHRASES either as the whole (normalized)
    utterance, or as a keyword contained in a short utterance - so "please
    stop" or "okay shut up" cancel too, without a longer, unrelated
    sentence that happens to contain "stop" (e.g. a question about a bus
    stop) false-triggering.
    """
    normalized = text.strip().lower().translate(_PUNCTUATION_TABLE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    if normalized in STOP_PHRASES:
        return True

    words = normalized.split()
    if len(words) <= STOP_PHRASE_MAX_WORDS:
        return any(phrase in normalized for phrase in STOP_PHRASES)

    return False


def ensure_voice_downloaded() -> tuple[Path, Path]:
    """
    Makes sure the Piper voice model and its config file exist locally,
    downloading them from the Piper voices repo on Hugging Face if not.
    Returns (model_path, config_path).
    """
    VOICES_DIR.mkdir(exist_ok=True)
    model_path = VOICES_DIR / f"{PIPER_VOICE_NAME}.onnx"
    config_path = VOICES_DIR / f"{PIPER_VOICE_NAME}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        print(f"Downloading Piper voice '{PIPER_VOICE_NAME}' (one-time, ~60MB)...")
        download_voice(PIPER_VOICE_NAME, VOICES_DIR)

    return model_path, config_path


def _get_piper_voice() -> PiperVoice:
    global _piper_voice
    if _piper_voice is None:
        model_path, config_path = ensure_voice_downloaded()
        _piper_voice = PiperVoice.load(model_path, config_path)
    return _piper_voice


def speak(text: str) -> None:
    """Synthesizes `text` with Piper and plays it through the default output device."""
    text = _normalize_symbols_for_speech(_strip_markdown_for_speech(text.strip())).strip()
    if not text:
        return

    print(f"Speaking: {text}")
    voice = _get_piper_voice()
    chunks = list(voice.synthesize(text, syn_config=SPEECH_SYNTHESIS_CONFIG))
    if not chunks:
        return

    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    with PLAYBACK_LOCK:
        try:
            sd.play(audio, samplerate=chunks[0].sample_rate, device=_get_output_device_index())
            sd.wait()
        except (sd.PortAudioError, RuntimeError) as e:
            print(f"Could not play audio: {e}")


def _play_chunk_watching_for_interrupt(
    audio: np.ndarray,
    sample_rate: int,
    audio_queue: queue.Queue,
) -> list[np.ndarray] | None:
    """
    Plays one chunk via non-blocking sd.play() while watching `audio_queue`
    for the user barging in. Polls sd.get_stream().active rather than a
    time.monotonic() duration estimate, since real playback runs measurably
    longer than len(audio)/sample_rate (buffering latency) - using a time
    estimate would risk cutting the chunk's tail off early on the next
    sd.play() call.

    Reads the live threshold from the shared _noise_floor estimator on
    every block and feeds every block classified as silence back into it,
    same as _capture_until_silence - see _NoiseFloorEstimator.

    Runs acoustic echo cancellation (see aec.py) on every mic block before
    anything else touches it - `audio` (the exact reference signal being
    sent to the speakers this chunk, resampled to SAMPLE_RATE once up
    front) is fed to the AEC backend alongside the live mic block, and
    everything downstream (RMS, threshold comparison, the returned
    blocks) uses the AEC-cleaned signal, not the raw mic input. The
    reference slice for a given mic block is looked up by elapsed
    wall-clock time since sd.play() was called - there's no true
    sample-accurate loopback sync available here, so this is an
    approximation (see aec.py's module docstring for the same caveat).

    Returns None if the chunk finished playing without interruption, or
    the list of mic blocks captured so far (starting from the first block
    that crossed the threshold) if the user interrupted - sd.stop() has
    already been called by the time this returns.

    An interruption triggers on a rolling window (at least
    INTERRUPT_WINDOW_TRIGGER of the last INTERRUPT_WINDOW_BLOCKS blocks
    over threshold) rather than a strict consecutive-block streak, so a
    natural micro-pause between words doesn't reset progress toward
    detecting a real interruption. The returned blocks start from the
    first block of the current speech attempt (i.e. the first block after
    the window last contained zero speech blocks), not just the ones
    counted toward the trigger, so the leading word isn't lost.

    INTERRUPT_WARMUP_SECONDS at the start of the chunk is ignored, so the
    tail of the *previous* chunk (still settling in the room/mic) doesn't
    falsely trigger a barge-in on this one.
    """
    # Drop any input blocks queued up before this chunk started (e.g. while
    # Piper was synthesizing it), so the warm-up timer isn't thrown off by
    # stale audio.
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    # Resample the reference once, up front, rather than per mic block -
    # it's the same underlying audio for the whole chunk, just sliced
    # differently per block below.
    aec_backend = aec.get_backend()
    reference_at_mic_rate = aec.resample_reference(audio, sample_rate, SAMPLE_RATE)

    # Held for the whole time this chunk is actually coming out of the
    # speaker (see PLAYBACK_LOCK) - released via the `with` block on every
    # return path below, including the early one on a PortAudioError.
    with PLAYBACK_LOCK:
        try:
            sd.play(audio, samplerate=sample_rate, device=_get_output_device_index())
        except (sd.PortAudioError, RuntimeError) as e:
            print(f"Could not play audio: {e}")
            return None

        output_stream = sd.get_stream()
        chunk_start = time.monotonic()
        window: deque[bool] = deque(maxlen=INTERRUPT_WINDOW_BLOCKS)
        pending_blocks: list[np.ndarray] = []

        while output_stream.active:
            elapsed = time.monotonic() - chunk_start
            try:
                block = audio_queue.get(timeout=BLOCK_SECONDS)
            except queue.Empty:
                continue

            if elapsed < INTERRUPT_WARMUP_SECONDS:
                continue  # still inside the warm-up guard - ignore this block

            # Slice out the reference audio for this same elapsed-time
            # window (see the docstring's note on why this is elapsed-time
            # aligned rather than sample-accurate), padding with silence
            # if the block runs past the end of this chunk's reference.
            ref_start = int(elapsed * SAMPLE_RATE)
            ref_end = ref_start + len(block)
            reference_slice = reference_at_mic_rate[ref_start:ref_end]
            if len(reference_slice) < len(block):
                reference_slice = np.concatenate([
                    reference_slice,
                    np.zeros(len(block) - len(reference_slice), dtype=np.float32),
                ])

            cleaned_block = aec_backend.process(block, reference_slice)

            raw_rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            rms = float(np.sqrt(np.mean(cleaned_block.astype(np.float64) ** 2)))
            threshold = _noise_floor.threshold
            is_speech = rms > threshold
            window.append(is_speech)

            if not is_speech:
                _noise_floor.update(rms)

            # Keep accumulating through the whole current speech attempt,
            # not just the blocks that end up counted toward the trigger -
            # as long as the window has *any* speech in it, we're still
            # mid-attempt (a lone silent block, e.g. a natural pause,
            # doesn't reset this). Only a window that's gone fully quiet
            # clears the buffer. Accumulates the AEC-cleaned block (not
            # the raw one), since that's what gets transcribed later.
            if any(window):
                pending_blocks.append(cleaned_block)
            else:
                pending_blocks.clear()

            if DEBUG:
                marker = "SPEECH" if is_speech else "silence"
                print(
                    f"[debug:interrupt-watch] rms={rms:.5f}  threshold={threshold:.5f}  "
                    f"({marker})  window={sum(window)}/{len(window)}"
                )
            if DEBUG_AEC:
                print(f"[debug:aec] raw_rms={raw_rms:.5f}  post_aec_rms={rms:.5f}  delta={raw_rms - rms:+.5f}")

            if len(window) == INTERRUPT_WINDOW_BLOCKS and sum(window) >= INTERRUPT_WINDOW_TRIGGER:
                sd.stop()
                return pending_blocks

        return None


def _synthesize_group(piper: PiperVoice, text: str) -> tuple[np.ndarray, int] | None:
    """Synthesizes one playback chunk (a group of sentences) with Piper."""
    chunks = list(piper.synthesize(text, syn_config=SPEECH_SYNTHESIS_CONFIG))
    if not chunks:
        return None
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    return audio, chunks[0].sample_rate


def speak_interruptible(text: str) -> str | None:
    """
    Like speak(), but plays `text` in multi-sentence chunks (see
    SENTENCES_PER_CHUNK) and listens in the background for the user to
    barge in - the way Google Home/Alexa handle interruption. This bounds
    interrupt latency to roughly SENTENCES_PER_CHUNK sentences instead of
    the whole response.

    While one chunk plays, the next chunk's synthesis runs concurrently on
    a background thread (see _synthesize_group), so it's already to hand
    the moment playback of the current one ends - no dead-air gap waiting
    on Piper between chunks.

    Returns:
      - None if playback finished with no interruption, OR the user
        interrupted with a stop command ("stop", "cancel", ...) and said
        nothing further within STOP_GRACE_LISTEN_SECONDS - either way
        there's nothing new to send to Claude.
      - The transcribed text of a non-stop-command interruption - or, if
        a stop command was immediately followed by another utterance
        within the grace window (e.g. "stop - actually, turn off the
        lights"), the transcribed text of that follow-up - so the caller
        can feed it straight into the next run_turn() exactly as if the
        user had done normal push-to-talk.
    """
    text = _normalize_symbols_for_speech(_strip_markdown_for_speech(text.strip())).strip()
    if not text:
        return None

    print(f"Speaking: {text}")
    groups = _group_sentences(_split_into_sentences(text))
    if not groups:
        return None

    piper = _get_piper_voice()
    block_samples = int(BLOCK_SECONDS * SAMPLE_RATE)
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    try:
        input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            device=_get_input_device_index(),
            callback=callback,
        )
    except (sd.PortAudioError, RuntimeError) as e:
        print(f"Could not open microphone for interrupt detection: {e}")
        # No mic available for barge-in - just speak the whole thing normally.
        speak(text)
        return None

    result_text: str | None = None

    # Single background worker: synthesizes the *next* chunk while the
    # current one is playing. Torn down with cancel_futures=True/wait=False
    # (rather than a `with` block, which would block shutdown on whatever's
    # mid-synthesis) so an interruption returns immediately instead of
    # waiting on a synthesis job whose audio we're about to discard anyway.
    synth_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        with input_stream:
            _seed_noise_floor(audio_queue, debug_label="barge-in")

            next_synth = synth_executor.submit(_synthesize_group, piper, groups[0])

            for i, _group_text in enumerate(groups):
                synthesized = next_synth.result()

                # Kick off the next chunk's synthesis now, before playing
                # this one, so it overlaps with playback instead of only
                # starting once this chunk is already done.
                if i + 1 < len(groups):
                    next_synth = synth_executor.submit(_synthesize_group, piper, groups[i + 1])

                if synthesized is None:
                    continue
                audio, sample_rate = synthesized

                triggering_blocks = _play_chunk_watching_for_interrupt(
                    audio, sample_rate, audio_queue
                )

                if triggering_blocks is None:
                    continue  # this chunk finished without interruption - play the next one

                print("Interrupted - listening...")
                captured = _capture_until_silence(
                    audio_queue,
                    initial_blocks=triggering_blocks,
                    already_speaking=True,
                    debug_label="barge-in",
                )

                if captured is not None:
                    print("Transcribing...")
                    heard_text = transcribe(captured)
                    if not heard_text:
                        print("Didn't catch that.")
                    elif not _is_stop_command(heard_text):
                        print(f"Heard: {heard_text}")
                        result_text = heard_text
                    else:
                        print(f"Heard: {heard_text}")
                        print("Stopped.")
                        # Grace window: an immediate follow-up ("stop -
                        # actually turn off the lights") becomes the next
                        # turn instead of silently dropping back to the
                        # idle prompt.
                        follow_up = _capture_until_silence(
                            audio_queue,
                            already_speaking=False,
                            debug_label="barge-in-grace",
                            no_speech_timeout=STOP_GRACE_LISTEN_SECONDS,
                        )
                        if follow_up is not None:
                            print("Transcribing...")
                            follow_up_text = transcribe(follow_up)
                            if follow_up_text:
                                print(f"Heard: {follow_up_text}")
                                result_text = follow_up_text

                break  # cancel any remaining chunks regardless of outcome
    finally:
        synth_executor.shutdown(wait=False, cancel_futures=True)

    return result_text

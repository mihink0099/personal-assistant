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

import queue
import re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

# Set to False once voice activity detection is dialed in and you don't
# need to see mic/threshold/RMS details on every recording anymore.
DEBUG = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Whisper's feature extractor expects 16kHz mono audio - recording at that
# rate directly means we never have to resample before transcribing.
SAMPLE_RATE = 16_000
BLOCK_SECONDS = 0.1  # size of each chunk we read from the mic while recording

# Simple energy-based (RMS) voice activity detection - no extra dependency
# beyond what we already need. It measures a moment of background noise
# first, then treats anything meaningfully louder than that as speech.
CALIBRATION_SECONDS = 0.3          # how long to sample ambient noise first
SPEECH_RMS_FLOOR = 0.01            # minimum RMS to ever count as speech
SPEECH_RMS_MULTIPLIER = 3.0        # speech must be this many x louder than ambient noise
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
INTERRUPT_CONSECUTIVE_BLOCKS = 3     # this many over-threshold blocks in a row (~0.3s) = real interruption

# Case-insensitive, checked after stripping trailing punctuation. These
# cancel outright; anything else transcribed during an interruption is
# treated as a redirect - the next user turn - instead.
STOP_PHRASES = {"stop", "shut up", "cancel", "never mind", "quiet", "that's enough"}

# Model objects are slow to construct (loading weights), so we build each
# one at most once per process and reuse it across turns.
_whisper_model: WhisperModel | None = None
_piper_voice: PiperVoice | None = None


# ---------------------------------------------------------------------------
# Recording + voice activity detection
# ---------------------------------------------------------------------------
def _calibrate_threshold(audio_queue: queue.Queue, debug_label: str = "") -> float:
    """
    Samples CALIBRATION_SECONDS of ambient noise from `audio_queue` and
    returns the RMS threshold above which a block counts as speech.
    Shared by record_until_silence() (calibrates against room silence
    before listening) and speak_interruptible() (calibrates once before
    any chunk plays, so Claude's own voice never pollutes the threshold).
    """
    noise_samples = []
    elapsed = 0.0
    while elapsed < CALIBRATION_SECONDS:
        block = audio_queue.get()
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        noise_samples.append(rms)
        elapsed += BLOCK_SECONDS

    noise_floor = max(noise_samples)
    threshold = max(SPEECH_RMS_FLOOR, noise_floor * SPEECH_RMS_MULTIPLIER)
    if DEBUG:
        label = f":{debug_label}" if debug_label else ""
        print(f"[debug{label}] noise_floor={noise_floor:.5f}  threshold={threshold:.5f}")
    return threshold


def _capture_until_silence(
    audio_queue: queue.Queue,
    threshold: float,
    initial_blocks: list[np.ndarray] | None = None,
    already_speaking: bool = False,
    debug_label: str = "",
) -> np.ndarray | None:
    """
    Shared recording loop: reads blocks from `audio_queue` until the
    speaker has gone quiet for SILENCE_HOLD_SECONDS after talking, or
    MAX_RECORD_SECONDS elapses. Used by record_until_silence() (push-to-
    talk, starting from silence) and by speak_interruptible() (continuing
    to record after a barge-in has already been detected mid-playback).

    If `already_speaking` is True, speech is assumed to have already
    started - skips the "wait for speech to begin" phase and its
    NO_SPEECH_TIMEOUT_SECONDS give-up, since we already know the user is
    talking (that's why this was called). `initial_blocks`, if given, are
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
        elapsed += BLOCK_SECONDS
        block_count += 1

        if rms > threshold:
            speech_detected = True
            silence_run = 0.0
        elif speech_detected:
            silence_run += BLOCK_SECONDS

        if DEBUG and block_count % debug_print_every_n_blocks == 0:
            marker = "SPEECH" if rms > threshold else "silence"
            print(f"[debug{label}] rms={rms:.5f}  threshold={threshold:.5f}  ({marker})")

        if speech_detected and silence_run >= SILENCE_HOLD_SECONDS:
            break
        if elapsed >= MAX_RECORD_SECONDS:
            break
        if not speech_detected and elapsed >= NO_SPEECH_TIMEOUT_SECONDS:
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
    if DEBUG:
        try:
            input_device_index = sd.default.device[0]
            print(f"[debug] default input device: {sd.query_devices(input_device_index)}")
        except Exception as e:
            print(f"[debug] could not query default input device: {e}")

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
            callback=callback,
        )
    except sd.PortAudioError as e:
        print(f"Could not open the microphone: {e}")
        return None

    with stream:
        threshold = _calibrate_threshold(audio_queue, debug_label="ptt")
        return _capture_until_silence(audio_queue, threshold, debug_label="ptt")


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
    text = text.strip()
    if not text:
        return

    print(f"Speaking: {text}")
    voice = _get_piper_voice()
    chunks = list(voice.synthesize(text, syn_config=SPEECH_SYNTHESIS_CONFIG))
    if not chunks:
        return

    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    try:
        sd.play(audio, samplerate=chunks[0].sample_rate)
        sd.wait()
    except sd.PortAudioError as e:
        print(f"Could not play audio: {e}")


def _play_chunk_watching_for_interrupt(
    audio: np.ndarray,
    sample_rate: int,
    audio_queue: queue.Queue,
    threshold: float,
) -> list[np.ndarray] | None:
    """
    Plays one chunk via non-blocking sd.play() while watching `audio_queue`
    for the user barging in. Polls sd.get_stream().active rather than a
    time.monotonic() duration estimate, since real playback runs measurably
    longer than len(audio)/sample_rate (buffering latency) - using a time
    estimate would risk cutting the chunk's tail off early on the next
    sd.play() call.

    Returns None if the chunk finished playing without interruption, or
    the list of mic blocks captured so far (starting from the first block
    that crossed the threshold) if the user interrupted - sd.stop() has
    already been called by the time this returns.

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

    try:
        sd.play(audio, samplerate=sample_rate)
    except sd.PortAudioError as e:
        print(f"Could not play audio: {e}")
        return None

    output_stream = sd.get_stream()
    chunk_start = time.monotonic()
    consecutive_speech_blocks = 0
    pending_blocks: list[np.ndarray] = []

    while output_stream.active:
        elapsed = time.monotonic() - chunk_start
        try:
            block = audio_queue.get(timeout=BLOCK_SECONDS)
        except queue.Empty:
            continue

        if elapsed < INTERRUPT_WARMUP_SECONDS:
            continue  # still inside the warm-up guard - ignore this block

        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        if rms > threshold:
            consecutive_speech_blocks += 1
            pending_blocks.append(block)
        else:
            consecutive_speech_blocks = 0
            pending_blocks.clear()

        if DEBUG:
            marker = "SPEECH" if rms > threshold else "silence"
            print(
                f"[debug:interrupt-watch] rms={rms:.5f}  threshold={threshold:.5f}  "
                f"({marker})  streak={consecutive_speech_blocks}"
            )

        if consecutive_speech_blocks >= INTERRUPT_CONSECUTIVE_BLOCKS:
            sd.stop()
            return pending_blocks

    return None


def speak_interruptible(text: str) -> str | None:
    """
    Like speak(), but plays `text` sentence-by-sentence and listens in the
    background for the user to barge in - the way Google Home/Alexa handle
    interruption. This bounds interrupt latency to roughly one sentence
    instead of the whole response.

    Returns:
      - None if playback finished with no interruption, OR the user
        interrupted with a stop-phrase ("stop", "cancel", ...) - either
        way there's nothing new to send to Claude.
      - The transcribed text of a non-stop-phrase interruption, so the
        caller can feed it straight into the next run_turn() exactly as
        if the user had done normal push-to-talk.
    """
    text = text.strip()
    if not text:
        return None

    print(f"Speaking: {text}")
    sentences = _split_into_sentences(text)
    if not sentences:
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
            callback=callback,
        )
    except sd.PortAudioError as e:
        print(f"Could not open microphone for interrupt detection: {e}")
        # No mic available for barge-in - just speak the whole thing normally.
        speak(text)
        return None

    captured: np.ndarray | None = None

    with input_stream:
        threshold = _calibrate_threshold(audio_queue, debug_label="barge-in")

        for sentence in sentences:
            chunks = list(piper.synthesize(sentence, syn_config=SPEECH_SYNTHESIS_CONFIG))
            if not chunks:
                continue
            audio = np.concatenate([chunk.audio_float_array for chunk in chunks])

            triggering_blocks = _play_chunk_watching_for_interrupt(
                audio, chunks[0].sample_rate, audio_queue, threshold
            )

            if triggering_blocks is not None:
                print("Interrupted - listening...")
                captured = _capture_until_silence(
                    audio_queue,
                    threshold,
                    initial_blocks=triggering_blocks,
                    already_speaking=True,
                    debug_label="barge-in",
                )
                break  # cancel any remaining sentences

    if captured is None:
        return None

    print("Transcribing...")
    heard_text = transcribe(captured)
    if not heard_text:
        print("Didn't catch that.")
        return None

    print(f"Heard: {heard_text}")

    if heard_text.strip().lower().rstrip(".!?,") in STOP_PHRASES:
        print("Stopped.")
        return None

    return heard_text

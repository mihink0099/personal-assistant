"""
Acoustic echo cancellation (AEC) for the barge-in interrupt-watch.

While the assistant is speaking (see voice.py's
_play_chunk_watching_for_interrupt), the mic can pick up the assistant's
own voice bleeding out of the speakers. Left uncancelled, that self-echo
can either mask a real interruption or, worse, misfire as one. This module
removes the known-output signal from the mic signal before speech
detection runs on it.

Two backends, auto-selected once at import time - see _select_backend():

  1. aec_audio_processing (preferred) - a real WebRTC AEC3 implementation
     via prebuilt wheels. As of writing, PyPI only ships wheels for
     CPython 3.11-3.13 on Windows (https://pypi.org/project/aec-audio-processing/);
     `pip install aec-audio-processing` on a matching interpreter is all
     that's needed to use it - no code changes. On anything else (e.g.
     this project's Python 3.14 environment) pip falls through to a
     source build, which itself needs Meson/CMake/SWIG and a C++
     toolchain and was not attempted here as a hard requirement - if
     it's missing, this module falls back automatically.

  2. A pure-numpy NLMS (Normalized Least Mean Squares) adaptive filter -
     the fallback whenever (1) isn't importable. It adaptively learns a
     linear FIR model of "reference audio -> what shows up in the mic"
     and subtracts the prediction. Much cruder than a real AEC3 stack (no
     double-talk detection, no nonlinear residual echo suppression, no
     comfort noise) but needs nothing beyond numpy, which this project
     already depends on.

Whichever backend is selected is printed once at import time so it's
visible in the assistant's startup log without having to go look.

IMPORTANT CAVEAT - this cannot be meaningfully validated on headset audio.
A headset mic is acoustically isolated from its own earcups by design, so
there's essentially no echo path for either backend to have anything to
cancel - interrupt-watch will look and sound identical with AEC on or off
in that configuration, which is *not* the same as it working correctly.
To actually exercise (and tune NLMS_FILTER_TAPS/NLMS_STEP_SIZE, or verify
the WebRTC backend's delay estimate) you need a real acoustic echo path:
either genuine external speaker + separate mic hardware, or temporarily
switch Windows' default playback device to the laptop's built-in speakers
(Settings > System > Sound) while keeping a physically separate mic as
input, then test barge-in against that.
"""

import platform

import numpy as np

SAMPLE_RATE = 16_000  # must match voice.SAMPLE_RATE - the rate AEC operates at

# NLMS fallback tuning - see _NLMSEchoCanceller. TUNE THESE if you get a
# real echo path to test against and the fallback isn't converging.
NLMS_FILTER_TAPS = 256        # adaptive filter length in samples (~16ms of echo-path coverage at SAMPLE_RATE)
NLMS_STEP_SIZE = 0.5          # mu: adaptation rate - higher converges faster but is noisier/less stable
# eps: NOT just a divide-by-zero guard - it's the floor on the normalization
# term (norm = dot(history, history) + eps), which caps the effective
# per-sample gain at mu/eps. With float32 audio in roughly [-1, 1] and
# reference blocks that are frequently quiet or briefly silent (between
# words, start of a chunk), dot(history, history) collapses toward zero
# often. At 1e-6 that made mu/norm spike to ~500,000 during those moments,
# so a tiny error*history term produced a massive weight update and the
# filter's weights diverged (measured: post-AEC RMS up to ~100x the raw
# mic RMS - amplifying, not cancelling). 1e-2 caps the worst-case gain at
# mu/eps = 50, low enough to stay stable while quiet, still small enough
# not to meaningfully dampen adaptation once real reference energy is
# present (typical quiet-speech-level dot(history, history) over 256 taps
# is already on this order of magnitude).
NLMS_REGULARIZATION = 1e-2

# WebRTC backend tuning - a rough estimate of the mic/speaker round-trip
# delay in milliseconds (buffering + acoustic travel time). TUNE THIS
# against real hardware; sd.play()'s own buffering latency alone has been
# measured elsewhere in this codebase (see voice.py) at ~11% of nominal
# chunk duration, so 50ms is a starting guess, not a measurement.
WEBRTC_STREAM_DELAY_MS = 50


def resample_reference(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    Lightweight linear-interpolation resampler, so the reference (Piper's
    native output rate, e.g. 22050Hz) can be aligned to the mic's
    SAMPLE_RATE before AEC compares them. Deliberately simple (avoids
    pulling in scipy for just this) - good enough for AEC reference
    alignment, which per the module docstring is already only
    approximately time-aligned; do not reuse this anywhere audio quality
    actually matters (playback, STT input).
    """
    if from_rate == to_rate or len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / from_rate
    new_length = max(1, int(round(duration * to_rate)))
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, num=new_length)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


class _NLMSEchoCanceller:
    """
    Minimal Normalized Least Mean Squares adaptive filter - the fallback
    echo canceller when aec_audio_processing isn't available. Filter
    weights and the reference history persist across calls (the same
    "keeps adapting for the session" pattern voice.py's _NoiseFloorEstimator
    uses), so it keeps tracking the echo path across the whole running
    process rather than re-learning it from scratch on every chunk.
    """

    def __init__(self, taps: int = NLMS_FILTER_TAPS, step_size: float = NLMS_STEP_SIZE):
        self._taps = taps
        self._mu = step_size
        self._weights = np.zeros(taps, dtype=np.float64)
        self._ref_history = np.zeros(taps, dtype=np.float64)

    def process(self, mic_block: np.ndarray, reference_block: np.ndarray) -> np.ndarray:
        """
        Returns `mic_block` with the predicted echo (built from
        `reference_block`, the audio actually being played over the same
        time window) subtracted out. Both must be float32 mono at
        SAMPLE_RATE and the same length.
        """
        mic = mic_block.astype(np.float64)
        ref = reference_block.astype(np.float64)
        cleaned = np.empty_like(mic)

        history = self._ref_history
        weights = self._weights
        mu = self._mu

        # Inherently sequential (each sample's weight update depends on
        # the previous one) - if this can't keep up with BLOCK_SECONDS in
        # real time on your hardware, reduce NLMS_FILTER_TAPS first (a
        # shorter filter is cheaper per sample, at the cost of being able
        # to model a shorter echo path).
        for i in range(len(mic)):
            history[1:] = history[:-1]
            history[0] = ref[i]

            predicted_echo = np.dot(weights, history)
            error = mic[i] - predicted_echo
            cleaned[i] = error

            norm = np.dot(history, history) + NLMS_REGULARIZATION
            weights += (mu / norm) * error * history

        self._ref_history = history
        self._weights = weights
        return cleaned.astype(np.float32)


def _float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _pcm16_bytes_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class _WebRTCEchoCanceller:
    """
    Wraps aec_audio_processing.AudioProcessor (real WebRTC AEC3). The
    library operates on 10ms PCM16 frames and expects the far-end
    (reference/"reverse stream") frame for a given moment fed in before
    the corresponding near-end (mic) frame - see process().
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        from aec_audio_processing import AudioProcessor

        self._frame_samples = sample_rate // 100  # 10ms, the fixed frame size webrtc APM expects
        self._ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
        self._ap.set_stream_format(
            sample_rate_in=sample_rate, channel_count_in=1,
            sample_rate_out=sample_rate, channel_count_out=1,
        )
        self._ap.set_reverse_stream_format(sample_rate, 1)
        self._ap.set_stream_delay(WEBRTC_STREAM_DELAY_MS)

    def process(self, mic_block: np.ndarray, reference_block: np.ndarray) -> np.ndarray:
        frame = self._frame_samples
        n = len(mic_block)
        out = np.empty(n, dtype=np.float32)

        for start in range(0, n, frame):
            end = min(start + frame, n)
            mic_frame = mic_block[start:end]
            ref_frame = reference_block[start:end]

            pad = frame - len(mic_frame)
            if pad:
                # webrtc APM only accepts full 10ms frames - pad the final
                # partial frame with silence and trim the padding back off
                # the result below.
                mic_frame = np.concatenate([mic_frame, np.zeros(pad, dtype=np.float32)])
                ref_frame = np.concatenate([ref_frame, np.zeros(pad, dtype=np.float32)])

            self._ap.process_reverse_stream(_float_to_pcm16_bytes(ref_frame))
            cleaned_bytes = self._ap.process_stream(_float_to_pcm16_bytes(mic_frame))
            cleaned = _pcm16_bytes_to_float(cleaned_bytes)
            out[start:end] = cleaned[: end - start]

        return out


def _select_backend():
    try:
        import aec_audio_processing  # noqa: F401
    except ImportError:
        print(
            "[AEC] aec_audio_processing not installed (prebuilt wheels only cover "
            f"CPython 3.11-3.13 on Windows; this process is running "
            f"{platform.python_version()}). Using the pure-numpy NLMS fallback."
        )
        return _NLMSEchoCanceller()

    try:
        backend = _WebRTCEchoCanceller()
        print("[AEC] Using aec_audio_processing (WebRTC AEC3).")
        return backend
    except Exception as e:
        print(f"[AEC] aec_audio_processing installed but failed to initialize ({e}). "
              "Using the pure-numpy NLMS fallback.")
        return _NLMSEchoCanceller()


# Selected once per process - see module docstring for why there are two
# backends and _select_backend() for the selection logic.
_backend = _select_backend()


def get_backend():
    """Returns the process-wide echo canceller instance selected at import time."""
    return _backend

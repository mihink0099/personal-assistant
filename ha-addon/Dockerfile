# syntax=docker/dockerfile:1
# The line above opts into BuildKit's extended Dockerfile syntax - needed
# for the --mount=type=cache flags below, which persist pip's build cache
# across "Rebuild"s on the Pi even when the RUN layer itself is
# invalidated by a src/ change. Without it, pip re-downloads and
# recompiles every dependency (numpy, faster-whisper, piper-tts,
# sounddevice - most of the build's time) completely from scratch on
# every single rebuild, even ones where none of those dependencies
# actually changed.
#
# Deliberately NOT built from a Home Assistant base image (those are
# Alpine/musl-based, and faster-whisper/piper-tts/sounddevice's native
# dependencies are much easier to get working wheels/builds for on
# Debian/glibc). This means no `bashio` here - run.sh reads add-on options
# straight from /data/options.json with jq instead. See DEPLOY.md for the
# full rationale and the shared-vs-Pi-specific file breakdown.
ARG BUILD_FROM=python:3.11-slim
FROM $BUILD_FROM

# Docker's stdout isn't a terminal, so Python fully buffers print() output
# by default - it can sit invisible in the buffer indefinitely for a
# long-running process like this one instead of showing up in `docker logs`
# / the add-on's Log tab. uvicorn's own logging flushes explicitly (so the
# startup banner always appears), which made this easy to miss: the voice
# loop's prints (including its own error messages) were silently getting
# stuck in the buffer while everything looked fine.
ENV PYTHONUNBUFFERED=1

# - portaudio19-dev / libasound2-dev: PortAudio + ALSA headers, needed to
#   build `sounddevice` against the container's (passed-through) audio
#   devices.
# - ffmpeg: faster-whisper's audio decoding path.
# - git: pulls the shared personal_assistant package - see below.
# - jq: run.sh's options.json reader.
# - curl: handy for manual /health checks from inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev \
        libasound2-dev \
        ffmpeg \
        git \
        jq \
        curl \
    && rm -rf /var/lib/apt/lists/*

# assistant.py, voice.py, home_assistant.py, aec.py, battery_monitor.py
# (i.e. everything under src/personal_assistant/) are NOT duplicated into
# this add-on folder - they're pulled straight from this same GitHub repo
# at build time, so there is exactly one copy of that code to keep in
# sync, ever. Updating shared logic later is just "push to main, then hit
# Rebuild on the add-on in Home Assistant" - see DEPLOY.md.
ARG PERSONAL_ASSISTANT_REPO=https://github.com/mihink0099/personal-assistant.git
ARG PERSONAL_ASSISTANT_REF=main

# Cache-busts the git clone below whenever the repo has new commits that
# actually touch what gets installed here - without this, Docker's layer
# cache would happily keep reusing a stale clone across "Rebuild"s in the
# Home Assistant UI, since `git clone` has no other input that changes.
# Scoped to src/, pyproject.toml, and requirements.txt specifically
# (instead of the whole repo's commit history) so a push that only
# touches ha-addon/ files - which are just COPY'd in below, not cloned -
# doesn't force a full from-scratch re-clone and re-install of
# faster-whisper/piper-tts/sounddevice/numpy, most of which have to
# compile from source on aarch64 and take the bulk of the build time.
ADD https://api.github.com/repos/mihink0099/personal-assistant/commits?sha=$PERSONAL_ASSISTANT_REF&path=src&per_page=1 /tmp/repo-version-src.json
ADD https://api.github.com/repos/mihink0099/personal-assistant/commits?sha=$PERSONAL_ASSISTANT_REF&path=pyproject.toml&per_page=1 /tmp/repo-version-pyproject.json
ADD https://api.github.com/repos/mihink0099/personal-assistant/commits?sha=$PERSONAL_ASSISTANT_REF&path=requirements.txt&per_page=1 /tmp/repo-version-requirements.json

# NOT installing the package's optional [aec] extra (aec-audio-processing,
# real WebRTC AEC3) here: as of writing, that package only ships prebuilt
# wheels for Windows, and this is a Linux container (source-building
# WebRTC needs Meson/CMake/SWIG plus a full C++ toolchain, which isn't
# worth wiring up speculatively for a dependency this narrow). aec.py
# detects the missing import and falls back to its pure-numpy NLMS filter
# automatically - same behavior as the dev machine, just for a different
# reason (no Linux wheel, vs. no Python-3.14 wheel). Check the add-on log
# for "[AEC] Using ..." to confirm.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch "$PERSONAL_ASSISTANT_REF" "$PERSONAL_ASSISTANT_REPO" /opt/personal-assistant \
    && pip install -e /opt/personal-assistant

WORKDIR /app

# Add-on/Pi-specific files - these DO live in this folder (ha-addon/) in
# the repo, so a normal Docker COPY (build context = this folder, per Home
# Assistant's local add-on build convention) picks them up directly.
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# openwakeword needs its own melspectrogram/speech-embedding feature-
# extraction models (separate from the wake-word-specific model in
# models/, below) to turn raw audio into the features its classifier
# reads. Pre-fetch them at build time so wake_word.py never tries to
# download anything the first time it runs on the Pi.
RUN python3 -c "import openwakeword.utils; openwakeword.utils.download_models()"

COPY pi_server.py ./pi_server.py
COPY wake_word.py ./wake_word.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Your custom wake-word model(s) - wake_word.py's default path is
# models/hey_jarvis_custom.onnx, alongside pi_server.py/wake_word.py at
# /app. Override with the WAKE_WORD_MODEL_PATH add-on option if you keep
# yours somewhere else in the image (e.g. a different filename here).
COPY models ./models

EXPOSE 8000

CMD ["/run.sh"]

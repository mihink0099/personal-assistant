"""
Home Assistant add-on entry point: a FastAPI /chat endpoint and a
continuous-listening voice loop, both talking to the same shared Claude
conversation - the always-on Pi equivalent of assistant.py's terminal
loop, minus the terminal.

Deliberately does NOT reimplement any assistant logic: it imports
assistant.py, voice.py, and home_assistant.py (via assistant/voice) from
the personal_assistant package completely unchanged - see DEPLOY.md for
how those get into the Docker image - and just wires them into two entry
points instead of one input() loop. assistant.run_turn() and
voice.listen_for_speech()/voice.speak_interruptible() are exactly the
same functions the CLI uses.

Concurrency: /chat requests and the voice loop share one `messages` list
and one `conversation_lock`. A full turn (append to history, call Claude,
speak the reply) happens inside that lock for both entry points - and for
the voice loop's own command capture (voice.listen_for_speech()), so does
that (see the comment on voice_loop() for why). The practical effect: only
one of {a /chat request, the voice loop actively handling a command} is
ever using the mic or speaker at a time; the other waits its turn - never
both playing or recording at once. The one exception is wake-word
listening itself (see wake_word.py) - that runs outside the lock, since it
can block indefinitely and doesn't touch shared state or play any audio;
see voice_loop()'s comment for the reasoning.

Wake-word gating: the voice loop no longer treats "any sufficiently loud
sound" as a command. It idles in wake_word.wait_for_wake_word() (a
dedicated openwakeword model, not RMS/VAD-based like voice.py's own
detection) until the configured wake word is heard, *then* opens a real
voice.listen_for_speech() capture for the actual command - and returns to
wake-word idling once that turn finishes, not immediately back to
listening for anything loud.
"""

import os
import queue
import threading
import time
import traceback

from dotenv import load_dotenv

# Loaded (and validated below) before importing assistant.py, whose
# module-level code constructs the Anthropic client from ANTHROPIC_API_KEY
# at import time - see home_assistant.py's own load_dotenv() comment for
# why this ordering matters; the same import-order pitfall applies here.
load_dotenv()

_MISSING = [name for name in ("ANTHROPIC_API_KEY", "HA_TOKEN") if not os.getenv(name)]
if _MISSING:
    raise SystemExit(
        f"Missing required environment variable(s): {', '.join(_MISSING)}. "
        "Set them in the add-on's Configuration tab (see DEPLOY.md)."
    )

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from personal_assistant import assistant, voice

app = FastAPI(title="Personal Assistant")

# Shared, mutated in place by assistant.run_turn() - the same object read
# and appended to by both the /chat endpoint and the voice loop, always
# under conversation_lock.
messages: list = []
conversation_lock = threading.Lock()

# A barge-in follow-up (see voice.speak_interruptible()'s return value)
# captured while speaking a /chat-triggered reply still needs to become
# the voice loop's *next* turn, exactly as it would after a voice-
# triggered reply - this hands it across threads instead of just being a
# local variable in whichever function happened to trigger the speech.
pending_voice_turns: "queue.Queue[str]" = queue.Queue()

ENABLE_VOICE = os.getenv("ENABLE_VOICE", "true").strip().lower() not in ("false", "0", "no")


class ChatRequest(BaseModel):
    message: str
    speak: bool = True  # also speak the reply aloud through the Pi's speaker, not just return it as JSON


class ChatResponse(BaseModel):
    reply: str


def _run_turn_locked(user_text: str, speak_response: bool) -> str:
    """
    Runs one turn against the shared history and, optionally, speaks the
    reply. Assumes conversation_lock is already held by the caller - see
    run_chat_turn() and voice_loop().
    """
    history_len_before_turn = len(messages)
    messages.append({"role": "user", "content": user_text})
    try:
        final_text = assistant.run_turn(messages)
    except Exception as e:
        del messages[history_len_before_turn:]
        raise RuntimeError(f"Something went wrong talking to Claude: {e}") from e

    if speak_response and final_text:
        try:
            follow_up = voice.speak_interruptible(final_text)
            if follow_up:
                pending_voice_turns.put(follow_up)
        except Exception as e:
            print(f"[pi_server] Speaking failed: {e}")

    return final_text


def run_chat_turn(user_text: str, speak_response: bool) -> str:
    """Entry point for the /chat endpoint - acquires conversation_lock itself."""
    with conversation_lock:
        return _run_turn_locked(user_text, speak_response)


def _process_voice_turn(user_text: str) -> None:
    """Runs one turn under conversation_lock - shared by both code paths below."""
    with conversation_lock:
        try:
            _run_turn_locked(user_text, speak_response=True)
        except Exception as e:
            print(f"[pi_server] Turn failed: {e}")


def voice_loop() -> None:
    """
    The Pi's always-on ear: idles on the wake word, then listens for a
    command (or picks up a barge-in follow-up left behind by a previous
    turn - see pending_voice_turns) and runs it as a turn - then goes back
    to idling on the wake word, not straight back into listening for
    anything loud. There's no typed-input fallback here (no terminal on a
    headless add-on) - if nothing was heard after the wake word, it just
    goes back to idling.

    Imports wake_word lazily (here, not at module scope) so that
    ENABLE_VOICE=false deployments (/chat only, no mic) never load
    openwakeword or require a wake word model file to be present at all -
    see wake_word.py's own note about failing loudly rather than silently
    falling back to a different built-in model when one IS expected.

    The import (and everything below it) is wrapped in a try/except that
    catches SystemExit explicitly, alongside Exception: wake_word.py's
    _load_model() raises SystemExit if the model file or audio device
    isn't found, and Python's default behavior silently swallows
    SystemExit raised inside a daemon thread - no traceback, no message,
    the thread just quietly dies while the rest of the add-on looks fine.
    Catching it here means a startup failure is always visible in the log.
    """
    try:
        import wake_word

        print("[pi_server] Continuous voice listening started (wake-word gated).")
        while True:
            try:
                follow_up = pending_voice_turns.get_nowait()
            except queue.Empty:
                follow_up = None

            if follow_up is not None:
                # A barge-in follow-up from a previous turn - the user already
                # said this without a fresh wake word (same as the CLI's
                # barge-in behavior), so skip straight to processing it
                # instead of waiting to hear the wake word again.
                _process_voice_turn(follow_up)
                time.sleep(0.05)
                continue

            # Idle state: wait for the wake word. Deliberately NOT holding
            # conversation_lock here, unlike the command-capture step below -
            # this can block indefinitely (minutes or hours between wake
            # words), and holding a shared lock for an unbounded wait would
            # starve /chat almost permanently. wait_for_wake_word() doesn't
            # touch shared state or play any audio, so the only real cost of
            # not locking it is a narrow, low-probability window where it's
            # also listening while a /chat-triggered reply is being spoken
            # elsewhere - accepted, since the assistant's own voice saying the
            # actual wake word mid-reply is exceedingly unlikely.
            try:
                wake_word.wait_for_wake_word()
            except Exception:
                print("[pi_server] Wake-word listening failed:")
                traceback.print_exc()
                time.sleep(1)
                continue

            # From here through the end of the turn, hold conversation_lock -
            # same "has the mic/speaker = has the lock" reasoning as /chat,
            # now scoped to just the active-turn window rather than the whole
            # (now-unbounded) loop iteration.
            with conversation_lock:
                try:
                    user_text = voice.listen_for_speech()
                except Exception as e:
                    print(f"[pi_server] Listening failed: {e}")
                    user_text = None

                if user_text:
                    try:
                        _run_turn_locked(user_text, speak_response=True)
                    except Exception as e:
                        print(f"[pi_server] Turn failed: {e}")

            time.sleep(0.05)
    except (SystemExit, Exception):
        print("[pi_server] Voice loop failed to start:")
        traceback.print_exc()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message must not be empty.")
    try:
        reply = run_chat_turn(text, request.speak)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return ChatResponse(reply=reply)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "voice_enabled": ENABLE_VOICE}


if __name__ == "__main__":
    import uvicorn

    if ENABLE_VOICE:
        threading.Thread(target=voice_loop, daemon=True).start()
    else:
        print("[pi_server] ENABLE_VOICE is off - /chat only, no mic listening.")

    uvicorn.run(app, host="0.0.0.0", port=8000)

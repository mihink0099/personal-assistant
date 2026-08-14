"""
Terminal chat loop for a Claude-powered smart home assistant.

Architecture in one paragraph: every turn, we send Claude the full
conversation history plus a list of tools it's allowed to call. Claude
either replies with text (we're done for this turn) or asks to call one
or more tools (stop_reason == "tool_use"). When that happens, we run the
corresponding Python function ourselves, hand the result back to Claude
as a new message, and ask again - repeating until Claude stops asking
for tools and gives a final text answer.
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from anthropic import Anthropic

from . import home_assistant as ha
from . import voice

LOCAL_TIMEZONE = ZoneInfo("Australia/Melbourne")

# Windows terminals often default to a legacy codepage (cp1252) that can't
# print emoji or other non-ASCII characters Claude might use in a reply.
# Force UTF-8 on stdout so print() doesn't crash on those.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Using the cheapest model while iterating on the assistant's logic - swap
# back to something like "claude-sonnet-4-6" once things are working well
# and you want better judgment (e.g. entity matching, tool-use decisions).
MODEL = "claude-haiku-4-5"

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
# web_search is a *server-side* tool: Claude and Anthropic's servers handle
# the actual searching between themselves. We just declare it exists - there
# is no Python function for it and it never shows up in our dispatch table
# below, because we never have to execute it ourselves.
#
# The newer "web_search_20260209" tool version (with built-in dynamic
# filtering) only works on Opus 4.6+ and Sonnet 4.6+ tier models - Haiku
# 4.5 rejects it with a 400. Use the older/basic version on Haiku, and the
# newer one on anything from Sonnet 4.6 up.
_CHEAP_MODELS_NEEDING_BASIC_WEB_SEARCH = {"claude-haiku-4-5"}
_WEB_SEARCH_TYPE = (
    "web_search_20250305"
    if MODEL in _CHEAP_MODELS_NEEDING_BASIC_WEB_SEARCH
    else "web_search_20260209"
)
WEB_SEARCH_TOOL = {"type": _WEB_SEARCH_TYPE, "name": "web_search"}

# The rest are *client-side* (custom) tools: Claude only describes what it
# wants to call and with what arguments. Running the function and returning
# the result is entirely our job - that's the dispatch table further down.
HOME_ASSISTANT_TOOLS = [
    {
        "name": "list_entities",
        "description": (
            "List every entity Home Assistant knows about: its entity_id, "
            "friendly name, domain (light, switch, sensor, etc.), and current "
            "state. Call this whenever you need to find the entity_id for "
            "something the user described in plain language (e.g. 'the "
            "bedroom light') and you don't already know the exact id from "
            "earlier in the conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_entity_state",
        "description": (
            "Get the current state and attributes of a single Home Assistant "
            "entity by its exact entity_id (e.g. 'light.bedroom_lamp', "
            "'sensor.living_room_temperature')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The exact Home Assistant entity_id, e.g. 'light.bedroom_lamp'.",
                }
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "turn_on_light",
        "description": "Turn on a specific light in Home Assistant by its exact entity_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The exact entity_id of the light to turn on, e.g. 'light.bedroom_lamp'.",
                }
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "turn_off_light",
        "description": "Turn off a specific light in Home Assistant by its exact entity_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The exact entity_id of the light to turn off, e.g. 'light.bedroom_lamp'.",
                }
            },
            "required": ["entity_id"],
        },
    },
]

ALL_TOOLS = [WEB_SEARCH_TOOL] + HOME_ASSISTANT_TOOLS

# Maps a tool name to the Python function that implements it. web_search is
# deliberately absent - Anthropic's servers execute it, so there's nothing
# for us to dispatch to.
TOOL_DISPATCH = {
    "list_entities": lambda **kwargs: ha.list_entities(),
    "get_entity_state": lambda **kwargs: ha.get_entity_state(kwargs["entity_id"]),
    "turn_on_light": lambda **kwargs: ha.turn_on_light(kwargs["entity_id"]),
    "turn_off_light": lambda **kwargs: ha.turn_off_light(kwargs["entity_id"]),
}

SYSTEM_PROMPT_TEMPLATE = """You are a helpful smart home voice/text assistant controlling a Home \
Assistant instance, with the ability to search the web.

Tool usage rules:
- For home control requests (turning things on/off, checking a device's \
status), go straight to the relevant tool. Do not use web_search for these.
- If you don't already know the exact entity_id for something the user \
described (e.g. "the bedroom light"), call list_entities first, find the \
best matching entity, and then call the relevant tool with that entity_id. \
Prefer matching on friendly_name and domain.
- If multiple entities could plausibly match and it's genuinely ambiguous, \
ask the user to clarify instead of guessing.
- Only use web_search for questions about current events, prices, release \
dates, or other information that could have changed since your training \
and that you're not confident about.
- For general knowledge questions you're confident about, just answer \
directly without searching or using any tool.
- You already know the current date and time (given above) - never say you \
don't have access to it, and don't call web_search just to find out what \
day or time it is.
"""


def build_system_prompt() -> str:
    """
    Builds the system prompt fresh on every call, stamping in the current
    date and time. Without this, the prompt would be frozen at whatever
    moment the script started, and a long-running session would drift out
    of sync with reality - or, since Claude has no built-in clock, it would
    have no idea what "today" means at all and would (correctly, from its
    point of view) say it can't answer date/time questions.
    """
    now = datetime.now(LOCAL_TIMEZONE)
    # %A = full weekday name, %-d avoids a zero-padded day on most platforms.
    # %Z/%z give the timezone name and offset so "in 3 hours" style math is
    # unambiguous too.
    current_time = now.strftime("%A, %B %d, %Y, %I:%M %p %Z (%z)")
    date_line = f"The current date and time is {current_time}, in the Australia/Melbourne timezone.\n\n"
    return date_line + SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------
def print_tool_call(name: str, input_data: dict) -> None:
    print(f"\033[90m  -> calling {name}({input_data})\033[0m")


def print_tool_result(result: str) -> None:
    # Trim long results (like list_entities on a big HA instance) so the
    # terminal stays readable; the full result still goes to Claude.
    preview = result if len(result) <= 400 else result[:400] + "... [truncated for display]"
    print(f"\033[90m  <- result: {preview}\033[0m")


# ---------------------------------------------------------------------------
# The agentic loop
# ---------------------------------------------------------------------------
def run_turn(messages: list) -> str:
    """
    Sends `messages` to Claude and keeps looping - executing tool calls and
    feeding results back - until Claude responds with plain text instead of
    a tool request. Mutates `messages` in place so the caller's conversation
    history stays up to date. Returns Claude's final text reply so the
    caller can speak it aloud.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            # Rebuilt on every request (not just once at startup) so the
            # date/time stamped into it never goes stale, including across
            # the multiple tool-call round trips a single user turn can take.
            system=build_system_prompt(),
            tools=ALL_TOOLS,
            messages=messages,
        )

        # The assistant's turn (text, tool_use blocks, or both) becomes part
        # of the conversation history regardless of what happens next.
        messages.append({"role": "assistant", "content": response.content})

        # Claude often splits a reply into several small text blocks
        # (e.g. interleaved with web_search results) - buffer consecutive
        # text blocks and print them as one "Claude:" line instead of one
        # per block.
        text_buffer = []

        def flush_text() -> None:
            if text_buffer:
                print(f"Claude: {''.join(text_buffer)}")
                text_buffer.clear()

        if response.stop_reason != "tool_use":
            # Claude is done - print whatever text it produced and return
            # control to the user. (web_search results are also printed
            # here since they arrive as part of response.content already
            # resolved by Anthropic's servers.)
            for block in response.content:
                if block.type == "text" and block.text:
                    text_buffer.append(block.text)
            final_text = "".join(text_buffer)
            flush_text()
            return final_text

        # stop_reason == "tool_use": collect every tool_use block Claude
        # asked for (there can be more than one in a single response) and
        # run each one.
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text:
                # Claude sometimes narrates before calling a tool - buffer it.
                text_buffer.append(block.text)

            elif block.type == "server_tool_use":
                flush_text()
                # web_search's own call - nothing for us to execute.
                print_tool_call(block.name, block.input)

            elif block.type == "tool_use":
                flush_text()
                print_tool_call(block.name, block.input)
                func = TOOL_DISPATCH.get(block.name)
                if func is None:
                    result_text = f"Error: no such tool '{block.name}'."
                else:
                    try:
                        result_text = func(**block.input)
                    except Exception as e:
                        result_text = f"Error running tool '{block.name}': {e}"

                print_tool_result(result_text)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        # Any trailing narration after the last tool_use block in this
        # response (rare, but possible).
        flush_text()

        # All of this turn's tool results go back as a single user message,
        # then we loop and ask Claude again.
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # Only server-side tools (web_search) were used - nothing for
            # us to send back, Anthropic's servers already resolved them.
            # Loop again in case Claude has more to say.
            continue


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set - check your .env file.")
        return
    if not os.getenv("HA_TOKEN"):
        print("HA_TOKEN is not set - check your .env file.")
        return

    print("Smart home assistant ready.")
    print("Press Enter with nothing typed to talk (push-to-talk), or type a message directly.")
    print("Type 'exit' to quit.\n")

    messages: list = []
    pending_user_input: str | None = None

    while True:
        if pending_user_input is not None:
            # A barge-in during the previous reply produced a new question
            # directly (speak_interruptible() already transcribed it) -
            # treat it exactly like a normal push-to-talk turn, without
            # prompting again.
            user_input = pending_user_input
            pending_user_input = None
        else:
            try:
                typed = input("You (Enter to talk, or type): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if typed:
                user_input = typed
            else:
                # Enter was pressed with nothing typed - push-to-talk.
                try:
                    user_input = voice.listen_for_speech()
                except Exception as e:
                    print(f"\033[91mVoice input failed: {e}\033[0m")
                    user_input = None

                if not user_input:
                    # No speech detected, empty transcription, or a voice
                    # error - fall back to typing so the user is never stuck.
                    try:
                        user_input = input("Type your message instead: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nBye.")
                        break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        # Remember how long history was before this turn. If anything below
        # fails partway through a multi-tool-call turn, we roll all the way
        # back to this point rather than just popping the last message -
        # otherwise a failure after a tool call could leave a dangling
        # tool_use block with no matching tool_result, which would break
        # every subsequent request in this session.
        history_len_before_turn = len(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            final_text = run_turn(messages)
        except Exception as e:
            print(f"\033[91mSomething went wrong talking to Claude: {e}\033[0m")
            del messages[history_len_before_turn:]
            continue

        if final_text:
            try:
                pending_user_input = voice.speak_interruptible(final_text)
            except Exception as e:
                print(f"\033[91mVoice output failed: {e}\033[0m")


if __name__ == "__main__":
    main()

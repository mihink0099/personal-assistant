"""
Standalone diagnostic for an intermittent "connection refused" issue
reaching Home Assistant directly at 192.168.1.26:8123.

Not part of the main assistant - run this on its own while the issue is
happening. Every INTERVAL_SECONDS, for TOTAL_DURATION_SECONDS total, it
runs two independent checks:

  1. A raw TCP connect to the host:port. This proves whether *anything*
     is listening there at all, independent of HTTP or Home Assistant
     itself - useful for telling "nothing is listening on this port"
     apart from "something's listening but the HTTP server is unhappy".
  2. An HTTP GET to '/'. This proves the actual HA web server is
     answering, and captures the exact exception (connection refused,
     timeout, or something else) when it isn't.

Every attempt is printed with a timestamp and appended to
ha_diagnostic_log.txt in this same folder, so the full log can be shared
afterward. A summary at the end reports the longest unbroken run of
failures with its start/end timestamps - useful for correlating against
things like Wi-Fi roaming, DHCP renewals, or HA/add-on restarts.
"""

import socket
import time
from datetime import datetime
from pathlib import Path

import requests

HOST = "192.168.1.26"
PORT = 8123
URL = f"http://{HOST}:{PORT}/"

INTERVAL_SECONDS = 2
TOTAL_DURATION_SECONDS = 5 * 60
TIMEOUT_SECONDS = 3

LOG_PATH = Path(__file__).parent / "ha_diagnostic_log.txt"


def check_tcp() -> tuple[bool, float, str]:
    """
    Attempts a raw TCP connect to HOST:PORT.
    Returns (success, duration_ms, detail). detail is empty on success,
    or "<ExceptionType>: <message>" on failure (e.g. a refused connection
    raises ConnectionRefusedError, a dead/firewalled host times out with
    TimeoutError - the exception type itself tells them apart).
    """
    start = time.monotonic()
    try:
        with socket.create_connection((HOST, PORT), timeout=TIMEOUT_SECONDS):
            return True, (time.monotonic() - start) * 1000, ""
    except Exception as e:
        return False, (time.monotonic() - start) * 1000, f"{type(e).__name__}: {e}"


def check_http() -> tuple[bool, str]:
    """
    Attempts an HTTP GET to URL.
    Returns (success, detail). detail is "<status_code> <reason>" on
    success, or "<ExceptionType>: <message>" on failure. requests uses
    distinct exception types for connect-timeout, read-timeout, and
    connection errors (which includes "connection refused"), so the
    type name alone distinguishes them.
    """
    try:
        response = requests.get(URL, timeout=TIMEOUT_SECONDS)
        return True, f"{response.status_code} {response.reason}"
    except requests.exceptions.ConnectTimeout as e:
        return False, f"ConnectTimeout: {e}"
    except requests.exceptions.ReadTimeout as e:
        return False, f"ReadTimeout: {e}"
    except requests.exceptions.ConnectionError as e:
        # Covers connection-refused among other low-level connect failures;
        # the underlying OS error (e.g. WinError 10061) is inside the message.
        return False, f"ConnectionError: {e}"
    except requests.exceptions.RequestException as e:
        return False, f"{type(e).__name__}: {e}"


def format_line(
    timestamp: str,
    tcp_ok: bool,
    tcp_ms: float,
    tcp_detail: str,
    http_ok: bool,
    http_detail: str,
) -> str:
    tcp_part = f"TCP: OK ({tcp_ms:.0f}ms)" if tcp_ok else f"TCP: FAIL ({tcp_ms:.0f}ms, {tcp_detail})"
    http_part = f"HTTP: {http_detail}" if http_ok else f"HTTP: FAIL ({http_detail})"
    return f"{timestamp} | {tcp_part} | {http_part}"


def summarize(attempts: list[tuple[str, bool]]) -> str:
    """
    attempts is a list of (timestamp, failed) pairs, in order. Returns a
    printable summary: total attempts, how many failed, and the longest
    unbroken run of failures with its start/end timestamps.
    """
    total = len(attempts)
    failed_count = sum(1 for _, failed in attempts if failed)

    longest_run_len = 0
    longest_run_start = None
    longest_run_end = None
    current_run_len = 0
    current_run_start = None

    for timestamp, failed in attempts:
        if failed:
            if current_run_len == 0:
                current_run_start = timestamp
            current_run_len += 1
            if current_run_len > longest_run_len:
                longest_run_len = current_run_len
                longest_run_start = current_run_start
                longest_run_end = timestamp
        else:
            current_run_len = 0

    lines = [
        "",
        "=== Summary ===",
        f"Total attempts: {total}",
        f"Failed attempts: {failed_count} ({(failed_count / total * 100) if total else 0:.1f}%)",
    ]
    if longest_run_len:
        lines.append(
            f"Longest continuous failure run: {longest_run_len} attempts "
            f"({longest_run_start} to {longest_run_end})"
        )
    else:
        lines.append("Longest continuous failure run: none - every attempt succeeded")

    return "\n".join(lines)


def main() -> None:
    print(f"Diagnosing {HOST}:{PORT} every {INTERVAL_SECONDS}s for {TOTAL_DURATION_SECONDS // 60} minutes.")
    print(f"Logging to {LOG_PATH}")
    print("Press Ctrl+C to stop early - a summary will still be printed for what was collected.\n")

    attempts: list[tuple[str, bool]] = []
    end_time = time.monotonic() + TOTAL_DURATION_SECONDS

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write(f"Home Assistant diagnostic log - started {datetime.now().isoformat()}\n")
        log_file.write(f"Target: {HOST}:{PORT}\n\n")
        log_file.flush()

        try:
            while time.monotonic() < end_time:
                loop_start = time.monotonic()
                now = datetime.now()
                timestamp = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"

                tcp_ok, tcp_ms, tcp_detail = check_tcp()
                http_ok, http_detail = check_http()

                line = format_line(timestamp, tcp_ok, tcp_ms, tcp_detail, http_ok, http_detail)
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

                attempts.append((timestamp, (not tcp_ok) or (not http_ok)))

                # Sleep for whatever's left of the interval, accounting for
                # how long the two checks themselves took (they can each
                # take up to TIMEOUT_SECONDS if the host is unresponsive).
                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, INTERVAL_SECONDS - elapsed))
        except KeyboardInterrupt:
            print("\nStopped early.")

        summary_text = summarize(attempts)
        print(summary_text)
        log_file.write(summary_text + "\n")


if __name__ == "__main__":
    main()

"""Read joystick / gamepad events from Linux /dev/input/js* devices.

No external dependencies — parses the 8‑byte ``struct js_event`` directly.
"""

from __future__ import annotations

import os
import struct
import threading
from collections import defaultdict
from pathlib import Path

_JS_EVENT = struct.Struct("IhBB")  # time, value, type, number
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02


class JoystickReader:
    """Background reader that polls /dev/input/js0 for button + axis events.

    Parameters
    ----------
    device_path : Path or str
        Path to the joystick device (default /dev/input/js0).
    poll_interval : float
        Seconds between polls (default 0.01 = 100 Hz).
    """

    def __init__(
        self,
        device_path: str | Path = "/dev/input/js0",
        poll_interval: float = 0.01,
    ):
        self._path = Path(device_path)
        self._interval = poll_interval
        self._axes: dict[int, float] = defaultdict(float)   # axis# → normalised [-1,1]
        self._buttons: dict[int, bool] = defaultdict(bool)   # button# → pressed?
        self._buttons_oneshot: set[int] = set()               # buttons that fired since last consume
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ── public API ──────────────────────────────────────────

    def start(self) -> bool:
        """Open the device and begin background polling.

        Returns False if the device could not be opened.
        """
        if not self._path.exists():
            return False
        try:
            self._fd = os.open(str(self._path), os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop polling and close the device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if hasattr(self, "_fd"):
            os.close(self._fd)

    @property
    def connected(self) -> bool:
        return self._running

    def axis(self, num: int, default: float = 0.0) -> float:
        """Return the current normalised value of axis *num* ([-1, 1])."""
        with self._lock:
            return self._axes.get(num, default)

    def button_pressed(self, num: int) -> bool:
        """Return True if button *num* is currently held."""
        with self._lock:
            return self._buttons.get(num, False)

    def button_oneshot(self, num: int) -> bool:
        """Return True once per press of button *num* (edge‑triggered)."""
        with self._lock:
            if num in self._buttons_oneshot:
                self._buttons_oneshot.discard(num)
                return True
        return False

    # ── internals ───────────────────────────────────────────

    def _poll_loop(self) -> None:
        buf = bytearray()
        while self._running:
            try:
                data = os.read(self._fd, 256)
            except BlockingIOError:
                data = b""
            except OSError:
                break

            if not data:
                import time
                time.sleep(self._interval)
                continue

            buf.extend(data)
            while len(buf) >= _JS_EVENT.size:
                pkt = buf[:_JS_EVENT.size]
                del buf[:_JS_EVENT.size]
                time_us, value, etype, number = _JS_EVENT.unpack(pkt)
                with self._lock:
                    if etype & _JS_EVENT_BUTTON:
                        pressed = bool(value)
                        if pressed and not self._buttons.get(number):
                            self._buttons_oneshot.add(number)
                        self._buttons[number] = pressed
                    elif etype & _JS_EVENT_AXIS:
                        # Normalise: Linux axis range is typically [-32767, 32767]
                        self._axes[number] = max(-1.0, min(1.0, value / 32767.0))

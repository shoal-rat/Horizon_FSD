"""
telemetry_receiver.py - Horizon FSD, Phase 1

Background UDP listener that keeps the latest parsed ForzaTelemetry packet. The
env reads `latest()` each control step; the actual recv runs on a daemon thread
so it never blocks the control loop.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

from forza_telemetry import ForzaTelemetry, parse

logger = logging.getLogger(__name__)


class TelemetryReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 9999, recv_timeout: float = 1.0) -> None:
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(recv_timeout)

        self._latest: Optional[ForzaTelemetry] = None
        self._recv_t: Optional[float] = None   # perf_counter when _latest was received (freshness clock)
        self._packets = 0
        self._bad = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="telemetry-rx", daemon=True)
        self._thread.start()
        logger.info("Telemetry receiver listening on udp://%s:%d", self.host, self.port)

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                telemetry = parse(data)
            except ValueError:
                with self._lock:
                    self._bad += 1
                    bad, good = self._bad, self._packets
                if bad % 50 == 1:               # periodic, so a wrong-format flood is visible at runtime
                    logger.warning("telemetry: %d bad packets (last len=%d, good=%d) - wrong Data Out "
                                   "format/port? expecting the 324-byte 'Car Dash' format", bad, len(data), good)
                continue
            with self._lock:
                self._latest = telemetry
                self._recv_t = time.perf_counter()
                self._packets += 1

    def latest(self, max_age: Optional[float] = None) -> Optional[ForzaTelemetry]:
        """Most recent good packet, or None. With max_age set, also None if the stream has gone
        STALE (no fresh packet within max_age s) - so a frozen game/alt-tab can't feed phantom data."""
        with self._lock:
            t, rt = self._latest, self._recv_t
        if t is None:
            return None
        if max_age is not None and (rt is None or time.perf_counter() - rt > max_age):
            return None
        return t

    def age(self) -> Optional[float]:
        """Seconds since the last good packet (None if none yet)."""
        with self._lock:
            rt = self._recv_t
        return None if rt is None else time.perf_counter() - rt

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packets

    @property
    def bad_count(self) -> int:
        with self._lock:
            return self._bad

    def wait_for_packet(self, timeout: float = 5.0, max_age: float = 0.5) -> bool:
        """Wait until a FRESH packet has arrived (received within max_age s), not merely until one
        was ever seen - so a frozen stream reads as 'no live telemetry', not a stale success."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.latest(max_age=max_age) is not None:
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except Exception:  # pragma: no cover
            pass

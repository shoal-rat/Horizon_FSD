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
                continue
            with self._lock:
                self._latest = telemetry
                self._packets += 1

    def latest(self) -> Optional[ForzaTelemetry]:
        with self._lock:
            return self._latest

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packets

    @property
    def bad_count(self) -> int:
        with self._lock:
            return self._bad

    def wait_for_packet(self, timeout: float = 5.0) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.latest() is not None:
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except Exception:  # pragma: no cover
            pass

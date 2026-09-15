"""OneEuroFilter and KeypointStabilizer for temporal keypoint smoothing.

Uses 1-Euro adaptive filtering (Casiez et al., CHI 2012) to eliminate jitter
when a person is holding still while providing low lag during fast movements.
"""

from __future__ import annotations

import math
from typing import Sequence

Keypoint = tuple[float, float, float]


class LowPassFilter:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = float(alpha)
        self.s: float | None = None

    def reset(self) -> None:
        self.s = None

    def filter(self, val: float, alpha: float | None = None) -> float:
        if alpha is not None:
            self.alpha = float(alpha)
        if self.s is None:
            self.s = float(val)
        else:
            self.s = self.alpha * float(val) + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter:
    """Adaptive 1-Euro low-pass filter with dynamic cutoff frequency."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 0.5,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

    def filter(self, val: float, timestamp: float | None = None) -> float:
        if self.last_time is None or timestamp is None:
            dt = 1.0 / self.freq if self.freq > 0 else 0.033
        else:
            dt = max(timestamp - self.last_time, 1e-4)
        if timestamp is not None:
            self.last_time = timestamp

        prev_x = self.x_filter.s
        dx = 0.0 if prev_x is None else (val - prev_x) / dt
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(val, self._alpha(cutoff, dt))


class KeypointStabilizer:
    """Stabilizes 17 COCO pose keypoints using independent OneEuro filters."""

    def __init__(
        self,
        num_keypoints: int = 17,
        freq: float = 30.0,
        min_cutoff: float = 0.5,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
        conf_floor: float = 0.15,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.conf_floor = conf_floor
        self.filters_x = [
            OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(num_keypoints)
        ]
        self.filters_y = [
            OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(num_keypoints)
        ]

    def reset(self) -> None:
        for fx in self.filters_x:
            fx.reset()
        for fy in self.filters_y:
            fy.reset()

    def filter_keypoints(
        self,
        keypoints: Sequence[Keypoint],
        timestamp: float | None = None,
    ) -> list[Keypoint]:
        if not keypoints:
            return []
        stabilized: list[Keypoint] = []
        for i, pt in enumerate(keypoints):
            if i >= self.num_keypoints:
                stabilized.append((float(pt[0]), float(pt[1]), float(pt[2])))
                continue
            x, y, c = float(pt[0]), float(pt[1]), float(pt[2])
            if c < self.conf_floor:
                stabilized.append((x, y, c))
            else:
                sx = self.filters_x[i].filter(x, timestamp)
                sy = self.filters_y[i].filter(y, timestamp)
                stabilized.append((sx, sy, c))
        return stabilized

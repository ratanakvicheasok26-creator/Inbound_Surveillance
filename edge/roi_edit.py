"""Click-drag ROI on the OpenCV preview: move, resize handles, or draw a new box."""

from __future__ import annotations

import re
from pathlib import Path

import cv2

from occupancy import pixels_to_roi, roi_to_pixels

HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
HANDLE_EDGES = {
    "nw": ("n", "w"),
    "n": ("n",),
    "ne": ("n", "e"),
    "e": ("e",),
    "se": ("s", "e"),
    "s": ("s",),
    "sw": ("s", "w"),
    "w": ("w",),
}
_ROI_LINE = re.compile(r"^roi:\s*\[[^\]]*\]\s*$", re.M)


def handle_points(x1: int, y1: int, x2: int, y2: int) -> dict[str, tuple[int, int]]:
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    return {
        "nw": (x1, y1),
        "n": (mx, y1),
        "ne": (x2, y1),
        "e": (x2, my),
        "se": (x2, y2),
        "s": (mx, y2),
        "sw": (x1, y2),
        "w": (x1, my),
    }


def handle_radius(frame_w: int, frame_h: int) -> int:
    return max(18, int(0.03 * min(frame_w, frame_h)))


def hit_handle(
    x: int,
    y: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
) -> str | None:
    best: str | None = None
    best_d = radius * radius
    for name, (hx, hy) in handle_points(x1, y1, x2, y2).items():
        d = (x - hx) ** 2 + (y - hy) ** 2
        if d <= best_d:
            best_d = d
            best = name
    return best


def point_in_rect(x: int, y: int, x1: int, y1: int, x2: int, y2: int) -> bool:
    return x1 <= x <= x2 and y1 <= y <= y2


def clamp_xy(x: int, y: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    return max(0, min(frame_w - 1, x)), max(0, min(frame_h - 1, y))


def norm_rect(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def format_roi(roi: list[float]) -> str:
    return f"roi: [{roi[0]:.4f}, {roi[1]:.4f}, {roi[2]:.4f}, {roi[3]:.4f}]"


def persist_roi(path: Path, roi: list[float]) -> None:
    line = format_roi(roi)
    if path.exists():
        text = path.read_text()
        new, n = _ROI_LINE.subn(line, text, count=1)
        if n == 0:
            new = text.rstrip() + "\n\n" + line + "\n"
        path.write_text(new)
    else:
        path.write_text(line + "\n")


class RoiEditor:
    def __init__(self, roi: list[float], win: str, save_path: Path | None = None) -> None:
        self.roi = [float(v) for v in roi]
        self.win = win
        self.save_path = save_path
        self.frame_w = 1
        self.frame_h = 1
        self._mode: str | None = None
        self._handle: str | None = None
        self._origin = (0, 0)
        self._start_px: tuple[int, int, int, int] | None = None
        self._live_px: tuple[int, int, int, int] | None = None

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_w = max(1, width)
        self.frame_h = max(1, height)

    def pixels(self) -> tuple[int, int, int, int]:
        if self._live_px is not None:
            return norm_rect(*self._live_px)
        return roi_to_pixels(self.frame_w, self.frame_h, self.roi)

    def _commit(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.roi = pixels_to_roi(x1, y1, x2, y2, self.frame_w, self.frame_h)
        self._live_px = None

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        # Qt highgui already reports original image pixels (including zoom).
        # Scaling by getWindowImageRect double-maps and the box misses the cursor.
        x, y = clamp_xy(x, y, self.frame_w, self.frame_h)
        x1, y1, x2, y2 = self.pixels()
        radius = handle_radius(self.frame_w, self.frame_h)

        if event == cv2.EVENT_LBUTTONDOWN:
            handle = hit_handle(x, y, x1, y1, x2, y2, radius)
            if handle:
                self._mode = "resize"
                self._handle = handle
                self._start_px = (x1, y1, x2, y2)
                self._live_px = (x1, y1, x2, y2)
            elif point_in_rect(x, y, x1, y1, x2, y2):
                self._mode = "move"
                self._origin = (x, y)
                self._start_px = (x1, y1, x2, y2)
                self._live_px = (x1, y1, x2, y2)
            else:
                self._mode = "draw"
                self._origin = (x, y)
                self._live_px = (x, y, x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self._mode:
            if self._mode == "draw":
                ox, oy = self._origin
                self._live_px = (ox, oy, x, y)
            elif self._mode == "move" and self._start_px:
                dx = x - self._origin[0]
                dy = y - self._origin[1]
                sx1, sy1, sx2, sy2 = self._start_px
                bw, bh = sx2 - sx1, sy2 - sy1
                nx1 = max(0, min(self.frame_w - bw, sx1 + dx))
                ny1 = max(0, min(self.frame_h - bh, sy1 + dy))
                self._live_px = (nx1, ny1, nx1 + bw, ny1 + bh)
            elif self._mode == "resize" and self._start_px and self._handle:
                nx1, ny1, nx2, ny2 = self._start_px
                edges = HANDLE_EDGES[self._handle]
                if "n" in edges:
                    ny1 = y
                if "s" in edges:
                    ny2 = y
                if "w" in edges:
                    nx1 = x
                if "e" in edges:
                    nx2 = x
                self._live_px = (nx1, ny1, nx2, ny2)
            return

        if event == cv2.EVENT_LBUTTONUP and self._mode:
            box = self._live_px or self.pixels()
            self._commit(*box)
            self._mode = None
            self._handle = None
            self._start_px = None
            if self.save_path is not None:
                persist_roi(self.save_path, self.roi)
                print(f"[roi] {format_roi(self.roi)}")


def draw_roi_handles(frame, roi_px: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = roi_px
    radius = max(4, handle_radius(frame.shape[1], frame.shape[0]) // 3)
    for hx, hy in handle_points(x1, y1, x2, y2).values():
        cv2.rectangle(
            frame,
            (hx - radius, hy - radius),
            (hx + radius, hy + radius),
            color,
            -1,
            cv2.LINE_AA,
        )

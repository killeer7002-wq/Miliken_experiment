"""Чтение видео: кадры + время из PTS."""

from __future__ import annotations

import av
from dataclasses import dataclass
from pathlib import Path
import functools
import logging
from fractions import Fraction
from collections.abc import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    n_frames: int
    duration_s: float
    fps_mean: float


def _channel(frame_bgr: np.ndarray, channel: str) -> np.ndarray:
    if frame_bgr.ndim == 2 or channel == "bgr":
        return frame_bgr
    if channel == "green":
        return np.ascontiguousarray(frame_bgr[:, :, 1])
    if channel == "red":
        return np.ascontiguousarray(frame_bgr[:, :, 2])
    if channel == "blue":
        return np.ascontiguousarray(frame_bgr[:, :, 0])
    if channel == "gray":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"неизвестный канал {channel!r}")


@functools.lru_cache(maxsize=8)
def _pts_cached(path: str) -> tuple[tuple[int, ...], Fraction]:
    with av.open(path) as cap:
        st = cap.streams.video[0]
        tb = st.time_base
        pts = sorted(p.pts for p in cap.demux(st) if p.pts is not None)
    return tuple(pts), tb

def read_pts(path) -> tuple[np.ndarray, Fraction]:
    pts, tb = _pts_cached(str(Path(path).resolve()))
    return np.asarray(pts), tb

def read_timestamps(path: str | Path) -> np.ndarray:
    """Время каждого кадра (с), отсчитанное от первого кадра, по PTS."""
    pts, tb = read_pts(path)
    return (pts - pts[0]) * float(tb)

def video_info(path: str | Path) -> VideoInfo:
    path = str(Path(path).resolve())
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"не удалось открыть видео {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    t = read_timestamps(path)
    dur = float(t[-1] + np.median(np.diff(t))) if len(t) > 1 else 0.0
    return VideoInfo(path, w, h, len(t), dur, (len(t) - 1) / t[-1] if len(t) > 1 and t[-1] > 0 else 0.0)


def frame_range(t: np.ndarray, t_start: float | None, t_end: float | None) -> tuple[int, int]:
    """Полуинтервал индексов кадров [i0, i1), попадающих в [t_start, t_end]."""
    i0 = 0 if t_start is None else int(np.searchsorted(t, t_start - 1e-9))
    i1 = len(t) if t_end is None else int(np.searchsorted(t, t_end + 1e-9))
    return i0, max(i0, i1)


def iter_frames(
    path: str | Path,
    channel: str = "green",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    crop: tuple[int, int, int, int] | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Генератор (frame_index, t_seconds_from_pts, image_uint8).

    crop: (y0, y1, x0, x1); step: шаг выборки.
    """
    path = str(Path(path).resolve())
    pts, tb = read_pts(path)
    n = len(pts)
    stop = n if stop is None else min(stop, n)
    if start >= stop:
        return
    t = (pts - pts[0]) * float(tb)
    start_pts = int(pts[start])

    with av.open(path) as cap:
        st = cap.streams.video[0]
        st.thread_type = "AUTO"
        if start > 0:
            cap.seek(start_pts, stream=st, backward=True)

        i = start
        for frame in cap.decode(st):
            if frame.pts is None or frame.pts < start_pts:
                continue
            if i >= stop:
                break
            if (i - start) % step == 0:
                img = _channel(frame.to_ndarray(format="bgr24"), channel)
                if crop is not None:
                    y0, y1, x0, x1 = crop
                    img = np.ascontiguousarray(img[y0:y1, x0:x1])
                yield i, float(t[i]), img
            i += 1


def read_frames_at(
    path: str | Path,
    indices: list[int] | np.ndarray,
    channel: str = "green",
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Прочитать кадры с заданными индексами (по возрастанию) в массив (N, H, W)."""
    want = sorted(set(int(i) for i in indices))
    if not want:
        return np.empty((0, 0, 0), np.uint8)
    out = []
    wset = set(want)
    for i, _t, img in iter_frames(path, channel, start=want[0], stop=want[-1] + 1, crop=crop):
        if i in wset:
            out.append(img)
    return np.stack(out)

if __name__ == "__main__":
    with av.open("./media/VIDEO2.mp4") as c:
        s = c.streams.video[0]
        print(s.codec_context.name)
        print(s.width, s.height)
        print(s.average_rate)
        print(s.time_base)
        print(s.frames)
        target_sec = 120
        c.seek(int(target_sec / s.time_base), stream=s)
        for frame in c.decode(s):
            if frame.time >= target_sec:
                img = frame.to_image()
                img.save(f"./media/Frame_{target_sec}_sec.jpg")

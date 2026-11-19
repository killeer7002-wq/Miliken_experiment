"""Фон: скользящая временная медиана (или нижний перцентиль) по кадрам."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import BackgroundConfig
from .io_video import iter_frames
from .roi import Roi

log = logging.getLogger(__name__)


@dataclass
class Background:
    node_t: np.ndarray          # (K,) время узлов
    nodes: np.ndarray           # (K, h, w) — фон в узлах
    crop: tuple[int, int, int, int]
    mask: np.ndarray            # bool (h, w) — маска в кропе

    def at(self, t: float) -> np.ndarray:
        """Фон в момент t линейная апроксимация между узлами."""
        k = int(np.searchsorted(self.node_t, t))
        if k <= 0:
            return self.nodes[0].astype(np.float32)
        if k >= len(self.node_t):
            return self.nodes[-1].astype(np.float32)
        t0, t1 = self.node_t[k - 1], self.node_t[k]
        a = float((t - t0) / (t1 - t0)) if t1 > t0 else 0.0
        return np.float32(1 - a) * self.nodes[k - 1].astype(np.float32) + np.float32(a) * self.nodes[k].astype(np.float32)

    def difference(self, img_crop: np.ndarray, t: float) -> np.ndarray:
        """Разность кадр − фон без обрезки"""
        b = self.at(t)
        r = img_crop.astype(np.float32) - b
        r[~self.mask] = 0
        return r

    def residual(self, img_crop: np.ndarray, t: float) -> np.ndarray:
        """Остаток max(кадр − фон, 0) внутри маски."""
        return np.maximum(self.difference(img_crop, t), 0)


def build_background(video: str, roi: Roi, cfg: BackgroundConfig, channel: str,
                     i0: int, i1: int, t_all: np.ndarray) -> Background:
    """
    video - путь до видео
    roi - интересующая область
    cfg - конфигурация обработки фона
    channel - исследуемый цветовой канал записи
    i0 - начальный кадр
    i1 - конечный кадр
    t_all - времена для каждого кадра
    """
    crop = roi.bbox
    samples: list[np.ndarray] = []
    st: list[float] = []
    for _i, t, img in iter_frames(video, channel, start=i0, stop=i1, step=cfg.sample_every, crop=crop):
        samples.append(img)
        st.append(t)
    if not samples:
        raise RuntimeError("нет кадров для фона")

    stack = np.stack(samples)
    st_arr = np.asarray(st)

    t_a, t_b = float(t_all[i0]), float(t_all[i1 - 1])
    n_nodes = max(2, int(np.ceil((t_b - t_a) / cfg.update_every_s)) + 1)
    node_t = np.linspace(t_a, t_b, n_nodes)

    half = cfg.window_s / 2
    nodes = np.empty((n_nodes,) + stack.shape[1:], np.float16)
    for k, tn in enumerate(node_t):
        # окно сдвигается внутрь у краёв, чтобы ширина оставалась постоянной
        lo, hi = tn - half, tn + half
        if lo < t_a:
            lo, hi = t_a, min(t_b, t_a + cfg.window_s)
        if hi > t_b:
            lo, hi = max(t_a, t_b - cfg.window_s), t_b
        sel = (st_arr >= lo) & (st_arr <= hi)

        if sel.sum() < 3:
            # Гарантия что хотя бы 3 элемента мы используем для пересчёта фона
            sel = np.argsort(np.abs(st_arr - tn))[:3]
        sub = stack[sel]
        kth = int(round(cfg.percentile / 100.0 * (len(sub) - 1)))
        sub.partition(kth, axis=0)
        nodes[k] = sub[kth]
    y0, y1, x0, x1 = crop
    log.info("фон: %d кадров-образцов, %d узлов, окно %.1f с, перцентиль %.0f",
             len(samples), n_nodes, cfg.window_s, cfg.percentile)
    m = roi.mask[y0:y1, x0:x1].copy()
    return Background(node_t, nodes, crop, m)

"""Автоматическая маска круглого поля зрения окуляра.

Алгоритм: медиана ~30 равномерно взятых кадров → сглаживание → порог Оцу →
наибольшая связная компонента → подгонка окружности (МНК Каса) по точкам
границы, НЕ лежащим на краю кадра (поле зрения на этом видео обрезано правым
краем кадра) → диск радиуса R − erode_px, пересечённый с кадром.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import RoiConfig
from .io_video import read_frames_at

log = logging.getLogger(__name__)


@dataclass
class Roi:
    """Маска поля зрения в координатах полного кадра"""

    mask: np.ndarray
    cx: float
    cy: float
    radius: float
    erode_px: int
    bbox: tuple[int, int, int, int]  # y0, y1, x0, x1 — охватывающий прямоугольник маски
    median_frame: np.ndarray    # медианный кадр

    @property
    def dist(self) -> np.ndarray:
        """Расстояние (px) от каждого пикселя до края маски (0 вне маски)."""
        if not hasattr(self, "_dist"):
            self._dist = cv2.distanceTransform(self.mask.astype(np.uint8), cv2.DIST_L2, 5)
        return self._dist

    def dist_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h, w = self.mask.shape
        xi = np.clip(np.round(x).astype(int), 0, w - 1)
        yi = np.clip(np.round(y).astype(int), 0, h - 1)
        return self.dist[yi, xi]

    def crop_mask(self) -> np.ndarray:
        y0, y1, x0, x1 = self.bbox
        return self.mask[y0:y1, x0:x1]


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Алгебраическая подгонка окружности (Kasa)"""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = -d / 2, -e / 2
    r = np.sqrt(max(cx**2 + cy**2 - f, 0.0))
    return float(cx), float(cy), float(r)


def disk_mask(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def roi_from_image(med: np.ndarray, cfg: RoiConfig) -> Roi:
    """Построить маску по медианному кадру"""
    h, w = med.shape
    if not cfg.auto and cfg.circle is not None:
        cx, cy, r = map(float, cfg.circle)
    else:
        blur = cv2.GaussianBlur(med, (0, 0), 8)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
        if n < 2:
            raise RuntimeError("не удалось найти светлое поле зрения")
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = (lab == k).astype(np.uint8)
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
        m = 3
        keep = (pts[:, 0] > m) & (pts[:, 0] < w - 1 - m) & (pts[:, 1] > m) & (pts[:, 1] < h - 1 - m)
        pts = pts[keep]
        cx, cy, r = fit_circle(pts[:, 0], pts[:, 1])
        # Уточнение окружности
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        good = res < max(3.0, 2.5 * np.median(res))
        cx, cy, r = fit_circle(pts[good, 0], pts[good, 1])
    mask = disk_mask((h, w), cx, cy, r - cfg.erode_px)
    ys, xs = np.nonzero(mask)
    bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    log.info("поле зрения: центр (%.1f, %.1f), R = %.1f px (диаметр %.0f), маска %d px",
             cx, cy, r, 2 * r, mask.sum())
    return Roi(mask, cx, cy, r, cfg.erode_px, bbox, med)


def compute_roi(video: str, n_total: int, cfg: RoiConfig, channel: str = "green",
                i0: int = 0, i1: int | None = None) -> Roi:
    """Медиана cfg.n_frames равномерно взятых кадров для roi_from_image"""
    i1 = n_total if i1 is None else i1
    idx = np.unique(np.linspace(i0, i1 - 1, min(cfg.n_frames, i1 - i0)).astype(int))
    frames = read_frames_at(video, idx, channel)
    med = np.median(frames, axis=0).astype(np.uint8)
    return roi_from_image(med, cfg)

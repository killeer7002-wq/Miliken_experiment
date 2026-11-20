"""Детекция: точка и кольцо → ровно одна детекция, ошибка центра < 0.3 px."""

from __future__ import annotations

import numpy as np

from src.config import DetectConfig
from src.detect import detect_frame
from src.synth import _render_ring, _render_spot

H, W = 160, 160


def _frame(kind: str, r: float, x: float, y: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    drop = np.zeros((H, W), np.float32)
    if kind == "point":
        _render_spot(drop, x, y, r, 70.0)
    else:
        _render_ring(drop, x, y, r, 1.0, 40.0, 0.0, 0.25)
    # тёмная линия шкалы под 12°: у кольца пересекает обод, у точки проходит рядом
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.deg2rad(12)
    off = 0.6 * r + 1 if kind == "ring" else 3 * r + 3
    d = -(xx - x) * np.sin(th) + (yy - (y + off)) * np.cos(th)
    lines = 1 - 0.45 * np.exp(-(d**2) / (2 * 0.9**2))
    bg = 110.0
    frame = (bg + drop) * lines + rng.normal(0, 2.0, (H, W))
    D = frame - bg * lines              # идеальный фон (фон — отдельный модуль)
    mask = np.ones((H, W), bool)
    mask[:2] = mask[-2:] = False
    mask[:, :2] = mask[:, -2:] = False
    return D.astype(np.float32), mask


def _check(kind: str, r: float, seed: int) -> tuple[int, float]:
    rng = np.random.default_rng(100 + seed)
    x, y = 80 + rng.uniform(-0.5, 0.5), 80 + rng.uniform(-0.5, 0.5)
    D, mask = _frame(kind, r, x, y, seed)
    df = detect_frame(D, mask, DetectConfig())
    if len(df) == 0:
        return 0, np.inf
    j = int(np.argmin(np.hypot(df.x - x, df.y - y)))
    return len(df), float(np.hypot(df.x.iloc[j] - x, df.y.iloc[j] - y))


def test_point_single_detection():
    for r in (1.2, 1.6, 2.2):
        for seed in range(3):
            n, err = _check("point", r, seed)
            assert n == 1, f"точка r={r}: {n} детекций"
            assert err < 0.3, f"точка r={r}: ошибка {err:.2f} px"


def test_ring_single_detection():
    for r in (5, 8, 12, 16, 20, 25):
        for seed in range(3):
            n, err = _check("ring", r, seed)
            assert n == 1, f"кольцо r={r}: {n} детекций"
            assert err < 0.3, f"кольцо r={r}: ошибка {err:.2f} px"


def test_ring_radius_estimate():
    for r in (6, 12, 20):
        rng = np.random.default_rng(r)
        x, y = 80 + rng.uniform(-0.5, 0.5), 80 + rng.uniform(-0.5, 0.5)
        D, mask = _frame("ring", r, x, y, r)
        df = detect_frame(D, mask, DetectConfig())
        assert len(df) == 1 and df.kind.iloc[0] == "ring"
        assert abs(df.r.iloc[0] - r) < 0.15 * r + 1.0, f"радиус {df.r.iloc[0]:.1f} вместо {r}"

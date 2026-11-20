"""Поиск точек излома: PELT для сдвига среднего (взвешенная L2-стоимость).

Своя реализация вместо ``ruptures``: нужен вес у каждой точки (разные Δt
между реальными детекциями) и многомерный сигнал; к тому же так проект не
зависит от пакета, которого может не оказаться в окружении. Алгоритм — PELT
(Killick, Fearnhead, Eckley 2012): точное минимизирование
Σ cost(сегмент) + β·(число разрезов) с отсечением заведомо проигрышных
кандидатов. Стоимость сегмента — взвешенная сумма квадратов отклонений от
взвешенного среднего (вычисляется за O(1) по кумулятивным суммам).
"""

from __future__ import annotations

import numpy as np


def pelt_mean(y: np.ndarray, penalty: float, min_size: int = 2,
              weights: np.ndarray | None = None) -> list[int]:
    """Разрезы сигнала ``y`` (n,) или (n, d) по сдвигу среднего.

    Возвращает отсортированный список индексов начала новых сегментов
    (без 0 и n). ``min_size`` — минимальная длина сегмента в точках.
    """
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    if n < 2 * min_size:
        return []
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cwy = np.vstack([np.zeros(y.shape[1]), np.cumsum(w[:, None] * y, axis=0)])
    cwy2 = np.concatenate([[0.0], np.cumsum(w * (y**2).sum(axis=1))])

    def cost(a: np.ndarray, b: int) -> np.ndarray:
        sw = cw[b] - cw[a]
        sy = cwy[b] - cwy[a]
        return (cwy2[b] - cwy2[a]) - (sy**2).sum(axis=1) / np.maximum(sw, 1e-12)

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, int)
    cand = np.array([0])
    for t in range(min_size, n + 1):
        ok = cand[t - cand >= min_size]
        if ok.size == 0:
            continue
        vals = F[ok] + cost(ok, t) + penalty
        k = int(np.argmin(vals))
        F[t] = vals[k]
        last[t] = ok[k]
        # отсечение PELT: кандидаты, которые уже не могут выиграть
        keep = F[ok] + cost(ok, t) <= F[t]
        cand = np.concatenate([cand[t - cand < min_size], ok[keep], [t]])
    bps = []
    t = n
    while t > 0:
        t = last[t]
        if t > 0:
            bps.append(t)
    return sorted(bps)


def robust_noise_var(y: np.ndarray) -> float:
    """Оценка дисперсии шума по первым разностям (устойчиво к сдвигам среднего)."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    d = np.diff(y, axis=0)
    if len(d) == 0:
        return 1.0
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0) * 1.4826
    return float(np.mean(mad**2) / 2) + 1e-12

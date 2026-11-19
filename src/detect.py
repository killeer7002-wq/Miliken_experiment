"""Детекция капель: многомасштабный LoG + проверка колец + субпиксельные центры.

Шаги для одного кадра (на остатке после вычитания фона):

1. Scale-normalized LoG: для всех σ из конфига
2. Порог для каждого масштаба отдельно: k_mad * 1.4826 · MAD(L_σ) по
   пикселям маски
3. Кандидаты — локальные максимумы по пространству и масштабу (3×3×3).
4. Крупные кандидаты (r = σ√2 ≥ ring_min_r, порог k_mad_ring ниже, чем
   у точек) и кандидаты кольцевого согласованного фильтра (ring_radii:
   тонкие кольца большого радиуса почти не дают пика LoG в центре)
   проверяются на "круглый компактный объект" check_rings:
   радиальный профиль, край по полуспаду, покрытие по углам, согласованность
   точек края с окружностью и яркость вдоль подогнанной окружности.
   Расфокусированные капли на видео чаще выглядят как светлый диск с
   ярким ободом (иногда с точкой в центре), чем как тонкое кольцо. Это нужно потому, что у
   расфокусированной капли кольцо тонкое, и нормированный LoG на мелких σ даёт
   на самой окружности *более сильные* пики, чем в центре кольца на крупном σ.
   Правило "оставить самую сильную детекцию" из исходной постановки выбрало бы
   кусок окружности, а не центр; поэтому принятое кольцо *поглощает* все мелкие
   детекции в радиусе ρ*+2 px (точки на окружности и точку в центре).
   Центр кольца уточняется взвешенной подгонкой окружности по гребню кольца
   (по максимуму вдоль каждого луча); если в центре есть яркая точка, центр
   берётся по ней (она локализуется точнее).
5. Крупные кандидаты, не прошедшие проверку: если внутри них есть мелкие
   детекции — это артефакт пары близких капель, отбрасываем; если нет и
   r ≤ blob_max_r — это размытая капля без кольца, сохраняем как 'point';
   крупнее — туманное пятно, отбрасываем.
6. Подавление немаксимумов: детекции одного типа ближе
   merge_factor·max(r1, r2) сливаются, остаётся с большим откликом
   (для точек отклик: score, для колец: яркость·покрытие).
7. Центры точек: взвешенный центроид остатка (за вычетом фона окна) в
   круге радиуса clip(1.6*r, 3, 6) px, три итерации с пересчётом окна.

Итог: DataFrame [frame, t, x, y, r, intensity, kind, score, coverage].
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from .background import Background
from .config import DetectConfig
from .io_video import iter_frames

log = logging.getLogger(__name__)

COLUMNS = ["frame", "t", "x", "y", "r", "intensity", "kind", "score", "coverage"]


@dataclass
class RingParams:
    n_theta: int = 32
    dr: float = 0.5
    min_rho: float = 3.0
    max_rho: float = 30.0
    min_coverage: float = 0.55
    dip_ratio: float = 0.75        # внутренний минимум < dip_ratio*пик
    outer_ratio: float = 0.75      # снаружи (ρ*+3) < outer_ratio*пик
    absorb_margin: float = 2.0     # px сверх ρ*, в которых точки поглощаются кольцом
    center_dot_frac: float = 0.35  # точка ближе center_dot_frac·ρ* к центру = "точка в центре"
    blob_max_r: float = 8.0


RING = RingParams()


def log_stack(img: np.ndarray, sigmas: list[float], keep_blur: bool = False):
    """Нормированный LoG для всех σ: массив (S, H, W) float32.

    С "keep_blur=True" возвращает ещё и сглаженные изображения G_σ*img
    """
    img = np.asarray(img, np.float32)
    out = np.empty((len(sigmas),) + img.shape, np.float32)
    blurs = []
    for k, s in enumerate(sigmas):
        g = cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=1)
        out[k] = -(s * s) * lap
        if keep_blur:
            blurs.append(g)
    return (out, blurs) if keep_blur else out


def edge_ratio(blurs: list[np.ndarray], k: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Отношение главных кривизн гессиана G_σ*img в точке.
    """
    out = np.ones(len(x))
    for kk in np.unique(k):
        sel = k == kk
        g = blurs[kk]
        h, w = g.shape
        yy = np.clip(y[sel], 1, h - 2)
        xx = np.clip(x[sel], 1, w - 2)

        dxx = g[yy, xx + 1] - 2 * g[yy, xx] + g[yy, xx - 1]
        dyy = g[yy + 1, xx] - 2 * g[yy, xx] + g[yy - 1, xx]
        dxy = 0.25 * (g[yy + 1, xx + 1] - g[yy + 1, xx - 1] - g[yy - 1, xx + 1] + g[yy - 1, xx - 1])
        
        tr = dxx + dyy
        disc = np.sqrt(np.maximum((dxx - dyy) ** 2 / 4 + dxy**2, 0))
        l1 = np.abs(tr / 2 - disc)
        l2 = np.abs(tr / 2 + disc)
        
        big, small = np.maximum(l1, l2), np.minimum(l1, l2)

        saddle = (tr / 2 - disc) * (tr / 2 + disc) < 0
        
        r = big / np.maximum(small, 1e-6)
        r[saddle] = np.inf
        out[sel] = r
    return out


def _annulus_kernel(rho: float, width: float = 1.0) -> np.ndarray:
    """Ядро "кольцо минус окрестность" с нулевой суммой.

    Положительная часть это гауссов профиль на радиусе rho (с суммой +1);
    отрицательная это кольца [rho−2.5w … rho−1.5w] и [rho+1.5w … rho+3w] (сумма −1). 
    Отклик = средняя яркость обода минус яркость вокруг.
    """
    h = int(np.ceil(rho + 3.5 * width)) + 1
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    d = np.hypot(xx, yy)
    pos = np.exp(-((d - rho) ** 2) / (2 * (0.6 * width) ** 2))
    pos /= pos.sum()
    neg = (((d >= max(1.5, rho - 2.5 * width)) & (d <= rho - 1.5 * width)) |
           ((d >= rho + 1.5 * width) & (d <= rho + 3.0 * width))).astype(float)
    neg /= max(neg.sum(), 1)
    return (pos - neg).astype(np.float32)


_ANN_CACHE: dict[tuple[float, ...], list[np.ndarray]] = {}


def annulus_candidates(R: np.ndarray, mask: np.ndarray, radii: list[float], k_mad: float,
                       factor: int = 2, width: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Кандидаты в кольца: максимумы согласованного кольцевого фильтра по (ρ, x, y).

    Считается на изображении, уменьшенном в factor раз.
    Возвращает (x, y, ρ, score), score — отклик в единицах робастного шума этого радиуса.
    """
    if not radii:
        z = np.zeros(0)
        return z, z, z, z

    small = cv2.resize(R, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
    msk = cv2.resize(mask.astype(np.uint8), None, fx=1 / factor, fy=1 / factor,
                     interpolation=cv2.INTER_NEAREST).astype(bool)

    key = tuple(radii) + (factor,)
    if key not in _ANN_CACHE:
        _ANN_CACHE[key] = [_annulus_kernel(r / factor, width=width) for r in radii]
    resp = np.stack([cv2.filter2D(small, cv2.CV_32F, kk, borderType=cv2.BORDER_CONSTANT)
                     for kk in _ANN_CACHE[key]])

    S = len(radii)
    sub = msk[::2, ::2]
    zsc = np.empty_like(resp)
    for k in range(S):
        sig = _robust_sigma(resp[k][::2, ::2][sub])
        zsc[k] = resp[k] / sig

    kernel = np.ones((3, 3), np.uint8)
    dil = np.stack([cv2.dilate(zsc[k], kernel) for k in range(S)])
    mx = dil.copy()
    mx[1:] = np.maximum(mx[1:], dil[:-1])
    mx[:-1] = np.maximum(mx[:-1], dil[1:])

    peak = (zsc >= mx) & (zsc > k_mad) & msk[None]
    k, y, x = np.nonzero(peak)
    return (x * factor + (factor - 1) / 2).astype(float), (y * factor + (factor - 1) / 2).astype(float), \
        np.asarray(radii, float)[k], zsc[k, y, x]


def _robust_sigma(v: np.ndarray) -> float:
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med))) + 1e-6


def _candidates(L: np.ndarray, mask: np.ndarray, cfg: DetectConfig,
                sigmas: list[float] | None = None) -> tuple[np.ndarray, ...]:
    """Локальные максимумы 3×3×3 выше порога. Возвращает (k, y, x, value, thr_k).

    sigmas это масштабы слоёв L в пикселях полного кадра.
    """
    sigmas = list(cfg.sigmas) if sigmas is None else sigmas
    S = L.shape[0]
    sub = mask[::3, ::3]
    thr = np.empty(S, np.float32)
    for k in range(S):
        if cfg.threshold == "auto":
            kk = cfg.k_mad_ring if sigmas[k] * np.sqrt(2) >= cfg.ring_min_r else cfg.k_mad
            thr[k] = kk * _robust_sigma(L[k][::3, ::3][sub])
        else:
            thr[k] = float(cfg.threshold)
    kernel = np.ones((3, 3), np.uint8)
    dil = np.stack([cv2.dilate(L[k], kernel) for k in range(S)])

    mx = dil.copy()
    mx[1:] = np.maximum(mx[1:], dil[:-1])
    mx[:-1] = np.maximum(mx[:-1], dil[1:])
    peak = (L >= mx) & (L > thr[:, None, None]) & mask[None]
    k, y, x = np.nonzero(peak)
    return k, y, x, L[k, y, x], thr


def _polar_samples(R: np.ndarray, x: np.ndarray, y: np.ndarray, rho: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
    """Яркости R на полярной сетке вокруг каждого кандидата (x, y), выход: (N, n_rho, n_theta)
    (i, j, k) : яркость относительно i-ого кандидата, на rho[j]-ом радиусе и theta[m]-ом угле
    """
    xs = x[:, None, None] + rho[None, :, None] * np.cos(theta)[None, None, :]
    ys = y[:, None, None] + rho[None, :, None] * np.sin(theta)[None, None, :]
    return ndimage.map_coordinates(R, [ys, xs], order=1, mode="constant", cval=0.0)


def check_rings(R: np.ndarray, x: np.ndarray, y: np.ndarray, sig: np.ndarray,
                min_intensity: float, p: RingParams = RING,
                sigma_pix: float = 0.0, min_snr: float = 0.0) -> dict[str, np.ndarray]:
    """Проверка крупных кандидатов: круглый компактный объект (кольцо или диск).

    Для каждого кандидата строится полярная выборка остатка. pmax - максимум
    среднего радиального профиля; край объекта ρe — наибольший радиус, где
    профиль ≥ 0.5·pmax. Объект принимается, если
    * снаружи (ρe + 3 px) профиль падает ниже outer_ratio * pmax;
    * pmax >= min_snr * σ_pix (σ_pix — робастный шум пикселя разности кадра);
    * покрытие >= min_coverage: доля лучей, на которых в полосе
      [0.4ρe, ρe] есть яркость >= max(0.4·pmax, 2.5·σ_pix).
    * край найден на >= min_coverage лучей, и точки края ложатся на
      окружность: инлайеров (|d − R| < max(1.5, 0.15R)) ≥ min_coverage лучей,
      центр подгонки не дальше 0.6ρe от кандидата, 0.5ρe < R < 1.6ρe;
    * яркость вдоль самой подогнанной окружности (на 0.7 px внутри края)
      >= max(0.35·pmax, 2.5·σ_pix) на ≥ min_coverage углов 
    Центр и радиус - это подгонка окружности.

    Возвращает словарь массивов длины N: ok, cx, cy, rho, intensity, coverage.
    """
    n = len(x)
    res = {k: np.zeros(n) for k in ("cx", "cy", "rho", "intensity", "coverage")}
    res["ok"] = np.zeros(n, bool)
    if n == 0:
        return res

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sig = np.asarray(sig, float)
    max_rho = min(p.max_rho, float(np.max(sig)) * 2.5 + 6)
    rho = np.arange(0.0, max_rho + 1e-9, p.dr)

    nr, nt = len(rho), p.n_theta
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)

    P = _polar_samples(R, x, y, rho, theta)
    prof = P.mean(axis=2)

    ridx = np.arange(nr)[None, :]
    lim = np.minimum(nr - 1, np.ceil((sig * 2.5 + 4) / p.dr).astype(int))[:, None]
    inlim = ridx <= lim

    base = np.where(ridx >= lim // 2, prof, np.inf).min(axis=1)
    base = np.where(np.isfinite(base), base, 0.0)

    q = np.where(inlim, prof - base[:, None], -np.inf)
    pmax = q.max(axis=1)

    above = (q >= 0.5 * pmax[:, None]) & inlim
    je = nr - 1 - np.argmax(above[:, ::-1], axis=1)
    rho_e = rho[je]

    ok = (pmax + base >= min_intensity) & (pmax > 0) & (pmax >= min_snr * sigma_pix) & (je < lim[:, 0] - 1)
    jout = np.minimum(nr - 1, je + int(3 / p.dr))
    ok &= prof[np.arange(n), jout] - base <= p.outer_ratio * pmax

    # покрытие по углам в полосе [0.4ρe, ρe]
    j0 = (0.4 * rho_e / p.dr).astype(int)
    band = (ridx >= j0[:, None]) & (ridx <= je[:, None])
    bmax = np.where(band[:, :, None], P, -np.inf).max(axis=1) - base[:, None]

    thr_ray = np.maximum(0.4 * pmax, 2.5 * sigma_pix)[:, None]
    cov = (bmax >= thr_ray).mean(axis=1)
    ok &= cov >= p.min_coverage

    # край на каждом луче: последний индекс >= полуспада в пределах ρe+3
    ray = P - base[:, None, None]
    okr = (ray >= 0.5 * pmax[:, None, None]) & (np.arange(nr)[None, :, None] <= jout[:, None, None])
    has = okr.any(axis=1)                                        # (N, nt)
    last = nr - 1 - np.argmax(okr[:, ::-1, :], axis=1)           # (N, nt)
    li = np.clip(last, 0, nr - 2)
    v0 = np.take_along_axis(ray, li[:, None, :], axis=1)[:, 0, :]
    v1 = np.take_along_axis(ray, li[:, None, :] + 1, axis=1)[:, 0, :]

    frac = np.clip((v0 - 0.5 * pmax[:, None]) / np.maximum(v0 - v1, 1e-6), 0, 1)
    re = rho[li] + frac * p.dr
    sel = has & (re > 0.3 * rho_e[:, None])
    ok &= sel.sum(axis=1) >= p.min_coverage * nt

    ex = x[:, None] + re * ct[None]
    ey = y[:, None] + re * st[None]

    def fit(w: np.ndarray):
        """Подгонка окружности методом Каса"""
        A = np.stack([ex, ey, np.ones_like(ex)], axis=2)
        bb = -(ex**2 + ey**2)
        AtA = np.einsum("ntk,ntl,nt->nkl", A, A, w) + 1e-9 * np.eye(3)[None]
        Atb = np.einsum("ntk,nt,nt->nk", A, bb, w)
        good = np.linalg.det(AtA) > 1e-6
        sol = np.zeros((len(w), 3))
        if good.any():
            sol[good] = np.linalg.solve(AtA[good], Atb[good][..., None])[..., 0]
        fx, fy = -sol[:, 0] / 2, -sol[:, 1] / 2
        fr = np.sqrt(np.maximum(fx**2 + fy**2 - sol[:, 2], 0))
        return fx, fy, fr, good

    w = sel.astype(float)
    fx, fy, fr, good = fit(w)

    d = np.abs(np.hypot(ex - fx[:, None], ey - fy[:, None]) - fr[:, None])
    inl = sel & (d < np.maximum(1.5, 0.15 * fr)[:, None])
    ok &= good & (inl.sum(axis=1) >= p.min_coverage * nt)

    fx, fy, fr, good = fit(inl.astype(float))
    ok &= good & np.isfinite(fr) & (np.hypot(fx - x, fy - y) < 0.6 * rho_e) \
        & (fr > 0.5 * rho_e) & (fr < 1.6 * rho_e)

    # яркость вдоль подогнанной окружности (на 0.7 px внутри края) по всему кругу
    rr_on = np.maximum(fr - 0.7, 0.5)
    on = ndimage.map_coordinates(R, [(fy[:, None] + rr_on[:, None] * st[None]).ravel(),
                                     (fx[:, None] + rr_on[:, None] * ct[None]).ravel()],
                                 order=1).reshape(n, nt) - base[:, None]
    ok &= (on >= np.maximum(0.35 * pmax, 2.5 * sigma_pix)[:, None]).mean(axis=1) >= p.min_coverage

    res["ok"] = ok
    res["cx"], res["cy"], res["rho"] = np.where(ok, fx, x), np.where(ok, fy, y), np.where(ok, fr, rho_e)
    res["intensity"] = pmax + base
    res["coverage"] = cov
    return res


def _centroid_w(R: np.ndarray, x: np.ndarray, y: np.ndarray, w: int, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    off = np.arange(-w, w + 1)
    DY, DX = np.meshgrid(off, off, indexing="ij")
    disk = (DX**2 + DY**2) <= w * w
    h, wd = R.shape
    cx, cy = x.astype(float), y.astype(float)
    for _ in range(n_iter):
        # подгонца центроида с помощью центра масс
        xi = np.clip(np.round(cx).astype(int), w, wd - w - 1)
        yi = np.clip(np.round(cy).astype(int), w, h - w - 1)
        patch = R[yi[:, None, None] + DY[None], xi[:, None, None] + DX[None]]

        # фон окна - медиана его границы (кольцо радиуса w)
        edge = disk & ((DX**2 + DY**2) > (w - 1) ** 2)
        base = np.median(patch[:, edge], axis=1)

        wgt = np.clip(patch - base[:, None, None], 0, None) * disk[None]
        s = wgt.sum(axis=(1, 2)) + 1e-9
        ncx = xi + (wgt * DX[None]).sum(axis=(1, 2)) / s
        ncy = yi + (wgt * DY[None]).sum(axis=(1, 2)) / s

        far = np.hypot(ncx - x, ncy - y) > 2.0
        ncx[far], ncy[far] = cx[far], cy[far]
        cx, cy = ncx, ncy
    return cx, cy


def centroid(R: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray | None = None,
             n_iter: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Взвешенный центроид остатка в круге радиуса w = clip(ceil(1.6·r), 3, 6) px.

    Фон окна вычитается (медиана границы окна), окно пересчитывается вокруг
    нового центра n_iter раз.
    """
    cx, cy = x.astype(float).copy(), y.astype(float).copy()
    if len(x) == 0:
        return cx, cy
    ws = np.full(len(x), 3) if r is None else np.clip(np.ceil(1.6 * np.asarray(r)), 3, 6).astype(int)
    for w in np.unique(ws):
        m = ws == w
        cx[m], cy[m] = _centroid_w(R, x[m], y[m], int(w), n_iter)
    return cx, cy


def local_contrast(R: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Яркость в центре минус медиана кольца радиусов [r+2, r+4] вокруг точки."""
    if len(x) == 0:
        return np.zeros(0)
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rad = (r[:, None, None] + np.array([2.0, 3.0, 4.0])[None, :, None])
    xs = x[:, None, None] + rad * np.cos(theta)[None, None, :]
    ys = y[:, None, None] + rad * np.sin(theta)[None, None, :]
    ring = ndimage.map_coordinates(R, [ys.ravel(), xs.ravel()], order=1).reshape(xs.shape)
    center = ndimage.map_coordinates(R, [y, x], order=1)
    return center - np.median(ring.reshape(len(x), -1), axis=1)


def _nms(x: np.ndarray, y: np.ndarray, r: np.ndarray, strength: np.ndarray, factor: float) -> np.ndarray:
    """Жадное подавление: детекции ближе factor·max(r1, r2) — остаётся сильнейшая.

    Возвращает индексы оставленных детекций. Цикл идёт только по детекциям,
    у которых есть конфликтующие соседи.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, int)
    from scipy.spatial import cKDTree

    pts = np.column_stack([x, y])
    pairs = cKDTree(pts).query_pairs(factor * float(np.max(r)), output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    conf = d < factor * np.maximum(r[i], r[j])
    i, j = i[conf], j[conf]
    if len(i) == 0:
        return np.arange(n)
    nbr: dict[int, list[int]] = {}
    for a_, b_ in zip(i.tolist(), j.tolist()):
        nbr.setdefault(a_, []).append(b_)
        nbr.setdefault(b_, []).append(a_)
    removed = np.zeros(n, bool)
    involved = np.fromiter(nbr.keys(), int)
    for k in involved[np.argsort(-strength[involved])]:
        if removed[k]:
            continue
        for m in nbr[k]:
            removed[m] = True
    return np.nonzero(~removed)[0]


def detect_frame(D: np.ndarray, mask: np.ndarray, cfg: DetectConfig,
                 dist: np.ndarray | None = None) -> pd.DataFrame:
    """Детекции на одном кадре по разности D = кадр - фон (координаты в системе D).

    LoG и порог считаются по необрезанной разности D, центроиды, 
    профили колец и яркости по остатку max(D, 0).
    """
    sigmas = list(cfg.sigmas)
    D = np.asarray(D, np.float32)
    R = np.maximum(D, 0)
    if dist is None:
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    inner = dist > cfg.border_px
    sigma_pix = _robust_sigma(D[::5, ::5][inner[::5, ::5]])

    fine = [s_ for s_ in sigmas if s_ < cfg.coarse_sigma]
    coarse = [s_ for s_ in sigmas if s_ >= cfg.coarse_sigma]
    k_list, y_list, x_list, v_list, thr_all = [], [], [], [], {}
    if fine:
        guard = coarse[:1]
        sg = fine + guard
        Lf, blurs = log_stack(D, sg, keep_blur=True)
        kf, yf, xf, vf, thrf = _candidates(Lf, inner, cfg, sg)
        keep = kf < len(fine)
        kf, yf, xf, vf = kf[keep], yf[keep], xf[keep], vf[keep]
        # отсев вытянутых пиков (куски ободов колец, линии) — для мелких масштабов
        iso = edge_ratio(blurs, kf, yf, xf) <= cfg.max_edge_ratio
        kf, yf, xf, vf = kf[iso], yf[iso], xf[iso], vf[iso]
        k_list.append(kf)
        y_list.append(yf.astype(float))
        x_list.append(xf.astype(float))
        v_list.append(vf / thrf[kf])
    if coarse:
        guard = fine[-1:]
        sg = guard + coarse
        Ds = cv2.resize(D, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ins = cv2.resize(inner.astype(np.uint8), None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        Lc = log_stack(Ds, [s_ / 2 for s_ in sg])
        kc, yc, xc, vc, thrc = _candidates(Lc, ins, cfg, sg)
        keep = kc >= len(guard)
        kc, yc, xc, vc = kc[keep], yc[keep], xc[keep], vc[keep]
        k_list.append(kc - len(guard) + len(fine))
        y_list.append(yc * 2.0 + 0.5)
        x_list.append(xc * 2.0 + 0.5)
        v_list.append(vc / thrc[kc])
    k = np.concatenate(k_list).astype(int)
    y = np.concatenate(y_list)
    x = np.concatenate(x_list)
    score = np.concatenate(v_list)
    sig = np.asarray(sigmas, float)[k]
    r = sig * np.sqrt(2)
    small = r < cfg.ring_min_r
    big = ~small

    # --- точки ---
    px, py, pr, ps = x[small].astype(float), y[small].astype(float), r[small], score[small]
    keep = _nms(px, py, pr, ps, cfg.merge_factor)
    px, py, pr, ps = px[keep], py[keep], pr[keep], ps[keep]
    pcx, pcy = centroid(R, px, py, pr)
    pint = ndimage.map_coordinates(R, [pcy, pcx], order=1)
    okp = (pint >= cfg.min_intensity) & (local_contrast(R, pcx, pcy, pr) >= cfg.min_contrast)
    pcx, pcy, pr, ps, pint = pcx[okp], pcy[okp], pr[okp], ps[okp], pint[okp]

    # --- крупные кандидаты: кольца / размытые пятна ---
    bx, by, bs, bsig = x[big].astype(float), y[big].astype(float), score[big], sig[big]
    # сначала грубое NMS, чтобы не проверять одно кольцо на 5 масштабах
    keepb = _nms(bx, by, bsig * np.sqrt(2) * 0.5, bs, 1.0)
    bx, by, bs, bsig = bx[keepb], by[keepb], bs[keepb], bsig[keepb]
    rc = check_rings(R, bx, by, bsig, cfg.min_intensity, RING, sigma_pix, cfg.ring_min_snr)
    ok = rc["ok"] & (rc["rho"] >= cfg.ring_min_r)
    # компактные объекты меньше ring_min_r это точки; добавляем их в список точек
    sm = rc["ok"] & ~ok
    if sm.any():
        pcx = np.concatenate([pcx, rc["cx"][sm]])
        pcy = np.concatenate([pcy, rc["cy"][sm]])
        pr = np.concatenate([pr, rc["rho"][sm]])
        ps = np.concatenate([ps, bs[sm]])
        pint = np.concatenate([pint, rc["intensity"][sm]])
        kp = _nms(pcx, pcy, pr, ps, cfg.merge_factor)
        pcx, pcy, pr, ps, pint = pcx[kp], pcy[kp], pr[kp], ps[kp], pint[kp]
    rx, ry, rr = rc["cx"][ok], rc["cy"][ok], rc["rho"][ok]
    rint, rcov, rs = rc["intensity"][ok], rc["coverage"][ok], bs[ok]
    # + кандидаты кольцевого фильтра (крупные тонкие кольца, у которых нет пика LoG)
    if cfg.ring_radii:
        ax_, ay_, ar_, as_ = annulus_candidates(D, inner, list(cfg.ring_radii), cfg.k_mad_annulus, width=cfg.width)
        ka = _nms(ax_, ay_, 0.5 * ar_, as_, 1.0)
        ax_, ay_, ar_, as_ = ax_[ka], ay_[ka], ar_[ka], as_[ka]
        ra = check_rings(R, ax_, ay_, ar_ / 1.14, cfg.min_intensity, RING,
                         sigma_pix, cfg.ring_min_snr)
        # принимаем, только если подогнанный радиус близок к радиусу фильтра
        # (иначе это частичное перекрытие кольца-фильтра с чужим объектом)
        oka = ra["ok"] & (ra["rho"] >= cfg.ring_min_r) & (np.abs(ra["rho"] - ar_) < 0.35 * ar_)
        rx = np.concatenate([rx, ra["cx"][oka]])
        ry = np.concatenate([ry, ra["cy"][oka]])
        rr = np.concatenate([rr, ra["rho"][oka]])
        rint = np.concatenate([rint, ra["intensity"][oka]])
        rcov = np.concatenate([rcov, ra["coverage"][oka]])
        rs = np.concatenate([rs, as_[oka]])
    # убираем совпадения колец между собой (концентрические/повторные) силa = яркость * покрытие
    keepr = _nms(rx, ry, rr, rint * rcov, cfg.merge_factor)
    rx, ry, rr, rint, rcov, rs = rx[keepr], ry[keepr], rr[keepr], rint[keepr], rcov[keepr], rs[keepr]

    # поглощение точек кольцами
    absorbed = np.zeros(len(pcx), bool)
    if len(rx) and len(pcx):
        d = np.hypot(pcx[:, None] - rx[None], pcy[:, None] - ry[None])  # (Np, Nr)
        inside = d < (rr[None] + RING.absorb_margin)
        absorbed = inside.any(axis=1)
        # точка в центре кольца уточняет центр
        central = d < RING.center_dot_frac * rr[None]
        for jr in range(len(rx)):
            ip = np.nonzero(central[:, jr])[0]
            if ip.size:
                b = ip[np.argmax(pint[ip])]
                rx[jr], ry[jr] = pcx[b], pcy[b]

    # неподтверждённые крупные: размытая капля без кольца или артефакт
    nb = ~rc["ok"]
    ux, uy, us, usig = bx[nb], by[nb], bs[nb], bsig[nb]
    ur = usig * np.sqrt(2)
    blob_keep = np.zeros(len(ux), bool)
    if len(ux):
        has_pts = np.zeros(len(ux), bool)
        if len(pcx):
            dd = np.hypot(pcx[:, None] - ux[None], pcy[:, None] - uy[None])
            has_pts = (dd < ur[None]).any(axis=0)
        has_ring = np.zeros(len(ux), bool)
        if len(rx):
            dr_ = np.hypot(rx[:, None] - ux[None], ry[:, None] - uy[None])
            has_ring = (dr_ < (rr[:, None] + ur[None])).any(axis=0)
        uint = ndimage.map_coordinates(R, [uy, ux], order=1)
        blob_keep = (~has_pts) & (~has_ring) & (ur <= RING.blob_max_r) & (uint >= cfg.min_intensity)
    ux, uy, ur, us = ux[blob_keep], uy[blob_keep], ur[blob_keep], us[blob_keep]
    if len(ux):
        ux, uy = centroid(R, ux, uy, ur)
        uint = ndimage.map_coordinates(R, [uy, ux], order=1)
    else:
        uint = np.zeros(0)

    m = ~absorbed
    xs = np.concatenate([pcx[m], ux, rx])
    ys = np.concatenate([pcy[m], uy, ry])
    out = pd.DataFrame({
        "x": xs, "y": ys,
        "r": np.concatenate([pr[m], ur, rr]),
        "intensity": np.concatenate([pint[m], uint, rint]),
        "kind": ["point"] * int(m.sum() + len(ux)) + ["ring"] * len(rx),
        "score": np.concatenate([ps[m], us, rs]),
        "coverage": np.concatenate([np.zeros(int(m.sum() + len(ux))), rcov]),
    })
    # последнее слияние "точка-пятно" внутри других точек
    if len(out):
        keep = _nms(out.x.values, out.y.values, out.r.values,
                    out.score.values + 1e3 * (out.kind.values == "ring"), cfg.merge_factor)
        out = out.iloc[np.sort(keep)].reset_index(drop=True)
    return out


_G: dict = {}


def _worker(args: tuple[int, int]) -> pd.DataFrame:
    a, b = args
    video, bg, cfg, channel = _G["video"], _G["bg"], _G["cfg"], _G["channel"]
    y0, _y1, x0, _x1 = bg.crop
    dist = cv2.distanceTransform(bg.mask.astype(np.uint8), cv2.DIST_L2, 3)
    parts = []
    for i, t, img in iter_frames(video, channel, start=a, stop=b, crop=bg.crop):
        df = detect_frame(bg.difference(img, t), bg.mask, cfg, dist)
        df["x"] += x0
        df["y"] += y0
        df.insert(0, "t", t)
        df.insert(0, "frame", i)
        parts.append(df)
        if _G.get("counter") is not None:
            with _G["counter"].get_lock():
                _G["counter"].value += 1
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)


def detect_video(video: str, bg: Background, cfg: DetectConfig, i0: int, i1: int,
                 progress: bool = True) -> pd.DataFrame:
    """Детекции на кадрах [i0, i1). Параллельно по кускам кадров (fork)."""
    n_workers = cfg.workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, (i1 - i0) // 20 or 1))
    _G.update(video=video, bg=bg, cfg=cfg, channel=cfg.channel)
    chunks = np.linspace(i0, i1, n_workers * 4 + 1).astype(int)
    tasks = [(int(a), int(b)) for a, b in zip(chunks[:-1], chunks[1:]) if b > a]
    bar = _progress(i1 - i0, "детекция", progress)
    if n_workers == 1:
        _G["counter"] = None
        parts = []
        for tsk in tasks:
            parts.append(_worker(tsk))
            bar.update(tsk[1] - tsk[0])
    else:
        ctx = mp.get_context("fork")
        counter = ctx.Value("i", 0)
        _G["counter"] = counter
        with ctx.Pool(n_workers) as pool:
            res = pool.map_async(_worker, tasks, chunksize=1)
            last = 0
            while not res.ready():
                res.wait(1.0)
                v = counter.value
                bar.update(v - last)
                last = v
            parts = res.get()
            bar.update(counter.value - last)
        _G["counter"] = None
    bar.close()
    df = pd.concat(parts, ignore_index=True).sort_values(["frame", "y", "x"]).reset_index(drop=True)
    df["kind"] = df["kind"].astype(str)
    return df[COLUMNS]


class _NoBar:
    def update(self, n: int) -> None:  # noqa: D401
        pass

    def close(self) -> None:
        pass


class _LogBar:
    def __init__(self, total: int, desc: str):
        self.total, self.desc, self.n, self.next = total, desc, 0, 0.1

    def update(self, n: int) -> None:
        self.n += n
        if self.total and self.n / self.total >= self.next:
            log.info("%s: %d/%d", self.desc, self.n, self.total)
            self.next += 0.1

    def close(self) -> None:
        pass


def _progress(total: int, desc: str, enabled: bool):
    if not enabled:
        return _NoBar()
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, unit="кадр")
    except Exception:
        return _LogBar(total, desc)


def save_detections(df: pd.DataFrame, path_stem: str) -> str:
    """Сохранить кэш: parquet, если есть pyarrow/fastparquet, иначе pickle."""
    try:
        df.to_parquet(path_stem + ".parquet", index=False)
        return path_stem + ".parquet"
    except Exception:
        df.to_pickle(path_stem + ".pkl")
        return path_stem + ".pkl"


def load_detections(path_stem: str) -> pd.DataFrame | None:
    for ext, fn in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        if os.path.exists(path_stem + ext):
            try:
                return fn(path_stem + ext)
            except Exception:
                continue
    return None

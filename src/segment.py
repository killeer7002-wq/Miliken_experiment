r"""Нарезка треков: по переключениям поля (известны) и по перезарядкам (детектор).

Модель внутри интервала постоянного поля и заряда:

    z(t_i) = z0 + v·t_i + B(t_i) + ε_i,

где B — броуновское блуждание с коэффициентом диффузии D (1D:
⟨ΔB²⟩ = 2Dτ), ε_i — ошибка локализации (независимая, дисперсия σ_loc²).

**Оценка D и σ_loc.** Остатки от линейной подгонки дают MSD при малых лагах
τ = k·dt (k = 1..msd_max_lag): MSD(τ) = 2Dτ + 2σ_loc². Наклон прямой
MSD(τ) даёт D, свободный член — σ_loc² (оба ограничены снизу нулём).

**Погрешность скорости.** Для МНК-наклона по N равномерно
распределённым точкам на отрезке длины T:

* вклад локализации (независимый шум): σ²_loc·12 / (N·T²)
  (так как Σ(t_i − t̄)² ≈ N·T²/12);
* вклад диффузии — не независимый шум, а коррелированный: x(t) = √(2D)·W(t).
  Наклон b = ∫(t−T/2)x(t)dt / (T³/12). Поскольку
  ∫∫ (t−T/2)(u−T/2)·min(t,u) dt du = ∫₀ᵀ [s(T−s)/2]² ds = T⁵/120,
  Var(b) = 2D·(T⁵/120)·(12/T³)² = 12D / (5T).

Итого  **σ_v² = 12·D / (5·T) + 12·σ_loc² / (N·T²)**.

Обычная стандартная ошибка МНК, которая считает остатки независимыми,
занижает погрешность при заметной диффузии (остатки броуновского движения
сильно коррелированы), поэтому используется формула выше.

**Поиск перезарядки.** Скорость заряженной капли в поле меняется скачком,
координата непрерывна. Поиск по самим координатам (а не по покадровым
приращениям: среднее приращение за окно зависит только от крайних точек,
и при заметном σ_loc такой детектор почти слеп): рекурсивная бисекция —
на куске сканируется момент излома непрерывной ломаной (базис 1, t,
(t−τ)+, минимум SSE), затем прямые по обе стороны дают
z = |v1 − v2| / √(σ²_v1 + σ²_v2) с σ_v по формуле выше (D и σ_loc —
робастной MSD приращений по всему интервалу, чтобы сам излом не завышал
шум; по умолчанию — по поперечной координате s, где перезарядки не
видно). Излом принимается при z ≥ k_sigma и обоих кусках ≥ min_piece_s.
В конце каждый излом перепроверяется по соседним кускам; слабейший
непрошедший удаляется, пока все не пройдут.

**Порог.** На синтетике (броуновское движение + белый шум) k_sigma = 4 с
учётом перебора положения излома даёт ~1% ложных изломов на кусок ≥ 2 с.
У реального видео шум хуже (сжатие, скачки центра при смене «точка ↔
кольцо»): тот же порог на поперечной координате s, где перезарядок нет,
даёт ~6%. Поэтому по умолчанию ``k_sigma: auto`` — порог берётся как
квантиль (1 − null_fa_rate) максимальной z-статистики на s того же видео
(:func:`calibrate_k_sigma`), но не ниже ``k_sigma_floor``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import Calibration
from .config import SegmentConfig
from .field_switch import FieldSwitches

log = logging.getLogger(__name__)


@dataclass
class LineFit:
    v: float
    v_err: float
    z0: float
    D: float
    sigma_loc: float
    n: int
    T: float


def msd_noise(t: np.ndarray, resid: np.ndarray, max_lag: int) -> tuple[float, float]:
    """D (px²/с) и σ_loc (px) по MSD остатков при лагах 1..max_lag (в точках)."""
    n = len(resid)
    if n < 4:
        return 0.0, 0.0
    lags, msd = [], []
    for k in range(1, min(max_lag, n - 2) + 1):
        dz = resid[k:] - resid[:-k]
        dt = t[k:] - t[:-k]
        lags.append(np.median(dt))
        msd.append(np.mean(dz**2))
    lags_a, msd_a = np.asarray(lags), np.asarray(msd)
    if len(lags_a) >= 2:
        slope, icpt = np.polyfit(lags_a, msd_a, 1)
    else:
        slope, icpt = msd_a[0] / lags_a[0], 0.0
    D = max(slope / 2, 0.0)
    s2 = max(icpt / 2, 0.0)
    if slope <= 0:  # шум локализации доминирует
        s2 = float(np.mean(msd_a)) / 2
    return float(D), float(np.sqrt(s2))


def velocity_error(D: float, sigma_loc: float, n: int, T: float) -> float:
    """σ_v = sqrt(12D/(5T) + 12σ_loc²/(N T²)); см. docstring модуля."""
    if n < 2 or T <= 0:
        return float("inf")
    return float(np.sqrt(12 * D / (5 * T) + 12 * sigma_loc**2 / (n * T * T)))


def fit_line(t: np.ndarray, z: np.ndarray, max_lag: int,
             D: float | None = None, sigma_loc: float | None = None) -> LineFit:
    n = len(t)
    if n < 2:
        return LineFit(np.nan, np.inf, float(z[0]) if n else np.nan, 0.0, 0.0, n, 0.0)
    tc = t - t.mean()
    v = float(np.sum(tc * (z - z.mean())) / max(np.sum(tc**2), 1e-12))
    z0 = float(z.mean() - v * t.mean())
    if D is None or sigma_loc is None:
        D_, s_ = msd_noise(t, z - (z0 + v * t), max_lag)
        D = D_ if D is None else D
        sigma_loc = s_ if sigma_loc is None else sigma_loc
    T = float(t[-1] - t[0])
    return LineFit(v, velocity_error(D, sigma_loc, n, T), z0, D, sigma_loc, n, T)


def robust_noise(t: np.ndarray, z: np.ndarray, max_lag_s: float = 1.0) -> tuple[float, float]:
    """D и σ_loc по робастной MSD приращений на лагах до ``max_lag_s``.

    Лаги — логарифмическая сетка от 1 кадра до min(max_lag_s, T/4): короткие
    лаги определяют σ_loc (свободный член), длинные — D (наклон). На одних
    коротких лагах при σ²_loc ≫ D·dt наклон почти не определён, и D часто
    выходит нулевым — тогда σ_v занижается и броуновское блуждание
    принимается за изломы. Дисперсия Δ_k z оценивается через MAD вокруг
    медианы (постоянный снос не мешает). MSD(τ) = 2Dτ + 2σ²: взвешенный МНК с
    весами n_k / MSD_k² (относительная ошибка оценки дисперсии ≈ √(2/n_k)).
    """
    n = len(z)
    if n < 8:
        return (msd_noise(t, z - np.polyval(np.polyfit(t, z, 1), t), 3) if n >= 4 else (0.0, 0.0))
    dt = float(np.median(np.diff(t)))
    kmax = int(max(2, min(max_lag_s / dt, (n - 1) / 4)))
    ks = np.unique(np.round(np.geomspace(1, kmax, 10)).astype(int))
    lags, var, cnt = [], [], []
    for k in ks:
        d = z[k:] - z[:-k]
        lags.append(float(np.median(t[k:] - t[:-k])))
        var.append((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2)
        cnt.append(len(d) / k)                     # примерно независимых приращений
    L, V, C = np.asarray(lags), np.asarray(var), np.asarray(cnt)
    w = C / np.maximum(V, 1e-9) ** 2
    A = np.column_stack([L, np.ones_like(L)]) * np.sqrt(w)[:, None]
    (slope, icpt), *_ = np.linalg.lstsq(A, V * np.sqrt(w), rcond=None)
    D = max(slope / 2, 0.0)
    s2 = max(icpt / 2, 0.0)
    if slope <= 0:
        s2 = float(np.average(V, weights=w)) / 2
    return float(D), float(np.sqrt(s2))


def _gls_scan(t: np.ndarray, z: np.ndarray, D: float, sl: float, lo: int, hi: int,
              min_pts: int, min_T: float) -> tuple[int, float]:
    """Лучший момент смены скорости на точках [lo, hi) по обобщённому МНК.

    Приращения y_i = z_{i+1} − z_i = v·Δt_i + (броуновский шаг) + ε_{i+1} − ε_i
    имеют трёхдиагональную ковариацию Σ: Σ_ii = 2D·Δt_i + 2σ², Σ_i,i±1 = −σ².
    Модель: v = v1 до излома и v2 после. Для каждого τ решаются нормальные
    уравнения GLS (через префиксные суммы Σ⁻¹-взвешенных величин), выбирается
    τ с максимальным приростом правдоподобия; возвращается (индекс точки начала
    второго куска, z = |v̂1 − v̂2| / √Var(v̂1 − v̂2)).
    """
    from scipy.linalg import solveh_banded

    tt, zz = t[lo:hi], z[lo:hi]
    n = len(tt) - 1                       # число приращений
    if n < 2 * min_pts:
        return -1, 0.0
    dt = np.diff(tt)
    y = np.diff(zz)
    s2 = max(sl * sl, 1e-6)
    diag = 2 * D * dt + 2 * s2
    ab = np.zeros((2, n))
    ab[0, 1:] = -s2                        # верхняя наддиагональ (upper form)
    ab[1] = diag
    rhs = np.column_stack([dt, y])
    try:
        sol = solveh_banded(ab, rhs)       # Σ⁻¹·dt, Σ⁻¹·y
    except np.linalg.LinAlgError:
        return -1, 0.0
    Pd, Py = sol[:, 0], sol[:, 1]
    # a(τ) = x1ᵀΣ⁻¹x1, x1 = dt·[i<τ]: нужны блочные суммы dt_i (Σ⁻¹)_ij dt_j —
    # считаем Σ⁻¹·diag(dt) столбцами только для кандидатов (разреженно)
    T_ = float(dt @ Pd)
    Ty = float(dt @ Py)
    e_pref = np.concatenate([[0.0], np.cumsum(dt * Pd)])    # x1ᵀΣ⁻¹dt
    b_pref = np.concatenate([[0.0], np.cumsum(dt * Py)])    # x1ᵀΣ⁻¹y
    # кандидаты τ (индекс приращения, с которого начинается 2-й кусок)
    cand = np.arange(min_pts, n - min_pts + 1)
    cand = cand[(tt[cand] - tt[0] >= min_T * 0.95) & (tt[-1] - tt[cand] >= min_T * 0.95)]
    if len(cand) == 0:
        return -1, 0.0
    step = max(1, len(cand) // 300)
    ks = cand[::step]
    # Σ⁻¹·x1(τ) для выбранных τ одним вызовом с несколькими правыми частями
    X1 = np.zeros((n, len(ks)))
    for j, k in enumerate(ks):
        X1[:k, j] = dt[:k]
    PX1 = solveh_banded(ab, X1)
    a = np.einsum("ij,ij->j", X1, PX1)
    e = e_pref[ks]
    bb = b_pref[ks]
    m11, m12, m22 = a, e - a, T_ - 2 * e + a
    r1, r2 = bb, Ty - bb
    det = m11 * m22 - m12 * m12
    ok = det > 1e-12
    gain = np.full(len(ks), -np.inf)
    zst = np.zeros(len(ks))
    i11, i12, i22 = m22[ok] / det[ok], -m12[ok] / det[ok], m11[ok] / det[ok]
    v1 = i11 * r1[ok] + i12 * r2[ok]
    v2 = i12 * r1[ok] + i22 * r2[ok]
    gain[ok] = r1[ok] * v1 + r2[ok] * v2
    var = i11 - 2 * i12 + i22
    zst[ok] = np.abs(v1 - v2) / np.sqrt(np.maximum(var, 1e-12))
    j = int(np.argmax(gain))
    if step > 1:                          # уточнение вокруг лучшего на полной сетке
        k0 = ks[j]
        fine = cand[(cand >= k0 - step) & (cand <= k0 + step)]
        X1 = np.zeros((n, len(fine)))
        for jj, k in enumerate(fine):
            X1[:k, jj] = dt[:k]
        PX1 = solveh_banded(ab, X1)
        a = np.einsum("ij,ij->j", X1, PX1)
        e, bb = e_pref[fine], b_pref[fine]
        m11, m12, m22 = a, e - a, T_ - 2 * e + a
        r1, r2 = bb, Ty - bb
        det = np.maximum(m11 * m22 - m12 * m12, 1e-12)
        i11, i12, i22 = m22 / det, -m12 / det, m11 / det
        v1 = i11 * r1 + i12 * r2
        v2 = i12 * r1 + i22 * r2
        g = r1 * v1 + r2 * v2
        jj = int(np.argmax(g))
        var = i11[jj] - 2 * i12[jj] + i22[jj]
        return lo + int(fine[jj]), float(abs(v1[jj] - v2[jj]) / np.sqrt(max(var, 1e-12)))
    return lo + int(ks[j]), float(zst[j])


def gls_zstat(t: np.ndarray, z: np.ndarray, D: float, sl: float, a0: int, b: int, a2: int) -> float:
    """z-статистика смены скорости ровно в точке b на куске [a0, a2) (GLS)."""
    from scipy.linalg import solveh_banded

    tt, zz = t[a0:a2], z[a0:a2]
    n = len(tt) - 1
    k = b - a0
    if k < 2 or n - k < 2:
        return -1.0
    dt, y = np.diff(tt), np.diff(zz)
    s2 = max(sl * sl, 1e-6)
    ab = np.zeros((2, n))
    ab[0, 1:] = -s2
    ab[1] = 2 * D * dt + 2 * s2
    X = np.zeros((n, 2))
    X[:k, 0] = dt[:k]
    X[k:, 1] = dt[k:]
    PX = solveh_banded(ab, np.column_stack([X, y]))
    M = X.T @ PX[:, :2]
    r = X.T @ PX[:, 2]
    Mi = np.linalg.inv(M)
    v = Mi @ r
    var = Mi[0, 0] - 2 * Mi[0, 1] + Mi[1, 1]
    return float(abs(v[0] - v[1]) / np.sqrt(max(var, 1e-12)))


def find_charge_breaks(t: np.ndarray, z: np.ndarray, cfg: SegmentConfig,
                       s: np.ndarray | None = None, d_floor: float = 0.0) -> tuple[list[int], LineFit]:
    """Изломы z(t) внутри интервала постоянного поля. Индексы — начало нового куска.

    Рекурсивная бисекция: на куске методом обобщённого МНК по приращениям
    (см. :func:`_gls_scan`) ищется лучший момент смены скорости; излом
    принимается при z ≥ k_sigma и обоих кусках ≥ min_piece_s, затем куски
    проверяются рекурсивно. В конце каждый излом перепроверяется на отрезке
    между соседними изломами; слабейший непрошедший удаляется.
    D и σ_loc — робастно по поперечной координате s (если дана).
    """
    n = len(t)
    whole = fit_line(t, z, cfg.msd_max_lag)
    if n < 6 or whole.T < 2 * cfg.min_piece_s:
        return [], whole
    # Шум (D, σ_loc) оценивается по поперечной координате s, если она дана:
    # диффузия изотропна, а перезарядка меняет скорость только вдоль поля,
    # поэтому излом по z не завышает оценку шума.
    D, sl = robust_noise(t, s if s is not None else z, cfg.noise_max_lag_s)
    # нижняя граница D (доля от типичного D по всем трекам): у отдельного трека
    # оценка D шумная, и заниженное D даёт ложные изломы на броуновском блуждании
    D = max(D, d_floor)
    whole = fit_line(t, z, cfg.msd_max_lag, D, sl)
    dt_med = float(np.median(np.diff(t)))
    min_pts = max(3, int(round(cfg.min_piece_s / dt_med * 0.8)))

    bps: list[int] = []
    stack = [(0, n)]
    while stack and len(bps) < cfg.max_breaks_per_interval * 2:
        lo, hi = stack.pop()
        if t[hi - 1] - t[lo] < 2 * cfg.min_piece_s:
            continue
        k, zs = _gls_scan(t, z, D, sl, lo, hi, min_pts, cfg.min_piece_s)
        if k >= 0 and zs >= cfg.k_sigma:
            bps.append(k)
            stack.extend([(lo, k + 1), (k, hi)])
    bps = sorted(set(bps))
    while bps:
        edges = [0] + bps + [n]
        sc = [gls_zstat(t, z, D, sl, edges[j], b, edges[j + 2]) for j, b in enumerate(bps)]
        j = int(np.argmin(sc))
        if sc[j] >= cfg.k_sigma and len(bps) <= cfg.max_breaks_per_interval:
            break
        bps.pop(j)
    return bps, whole


def population_D(tracks: pd.DataFrame, sw: np.ndarray, cfg: SegmentConfig, min_T: float = 3.0) -> float:
    """Типичный (медианный) D по поперечной координате s для кусков ≥ min_T секунд."""
    real = tracks[tracks.interpolated == 0]
    iv = np.searchsorted(sw, real.t_s.values, side="right")
    Ds = []
    for (_pid, _iv), g in real.assign(_iv=iv).groupby(["particle_id", "_iv"]):
        if len(g) < 30 or g.t_s.iloc[-1] - g.t_s.iloc[0] < min_T:
            continue
        D, _ = robust_noise(g.t_s.values, g.s_px.values, cfg.noise_max_lag_s)
        Ds.append(D)
        if len(Ds) >= 400:
            break
    return float(np.median(Ds)) if Ds else 0.0


def calibrate_k_sigma(tracks: pd.DataFrame, sw: np.ndarray, cfg: SegmentConfig, d_floor: float,
                      max_pieces: int = 2000) -> tuple[float, int]:
    """Порог z для изломов по «нулевой» координате s того же видео.

    Перезарядка меняет скорость только вдоль поля (z); вдоль линий шкалы (s)
    изломов быть не должно. Для кусков треков (постоянное поле, ≥ 3 с) считается
    максимальная по положению излома z-статистика GLS на s; порог —
    её квантиль (1 − null_fa_rate). Так учитываются особенности реального шума
    (сжатие видео, скачки центра при смене «точка ↔ кольцо»), которых нет в
    модели «броуновское движение + белый шум локализации».
    Возвращает (порог, число кусков).
    """
    real = tracks[tracks.interpolated == 0]
    iv = np.searchsorted(sw, real.t_s.values, side="right")
    zs = []
    for (_pid, _iv), g in real.assign(_iv=iv).groupby(["particle_id", "_iv"]):
        t = g.t_s.values
        if len(t) < 30 or t[-1] - t[0] < max(3.0, 2 * cfg.min_piece_s):
            continue
        sv = g.s_px.values
        D, sl = robust_noise(t, sv, cfg.noise_max_lag_s)
        D = max(D, d_floor)
        mp = max(3, int(round(cfg.min_piece_s / float(np.median(np.diff(t))) * 0.8)))
        _k, zmax = _gls_scan(t, sv, D, sl, 0, len(t), mp, cfg.min_piece_s)
        zs.append(zmax)
        if len(zs) >= max_pieces:
            break
    if len(zs) < 50:
        return float(cfg.k_sigma_floor), len(zs)
    return float(max(cfg.k_sigma_floor, np.quantile(zs, 1 - cfg.null_fa_rate))), len(zs)


def segment_tracks(tracks: pd.DataFrame, fs: FieldSwitches, cal: Calibration,
                   cfg: SegmentConfig, t_last: float, max_gap_s: float,
                   edge_ids: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Нарезать треки. Возвращает (tracks с segment_id, segments).

    В режиме ``stop`` строки после первой перезарядки остаются в tracks с
    segment_id = −1 и флагом ``after_charge_change`` (ничего не удаляется молча).
    """
    tracks = tracks.copy()
    tracks["segment_id"] = -1
    tracks["field_state"] = fs.state_at(tracks.t_s.values)
    sw = np.asarray(fs.times)
    edge_ids = edge_ids or set()
    d_floor = cfg.d_floor_frac * population_D(tracks, sw, cfg)
    log.info("нарезка: нижняя граница D = %.3f px²/с (%.2f × типичного D по трекам)", d_floor, cfg.d_floor_frac)
    if cfg.k_sigma == "auto":
        k, n_null = calibrate_k_sigma(tracks, sw, cfg, d_floor)
        log.info("нарезка: порог k_sigma = %.2f (квантиль %.3f нулевого распределения по s, %d кусков)",
                 k, 1 - cfg.null_fa_rate, n_null)
        cfg = dataclasses.replace(cfg, k_sigma=k)
    self_k = float(cfg.k_sigma)
    seg_rows = []
    after_charge = np.zeros(len(tracks), bool)
    for pid, g in tracks.groupby("particle_id", sort=True):
        idx = g.index.values
        t_all = g.t_s.values
        real = g.interpolated.values == 0
        interval = np.searchsorted(sw, t_all, side="right")
        seg_id = 0
        stopped = False
        pieces: list[tuple[np.ndarray, str]] = []   # (маска строк, end_reason)
        for iv in np.unique(interval):
            m_iv = interval == iv
            if stopped:
                after_charge[idx[m_iv]] = True
                continue
            lo = sw[iv - 1] + cfg.skip_after_switch_s if iv > 0 else -np.inf
            fit_mask = m_iv & real & (t_all >= lo)
            tt, zz = t_all[fit_mask], g.z_px.values[fit_mask]
            ss = g.s_px.values[fit_mask]
            breaks, _whole = find_charge_breaks(tt, zz, cfg, ss, d_floor) if len(tt) else ([], None)
            # границы кусков во времени
            cut_t = [tt[b] for b in breaks]
            bounds = [-np.inf] + cut_t + [np.inf]
            for k in range(len(bounds) - 1):
                m = m_iv & (t_all >= bounds[k]) & (t_all < bounds[k + 1])
                if k < len(bounds) - 2:
                    reason = "charge_change"
                elif np.any(interval > iv):      # трек продолжается после переключения
                    reason = "field_switch"
                else:
                    reason = "end"
                pieces.append((m, reason))
                if reason == "charge_change" and cfg.mode == "stop":
                    rest = m_iv & (t_all >= bounds[k + 1])
                    after_charge[idx[rest]] = True
                    stopped = True
                    break
        last_t = t_all.max()
        for m, reason in pieces:
            if not m.any():
                continue
            rows = idx[m]
            tracks.loc[rows, "segment_id"] = seg_id
            if reason == "end":
                if last_t >= t_last - max_gap_s - 1e-9:
                    reason = "video_end"
                elif pid in edge_ids:
                    reason = "edge"
                else:
                    reason = "lost"
            gm = g.loc[rows]
            gr = gm[gm.interpolated == 0]
            iv0 = int(np.searchsorted(sw, gm.t_s.values[0], side="right"))
            lo = sw[iv0 - 1] + cfg.skip_after_switch_s if iv0 > 0 else -np.inf
            gr = gr[gr.t_s >= lo]
            if len(gr) >= 2:
                fz = fit_line(gr.t_s.values, gr.z_px.values, cfg.msd_max_lag)
                fsl = fit_line(gr.t_s.values, gr.s_px.values, cfg.msd_max_lag)
            else:
                fz = fsl = LineFit(np.nan, np.inf, np.nan, np.nan, np.nan, len(gr), 0.0)
            seg_rows.append({
                "particle_id": int(pid), "segment_id": seg_id,
                "t_start": float(gm.t_s.min()), "t_end": float(gm.t_s.max()),
                "n_frames": int(len(gm)), "n_real": int(len(gr)),
                "field_state": str(fs.state_at(gm.t_s.values[len(gm) // 2])),
                "v_z_px_s": fz.v, "v_z_err_px_s": fz.v_err,
                "v_s_px_s": fsl.v, "v_s_err_px_s": fsl.v_err,
                "D_z_px2_s": fz.D, "sigma_loc_px": fz.sigma_loc,
                "r_mean_px": float(gm.r_px.mean()),
                "z_mean_px": float(gr.z_px.mean()) if len(gr) else float(gm.z_px.mean()),
                "end_reason": reason,
            })
            seg_id += 1
    tracks.loc[after_charge, "segment_id"] = -1
    tracks["after_charge_change"] = after_charge.astype(int)
    seg = pd.DataFrame(seg_rows)
    if len(seg):
        # скорости в делениях шкалы/с: не зависят от того, как близко стояла камера;
        # шаг — местный (перспектива), в средней точке участка
        st = cal.local_step(seg.z_mean_px.values)
        seg["step_local_px"] = st
        seg["v_z_div_s"] = seg.v_z_px_s / st
        seg["v_z_err_div_s"] = seg.v_z_err_px_s / st
        seg["v_s_div_s"] = seg.v_s_px_s / st
        seg["v_s_err_div_s"] = seg.v_s_err_px_s / st
        seg["D_z_div2_s"] = seg.D_z_px2_s / st**2
        # мм = деления × цена деления (через местный шаг, т. е. с учётом перспективы)
        mmd = cal.mm_per_px * cal.step_px if cal.mm_per_px else None
        for a, b in (("v_z_div_s", "v_z_mm_s"), ("v_z_err_div_s", "v_z_err"), ("v_s_div_s", "v_s_mm_s"),
                     ("v_s_err_div_s", "v_s_err")):
            seg[b] = seg[a] * mmd if mmd else np.nan
        seg["D_mm2_s"] = seg["D_z_div2_s"] * mmd * mmd if mmd else np.nan
    if len(seg):
        seg["k_sigma_used"] = self_k
    n_cc = int((seg.end_reason == "charge_change").sum()) if len(seg) else 0
    log.info("нарезка: %d участков, перезарядок найдено: %d (режим %s)", len(seg), n_cc, cfg.mode)
    return tracks, seg

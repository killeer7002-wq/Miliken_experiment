"""Моменты переключения поля: из конфига, авто-поиск и уточнение по трекам.

Источник 1 (приоритет): ``field.switches_s`` из конфига (протокол опыта).

Источник 2 (авто): грубые скорости без трекинга — каждая детекция кадра f
сопоставляется с ближайшей детекцией кадра f+lag (гейт ``auto_gate_px``,
только взаимно-ближайшие пары), v_z = Δz/Δt. На окнах ``auto_window_s``
считаются 5-й и 95-й перцентили v_z и доля «быстрых» капель
(|v_z| > ``auto_fast_px_s``). Эти три ряда нормируются и подаются в PELT
(сдвиг среднего); штраф — BIC-подобный, из шума рядов.

Источник 3 (уточнение, ``auto_refine``): у треков, переживших переключение,
скорость меняется одновременно. Для каждого кандидата τ в окне
±``refine_max_shift_s`` вокруг момента считается суммарная по трекам
ошибка кусочно-линейной подгонки z(t) с изломом в τ (непрерывной в τ);
минимум даёт момент с точностью до кадра. Для диагностики сохраняется и
медиана поштучных оценок по трекам.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .calibration import Calibration
from .changepoint import pelt_mean, robust_noise_var
from .config import FieldConfig, field_states

log = logging.getLogger(__name__)


@dataclass
class FieldSwitches:
    times: list[float]
    frames: list[int]
    source: list[str]
    states: list[str]             # len(times) + 1
    auto_times: list[float] | None = None
    auto_track_times: list[float] | None = None
    config_times: list[float] | None = None
    refine_shift_s: list[float] | None = None
    refine_median_s: list[float] | None = None
    refine_n_tracks: list[int] | None = None

    def state_at(self, t: np.ndarray | float) -> np.ndarray:
        idx = np.searchsorted(np.asarray(self.times), np.asarray(t), side="right")
        return np.asarray(self.states, dtype=object)[idx]

    def interval_index(self, t: np.ndarray | float) -> np.ndarray:
        return np.searchsorted(np.asarray(self.times), np.asarray(t), side="right")

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for i, (t, f, s) in enumerate(zip(self.times, self.frames, self.source)):
            row = {"index": i, "t_s": round(t, 4), "frame": f, "source": s,
                   "state_before": self.states[i], "state_after": self.states[i + 1]}
            if self.config_times is not None and i < len(self.config_times):
                row["t_config_s"] = self.config_times[i]
            if self.auto_times is not None:
                if self.auto_times:
                    j = int(np.argmin(np.abs(np.asarray(self.auto_times) - t)))
                    row["t_auto_s"] = round(self.auto_times[j], 4)
                    row["auto_minus_used_s"] = round(self.auto_times[j] - t, 4)
                else:
                    row["t_auto_s"] = np.nan
            if self.auto_track_times is not None:
                if self.auto_track_times:
                    j = int(np.argmin(np.abs(np.asarray(self.auto_track_times) - t)))
                    row["t_auto_tracks_s"] = round(self.auto_track_times[j], 4)
                else:
                    row["t_auto_tracks_s"] = np.nan
            if self.refine_shift_s is not None:
                row["refine_shift_s"] = self.refine_shift_s[i]
                row["refine_median_s"] = self.refine_median_s[i] if self.refine_median_s else np.nan
                row["refine_n_tracks"] = self.refine_n_tracks[i] if self.refine_n_tracks else 0
            rows.append(row)
        cols = ["index", "t_s", "frame", "source", "state_before", "state_after"]
        return pd.DataFrame(rows, columns=None if rows else cols)


def coarse_velocity_stats(det: pd.DataFrame, cal: Calibration, t_frames: np.ndarray,
                          cfg: FieldConfig) -> pd.DataFrame:
    """Статистика грубых скоростей по окнам времени (без трекинга)."""
    z, s = cal.to_zs(det.x.values, det.y.values)
    fr = det.frame.values
    frames = np.unique(fr)
    by = {f: np.nonzero(fr == f)[0] for f in frames}
    lag = cfg.auto_lag_frames
    vt, vz = [], []
    for f in frames:
        g = f + lag
        if g not in by:
            continue
        a, b = by[f], by[g]
        if len(a) == 0 or len(b) == 0:
            continue
        pa = np.column_stack([z[a], s[a]])
        pb = np.column_stack([z[b], s[b]])
        d1, j1 = cKDTree(pb).query(pa, distance_upper_bound=cfg.auto_gate_px)
        ok = np.isfinite(d1)
        if not ok.any():
            continue
        _d2, j2 = cKDTree(pa).query(pb, distance_upper_bound=cfg.auto_gate_px)
        ia = np.nonzero(ok)[0]
        mutual = j2[j1[ia]] == ia
        ia = ia[mutual]
        dt = t_frames[g] - t_frames[f]
        v = (pb[j1[ia], 0] - pa[ia, 0]) / dt
        vz.append(v)
        vt.append(np.full(len(v), 0.5 * (t_frames[f] + t_frames[g])))
    if not vz:
        return pd.DataFrame(columns=["t", "p5", "p95", "frac_fast", "median", "n"])
    vz_a, vt_a = np.concatenate(vz), np.concatenate(vt)
    t0, t1 = float(t_frames[frames[0]]), float(t_frames[frames[-1]])
    edges = np.arange(t0, t1 + cfg.auto_window_s, cfg.auto_window_s)
    idx = np.digitize(vt_a, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        v = vz_a[idx == k]
        if len(v) < 10:
            continue
        rows.append({"t": 0.5 * (edges[k] + edges[k + 1]),
                     "p5": np.percentile(v, 5), "p95": np.percentile(v, 95),
                     "frac_fast": float(np.mean(np.abs(v) > cfg.auto_fast_px_s)),
                     "median": float(np.median(v)), "n": len(v)})
    return pd.DataFrame(rows)


def auto_switches(stats: pd.DataFrame, cfg: FieldConfig) -> list[float]:
    """Моменты скачков статистики скоростей (PELT по нормированным рядам)."""
    if len(stats) < 6:
        return []
    Y = stats[["p5", "p95", "frac_fast"]].values.astype(float)
    Y = (Y - np.median(Y, axis=0)) / (np.std(Y, axis=0) + 1e-9)
    var = robust_noise_var(Y)
    n, d = Y.shape
    if cfg.auto_penalty == "auto":
        pen = 3.0 * d * var * np.log(n)
    else:
        pen = float(cfg.auto_penalty)
    min_size = max(2, int(round(cfg.auto_min_size_s / cfg.auto_window_s)))
    bps = pelt_mean(Y, pen, min_size)
    # проверка величины скачка: медленный дрейф статистики PELT тоже режет,
    # а переключение поля — это резкий скачок на много σ шума хотя бы в одном ряду
    d = np.diff(Y, axis=0)
    sig = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0) / np.sqrt(2) + 1e-9
    w = min_size
    while bps:
        edges = [0] + bps + [n]
        scores = []
        for j, b in enumerate(bps):
            a0 = max(edges[j], b - w)
            a2 = min(edges[j + 2], b + w)
            jump = np.abs(np.median(Y[b:a2], axis=0) - np.median(Y[a0:b], axis=0)) / sig
            scores.append(float(jump.max()))
        k = int(np.argmin(scores))
        if scores[k] >= cfg.auto_min_jump_sigma:
            break
        bps.pop(k)
    t = stats.t.values
    # момент скачка — граница между окнами
    return [float(0.5 * (t[b - 1] + t[b])) for b in bps]


def track_change_series(tracks: pd.DataFrame, t_frames: np.ndarray, cfg: FieldConfig,
                        max_tracks: int = 3000) -> pd.DataFrame:
    """Знаконезависимый признак переключения по трекам.

    Для каждого трека и каждого кадра — МНК-скорости v_L и v_R на окнах
    ``track_window_s`` слева и справа; χ² = (v_R − v_L)² / Var, где Var —
    сумма дисперсий двух наклонов по формуле σ_v² = 12D/(5W) + 12σ²/(N W²)
    (D, σ — робастно по поперечной координате s трека). Ряд — среднее по
    трекам min(χ², 25) на каждом кадре. При переключении скачок скорости у
    всех заряженных капель одновременен, и ряд даёт острый пик — неважно,
    меняется ли знак поля или поле включается/выключается и какого знака
    заряды. Популяционная статистика (перцентили скоростей) этого не умеет,
    если заряды обоих знаков встречаются поровну.
    """
    from .segment import robust_noise

    nf = len(t_frames)
    acc = np.zeros(nf)
    cnt = np.zeros(nf)
    real = tracks[tracks.interpolated == 0]
    dur = real.groupby("particle_id").t_s.agg(lambda x: x.max() - x.min())
    W = cfg.track_window_s
    ids = dur.index[dur >= 2.5 * W]
    if len(ids) > max_tracks:
        ids = dur.loc[ids].sort_values(ascending=False).index[:max_tracks]
    for pid in ids:
        g = real[real.particle_id == pid]
        t = g.t_s.values
        z = g.z_px.values
        f = g.frame.values
        D, sl = robust_noise(t, g.s_px.values, 1.0)
        # кумулятивные суммы для МНК-наклона на любом окне [a, b)
        c1 = np.concatenate([[0], np.cumsum(np.ones_like(t))])
        ct = np.concatenate([[0], np.cumsum(t)])
        ct2 = np.concatenate([[0], np.cumsum(t * t)])
        cz = np.concatenate([[0], np.cumsum(z)])
        ctz = np.concatenate([[0], np.cumsum(t * z)])

        def slope(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = c1[b] - c1[a]
            st, st2 = ct[b] - ct[a], ct2[b] - ct2[a]
            sz, stz = cz[b] - cz[a], ctz[b] - ctz[a]
            den = n * st2 - st * st
            return (n * stz - st * sz) / np.maximum(den, 1e-12), n

        i = np.arange(len(t))
        aL = np.searchsorted(t, t - W)
        bR = np.searchsorted(t, t + W)
        ok = (t - t[0] >= W * 0.9) & (t[-1] - t >= W * 0.9)
        ok &= (i - aL >= 5) & (bR - i >= 5)
        if not ok.any():
            continue
        ii = i[ok]
        vL, nL = slope(aL[ok], ii)
        vR, nR = slope(ii, bR[ok])
        var = (12 * D / (5 * W) + 12 * sl * sl / (nL * W * W)) + (12 * D / (5 * W) + 12 * sl * sl / (nR * W * W))
        chi2 = np.minimum((vR - vL) ** 2 / np.maximum(var, 1e-9), 25.0)
        np.add.at(acc, f[ok], chi2)
        np.add.at(cnt, f[ok], 1)
    val = np.where(cnt >= 5, acc / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame({"frame": np.arange(nf), "t": t_frames, "chi2": val, "n": cnt.astype(int)})


def auto_switches_tracks(series: pd.DataFrame, cfg: FieldConfig) -> list[float]:
    """Пики ряда :func:`track_change_series` выше медиана + k·MAD."""
    v = series.chi2.values
    good = np.isfinite(v)
    if good.sum() < 10:
        return []
    vv = np.where(good, v, np.nanmedian(v))
    med = np.median(vv[good])
    mad = 1.4826 * np.median(np.abs(vv[good] - med)) + 1e-9
    thr = med + cfg.auto_track_k_mad * mad
    t = series.t.values
    min_dist = cfg.auto_min_size_s
    order = np.argsort(-vv)
    peaks: list[int] = []
    for k in order:
        if vv[k] < thr:
            break
        if all(abs(t[k] - t[p]) >= min_dist for p in peaks):
            peaks.append(int(k))
    return sorted(float(t[p]) for p in peaks)


def _piecewise_sse(tt: np.ndarray, zz: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """SSE непрерывной ломаной с изломом в τ для каждого τ (базис 1, t, (t−τ)+)."""
    out = np.empty(len(taus))
    for k, tau in enumerate(taus):
        A = np.column_stack([np.ones_like(tt), tt - tau, np.maximum(tt - tau, 0)])
        coef, res, rank, _ = np.linalg.lstsq(A, zz, rcond=None)
        out[k] = res[0] if res.size else float(np.sum((A @ coef - zz) ** 2))
    return out


def refine_switches(times: list[float], tracks: pd.DataFrame, t_frames: np.ndarray,
                    cfg: FieldConfig) -> tuple[list[float], list[float], list[float], list[int]]:
    """Уточнить моменты переключения по трекам, пережившим переключение."""
    new_t, shifts, medians, ntr = [], [], [], []
    real = tracks[tracks.interpolated == 0]
    W = cfg.refine_max_shift_s
    span = W + 1.5
    for s0 in times:
        cand_f = np.nonzero(np.abs(t_frames - s0) <= W)[0]
        taus = t_frames[cand_f]
        if len(taus) < 3:
            new_t.append(s0); shifts.append(0.0); medians.append(np.nan); ntr.append(0)
            continue
        win = real[(real.t_s >= s0 - span) & (real.t_s <= s0 + span)]
        total = np.zeros(len(taus))
        per = []
        n_used = 0
        for _pid, g in win.groupby("particle_id"):
            tt = g.t_s.values
            if tt.min() > s0 - W - 0.5 or tt.max() < s0 + W + 0.5 or len(tt) < 30:
                continue
            zz = g.z_px.values
            sse = _piecewise_sse(tt - s0, zz, taus - s0)
            # вклад трека нормируется на его шум, чтобы шумные треки не доминировали
            base = sse.max()
            gain = base - sse.min()
            noise = np.var(np.diff(zz)) * len(zz) + 1e-9
            if gain / noise < 0.05:      # излома нет — трек не информативен
                continue
            total += sse / noise
            per.append(taus[int(np.argmin(sse))])
            n_used += 1
        if n_used >= cfg.refine_min_tracks:
            t_best = float(taus[int(np.argmin(total))])
            new_t.append(t_best)
            shifts.append(round(t_best - s0, 4))
            medians.append(round(float(np.median(per)), 4))
        else:
            new_t.append(s0); shifts.append(0.0)
            medians.append(round(float(np.median(per)), 4) if per else np.nan)
        ntr.append(n_used)
    return new_t, shifts, medians, ntr


def determine_switches(det: pd.DataFrame, cal: Calibration, t_frames: np.ndarray,
                       cfg: FieldConfig, i0: int, i1: int) -> tuple[FieldSwitches, pd.DataFrame]:
    """Собрать моменты переключения (конфиг + авто), без уточнения по трекам."""
    stats = coarse_velocity_stats(det, cal, t_frames, cfg)
    auto = auto_switches(stats, cfg)
    t_lo, t_hi = float(t_frames[i0]), float(t_frames[i1 - 1])
    if cfg.switches_s is not None:
        cfg_t = [float(x) for x in cfg.switches_s]
        times = [x for x in cfg_t if t_lo < x < t_hi]
        source = ["config"] * len(times)
        log.info("переключения поля из конфига: %s", times)
    else:
        cfg_t = None
        times = auto
        source = ["auto"] * len(times)
    log.info("переключения поля (авто): %s", [round(a, 2) for a in auto])
    if cfg_t is not None:
        for x in times:
            if auto:
                j = int(np.argmin(np.abs(np.asarray(auto) - x)))
                log.info("  конфиг %.2f с ↔ авто %.2f с (расхождение %+.2f с)", x, auto[j], auto[j] - x)
            else:
                log.info("  конфиг %.2f с ↔ авто: не найдено", x)
    # подписи состояний: если интервал обрезан (--t-start), сдвигаем нумерацию
    all_states = field_states(cfg, len(cfg_t) if cfg_t is not None else len(times))
    if cfg_t is not None:
        n_before = sum(1 for x in cfg_t if x <= t_lo)
        states = all_states[n_before:n_before + len(times) + 1]
    else:
        states = all_states
    frames = [int(np.searchsorted(t_frames, x)) for x in times]
    fs = FieldSwitches(times, frames, source, states, auto_times=auto, config_times=cfg_t)
    return fs, stats


def apply_refinement(fs: FieldSwitches, tracks: pd.DataFrame, t_frames: np.ndarray,
                     cfg: FieldConfig) -> FieldSwitches:
    if not fs.times:
        fs.refine_shift_s, fs.refine_median_s, fs.refine_n_tracks = [], [], []
        return fs
    new_t, shifts, medians, ntr = refine_switches(fs.times, tracks, t_frames, cfg)
    for i, (a, b, n) in enumerate(zip(fs.times, new_t, ntr)):
        log.info("уточнение переключения %d: %.3f → %.3f с (%+.3f с, по %d трекам)", i, a, b, b - a, n)
    fs.times = new_t
    fs.frames = [int(np.searchsorted(t_frames, x)) for x in new_t]
    fs.source = [s + "+refined" if sh != 0 else s for s, sh in zip(fs.source, shifts)]
    fs.refine_shift_s, fs.refine_median_s, fs.refine_n_tracks = shifts, medians, ntr
    return fs


def replace_times(fs: FieldSwitches, times: list[float], t_frames: np.ndarray, cfg: FieldConfig,
                  source: str) -> FieldSwitches:
    """Заменить моменты переключения (например, на найденные по трекам)."""
    fs.times = list(times)
    fs.frames = [int(np.searchsorted(t_frames, x)) for x in times]
    fs.source = [source] * len(times)
    fs.states = field_states(cfg, len(times))
    return fs

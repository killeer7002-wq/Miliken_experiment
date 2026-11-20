"""Сопоставление детекций в треки: фильтр Калмана + венгерский алгоритм.

Модель на каждую каплю — постоянная скорость в координатах (z, s):
состояние X = [z, s, v_z, v_s], шаг dt берётся из PTS (частота плавающая).
Шум процесса: броуновская диффузия координаты (2·D·dt) плюс медленное
блуждание скорости (accel²·dt). Шум измерения — σ_loc (у колец больше,
пропорционально радиусу: их центр локализуется хуже).

Назначение на кадре: матрица стоимостей только для пар внутри гейта
(евклидово расстояние до предсказания ≤ gate), стоимость =
расстояние Махаланобиса² + w_r·|Δr|/r̄ + w_I·|Δ ln I|; решается
``scipy.optimize.linear_sum_assignment``.

Переключение поля (известный глобальный момент): у всех активных треков
скорость обнуляется, её дисперсия ставится v_init_sigma², гейт на
``switch_reset_s`` секунд расширяется до ``gate_after_switch_px``. Так
предсказатель не «улетает» по старой скорости, а новая скорость быстро
оценивается заново.

Пропуски: трек живёт до ``max_gap_frames`` кадров без детекций (капля на
линии шкалы, вне фокуса); при повторном обнаружении пропущенные кадры
заполняются линейной интерполяцией между реальными детекциями
(``interpolated = 1``). Если две капли ближе r1 + r2 — у обеих на этом кадре
``merge_risk = 1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .calibration import Calibration
from .config import LinkConfig

log = logging.getLogger(__name__)

BIG = 1e9


@dataclass
class _TrackRows:
    frames: list[int] = field(default_factory=list)
    det: list[int] = field(default_factory=list)
    merge: list[int] = field(default_factory=list)


class Linker:
    """Пошаговый трекер. Используйте :func:`link_detections` для всего ролика."""

    def __init__(self, cfg: LinkConfig, switch_times: list[float] | None = None):
        self.cfg = cfg
        self.switch_times = sorted(switch_times or [])
        self.X = np.zeros((0, 4))
        self.P = np.zeros((0, 4, 4))
        self.tid = np.zeros(0, int)
        self.miss = np.zeros(0, int)
        self.reset_until = np.zeros(0)
        self.last_r = np.zeros(0)
        self.last_li = np.zeros(0)
        self.rows: dict[int, _TrackRows] = {}
        self.closed: list[int] = []
        self.next_id = 0
        self.t_prev: float | None = None
        self.n_switch_resets = 0

    # ------------------------------------------------------------ Калман
    def _predict(self, dt: float) -> None:
        if len(self.X) == 0 or dt <= 0:
            return
        c = self.cfg
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        Q = np.diag([2 * c.diffusion_px2_s * dt, 2 * c.diffusion_px2_s * dt,
                     c.accel_px_s2**2 * dt, c.accel_px_s2**2 * dt])
        self.X = self.X @ F.T
        self.P = F[None] @ self.P @ F.T[None] + Q[None]

    def _switch_reset(self, t_sw: float) -> None:
        c = self.cfg
        if len(self.X):
            self.X[:, 2:] = 0.0
            self.P[:, 2:, :] = 0.0
            self.P[:, :, 2:] = 0.0
            self.P[:, 2, 2] = self.P[:, 3, 3] = c.v_init_sigma_px_s**2
            self.reset_until[:] = t_sw + c.switch_reset_s
        self.n_switch_resets += 1

    def _sigma_meas(self, r: np.ndarray) -> np.ndarray:
        return self.cfg.sigma_loc_px * np.maximum(1.0, r / 3.0)

    # ------------------------------------------------------------ шаг
    def step(self, frame: int, t: float, det_idx: np.ndarray, zs: np.ndarray,
             r: np.ndarray, inten: np.ndarray) -> None:
        c = self.cfg
        if self.t_prev is not None:
            dt = t - self.t_prev
            # переключение поля между кадрами: сначала прогноз до момента
            # переключения, сброс скорости, затем прогноз до текущего кадра
            sw = [s for s in self.switch_times if self.t_prev < s <= t]
            tp = self.t_prev
            for s in sw:
                self._predict(s - tp)
                self._switch_reset(s)
                tp = s
            self._predict(t - tp)
            del dt
        self.t_prev = t

        n, m = len(self.X), len(zs)
        li = np.log(np.maximum(inten, 1e-3))
        matched_t = np.full(n, -1)
        matched_d = np.full(m, -1)
        if n and m:
            pred = self.X[:, :2]
            gate = np.where(t < self.reset_until, c.gate_after_switch_px, c.gate_px)
            # гейт растёт с радиусом: центр кольца локализуется хуже, а при
            # переходе «точка ↔ кольцо» центр детекции смещается на доли r
            gmax = gate + c.gate_r_factor * np.maximum(self.last_r, float(r.max()))
            tree = cKDTree(zs)
            pairs_t, pairs_d = [], []
            for i, nb in enumerate(tree.query_ball_point(pred, gmax)):
                if nb:
                    pairs_t.extend([i] * len(nb))
                    pairs_d.extend(nb)
            if pairs_t:
                it = np.asarray(pairs_t)
                jd = np.asarray(pairs_d)
                nu = zs[jd] - pred[it]                          # (K, 2)
                g_pair = gate[it] + c.gate_r_factor * np.maximum(self.last_r[it], r[jd])
                inside = np.hypot(nu[:, 0], nu[:, 1]) <= g_pair
                it, jd, nu = it[inside], jd[inside], nu[inside]
            if pairs_t and len(it):
                sm = self._sigma_meas(r[jd]) ** 2
                a = self.P[it, 0, 0] + sm
                b = self.P[it, 0, 1]
                d_ = self.P[it, 1, 1] + sm
                det = a * d_ - b * b
                maha = (d_ * nu[:, 0] ** 2 - 2 * b * nu[:, 0] * nu[:, 1] + a * nu[:, 1] ** 2) / det
                rbar = 0.5 * (self.last_r[it] + r[jd])
                cost = maha + c.w_r * np.abs(self.last_r[it] - r[jd]) / np.maximum(rbar, 1e-3) \
                    + c.w_i * np.abs(self.last_li[it] - li[jd])
                rows_u, ri = np.unique(it, return_inverse=True)
                cols_u, ci = np.unique(jd, return_inverse=True)
                C = np.full((len(rows_u), len(cols_u)), BIG)
                np.minimum.at(C, (ri, ci), cost)
                rr, cc = linear_sum_assignment(C)
                ok = C[rr, cc] < BIG
                ti, dj = rows_u[rr[ok]], cols_u[cc[ok]]
                matched_t[ti] = dj
                matched_d[dj] = ti

        # обновление сопоставленных
        mt = np.nonzero(matched_t >= 0)[0]
        mt_tids = self.tid[mt].copy()
        if len(mt):
            dj = matched_t[mt]
            P = self.P[mt]
            sm = self._sigma_meas(r[dj]) ** 2
            S = P[:, :2, :2] + sm[:, None, None] * np.eye(2)[None]
            Sinv = np.linalg.inv(S)
            K = P[:, :, :2] @ Sinv                              # (k, 4, 2)
            nu = zs[dj] - self.X[mt, :2]
            self.X[mt] = self.X[mt] + np.einsum("kij,kj->ki", K, nu)
            IKH = np.eye(4)[None] - K @ np.concatenate([np.eye(2), np.zeros((2, 2))], axis=1)[None]
            self.P[mt] = IKH @ P
            self.miss[mt] = 0
            self.last_r[mt] = r[dj]
            self.last_li[mt] = li[dj]
            for i, j in zip(mt, dj):
                tr = self.rows[int(self.tid[i])]
                tr.frames.append(frame)
                tr.det.append(int(det_idx[j]))
                tr.merge.append(0)
        # несопоставленные треки
        um = np.nonzero(matched_t < 0)[0]
        self.miss[um] += 1
        dead = self.miss > c.max_gap_frames
        if dead.any():
            self.closed.extend(int(x) for x in self.tid[dead])
            keep = ~dead
            self._keep(keep)
        # новые треки
        new = np.nonzero(matched_d < 0)[0]
        ids = np.zeros(0, int)
        if len(new):
            k = len(new)
            X = np.zeros((k, 4))
            X[:, :2] = zs[new]
            P = np.zeros((k, 4, 4))
            sm = self._sigma_meas(r[new]) ** 2
            P[:, 0, 0] = P[:, 1, 1] = sm
            P[:, 2, 2] = P[:, 3, 3] = c.v_init_sigma_px_s**2
            ids = np.arange(self.next_id, self.next_id + k)
            self.next_id += k
            # новый трек сразу после переключения тоже получает широкий гейт
            ru = max((s + c.switch_reset_s for s in self.switch_times if s <= t), default=-np.inf)
            self.X = np.concatenate([self.X, X])
            self.P = np.concatenate([self.P, P])
            self.tid = np.concatenate([self.tid, ids])
            self.miss = np.concatenate([self.miss, np.zeros(k, int)])
            self.reset_until = np.concatenate([self.reset_until, np.full(k, ru)])
            self.last_r = np.concatenate([self.last_r, r[new]])
            self.last_li = np.concatenate([self.last_li, li[new]])
            for i, j in zip(ids, new):
                self.rows[int(i)] = _TrackRows([frame], [int(det_idx[j])], [0])

        # риск слияния: две детекции этого кадра ближе r1 + r2
        if m > 1:
            det2tid = np.full(m, -1)
            if len(mt):
                det2tid[matched_t[mt]] = mt_tids
            if len(new):
                det2tid[new] = ids
            tree = cKDTree(zs)
            for a_, b_ in tree.query_pairs(2 * float(r.max())):
                if np.hypot(*(zs[a_] - zs[b_])) < r[a_] + r[b_]:
                    for j in (a_, b_):
                        if det2tid[j] >= 0:
                            self.rows[int(det2tid[j])].merge[-1] = 1

    def _keep(self, keep: np.ndarray) -> None:
        self.X, self.P, self.tid = self.X[keep], self.P[keep], self.tid[keep]
        self.miss, self.reset_until = self.miss[keep], self.reset_until[keep]
        self.last_r, self.last_li = self.last_r[keep], self.last_li[keep]

    def finish(self) -> None:
        self.closed.extend(int(x) for x in self.tid)
        self._keep(np.zeros(len(self.tid), bool))


def _end_state(frames: np.ndarray, zs: np.ndarray, t_frames: np.ndarray, at_end: bool,
               n_fit: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Время, позиция и скорость на конце (at_end) или начале обрывка (МНК по n_fit точкам)."""
    sel = slice(-n_fit, None) if at_end else slice(0, n_fit)
    f, p = frames[sel], zs[sel]
    tt = t_frames[f]
    t_ref = tt[-1] if at_end else tt[0]
    if len(tt) >= 3 and tt[-1] > tt[0]:
        A = np.column_stack([np.ones_like(tt), tt - t_ref])
        coef, *_ = np.linalg.lstsq(A, p, rcond=None)
        return float(t_ref), coef[0], coef[1]
    return float(t_ref), (p[-1] if at_end else p[0]).astype(float), np.zeros(2)


def stitch_tracklets(rows: dict[int, _TrackRows], zs_all: np.ndarray, r_all: np.ndarray,
                     t_frames: np.ndarray, cfg: LinkConfig, switch_times: list[float]) -> int:
    """Склейка обрывков (gap closing): конец трека A ↔ начало трека B.

    Кандидаты: B начинается через 1..stitch_max_gap кадров после конца A;
    A экстраполируется вперёд, B — назад (по скорости из МНК на краях;
    если между ними переключение поля — скорость считается нулевой);
    рассогласование ≤ stitch_gate_px + stitch_gate_px_per_s·Δt + gate_r_factor·r. Назначение —
    венгерским алгоритмом; повторяется, пока есть соединения (цепочки).
    ``rows`` изменяется на месте. Возвращает число соединений.
    """
    total = 0
    for _it in range(5):
        ids = [k for k, v in rows.items() if len(v.frames) >= 2]
        if not ids:
            break
        info = {}
        for k in ids:
            fr = np.asarray(rows[k].frames)
            p = zs_all[np.asarray(rows[k].det)]
            info[k] = (fr[0], fr[-1], _end_state(fr, p, t_frames, True, cfg.stitch_fit_points),
                       _end_state(fr, p, t_frames, False, cfg.stitch_fit_points),
                       float(np.median(r_all[np.asarray(rows[k].det)[-5:]])),
                       float(np.median(r_all[np.asarray(rows[k].det)[:5]])))
        starts = sorted(ids, key=lambda k: info[k][0])
        start_f = np.asarray([info[k][0] for k in starts])
        tb_a = np.asarray([info[k][3][0] for k in starts])
        zb_a = np.asarray([info[k][3][1] for k in starts])
        vb_a = np.asarray([info[k][3][2] for k in starts])
        rb_a = np.asarray([info[k][5] for k in starts])
        starts_a = np.asarray(starts)
        sw = np.asarray(switch_times, float)
        pa, pb, cost = [], [], []
        for a in ids:
            fa_end = info[a][1]
            lo = np.searchsorted(start_f, fa_end + 1)
            hi = np.searchsorted(start_f, fa_end + cfg.stitch_max_gap + 1)
            if hi <= lo:
                continue
            ta, za, va = info[a][2]
            ra = info[a][4]
            tb, zb, vb, rb = tb_a[lo:hi], zb_a[lo:hi], vb_a[lo:hi], rb_a[lo:hi]
            dt = (tb - ta)[:, None]
            if len(sw):
                crossed = ((sw[None, :] > ta) & (sw[None, :] <= tb[:, None])).any(axis=1)
            else:
                crossed = np.zeros(len(tb), bool)
            va_ = np.where(crossed[:, None], 0.0, va[None, :])
            vb_ = np.where(crossed[:, None], 0.0, vb)
            e1 = zb - (za[None, :] + va_ * dt)
            e2 = za[None, :] - (zb - vb_ * dt)
            d = 0.5 * (np.hypot(e1[:, 0], e1[:, 1]) + np.hypot(e2[:, 0], e2[:, 1]))
            g = cfg.stitch_gate_px + cfg.stitch_gate_px_per_s * dt[:, 0] \
                + cfg.gate_r_factor * np.maximum(ra, rb)
            ok = d <= g
            if not ok.any():
                continue
            c_ = (d / g) ** 2 + cfg.w_r * np.abs(ra - rb) / np.maximum(0.5 * (ra + rb), 1e-3) \
                + 0.1 * (start_f[lo:hi] - fa_end)
            pa.extend([a] * int(ok.sum()))
            pb.extend(starts_a[lo:hi][ok].tolist())
            cost.extend(c_[ok].tolist())
        if not pa:
            break
        ua, ia = np.unique(pa, return_inverse=True)
        ub, ib = np.unique(pb, return_inverse=True)
        C = np.full((len(ua), len(ub)), BIG)
        np.minimum.at(C, (ia, ib), np.asarray(cost))
        rr, cc = linear_sum_assignment(C)
        ok = C[rr, cc] < BIG
        pairs = [(int(ua[i]), int(ub[j])) for i, j in zip(rr[ok], cc[ok])]
        # цепочки: a→b, b→c… обрабатываем от начала цепочки
        nxt = dict(pairs)
        prv = {b: a for a, b in pairs}
        n = 0
        for a in [a for a in nxt if a not in prv]:
            cur = a
            while cur in nxt:
                b = nxt[cur]
                rows[a].frames.extend(rows[b].frames)
                rows[a].det.extend(rows[b].det)
                rows[a].merge.extend(rows[b].merge)
                del rows[b]
                n += 1
                cur = b
        total += n
        if n == 0:
            break
    return total


def link_detections(det: pd.DataFrame, cal: Calibration, cfg: LinkConfig,
                    switch_times: list[float] | None, t_frames: np.ndarray,
                    frame_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Связать детекции всего ролика в треки.

    ``det`` — таблица детекций; ``t_frames`` — PTS-время каждого кадра видео.
    Возвращает длинную таблицу: одна строка = одна капля в одном кадре, включая
    интерполированные кадры пропусков.
    """
    det = det.reset_index(drop=True)
    z, s = cal.to_zs(det.x.values, det.y.values)
    zs_all = np.column_stack([z, s])
    r_all = det.r.values.astype(float)
    i_all = det.intensity.values.astype(float)
    fr = det.frame.values.astype(int)
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(fr.min(), fr.max() + 2)) if len(fr) else []
    f0 = int(fr.min()) if len(fr) else 0
    f_lo, f_hi = frame_range if frame_range else (f0, int(fr.max()) + 1 if len(fr) else 0)
    lk = Linker(cfg, switch_times)
    for f in range(f_lo, f_hi):
        k = f - f0
        if 0 <= k < len(bounds) - 1:
            idx = order[bounds[k]:bounds[k + 1]]
        else:
            idx = np.zeros(0, int)
        lk.step(f, float(t_frames[f]), idx, zs_all[idx], r_all[idx], i_all[idx])
    lk.finish()
    log.info("сопоставление: %d треков создано, сбросов скорости на переключениях: %d",
             lk.next_id, lk.n_switch_resets)
    n_join = stitch_tracklets(lk.rows, zs_all, r_all, t_frames, cfg, switch_times or [])
    log.info("склейка обрывков: %d соединений", n_join)

    # сборка таблицы с интерполяцией пропусков
    parts = []
    pid = 0
    for tid in sorted(lk.rows):
        tr = lk.rows[tid]
        if len(tr.frames) < cfg.min_track_frames_keep:
            continue
        frames = np.asarray(tr.frames)
        d = np.asarray(tr.det)
        full = np.arange(frames[0], frames[-1] + 1)
        tt = t_frames[full]
        tr_t = t_frames[frames]
        rec = {
            "particle_id": np.full(len(full), pid),
            "frame": full,
            "t_s": tt,
        }
        for col, arr in (("x_px", det.x.values), ("y_px", det.y.values), ("r_px", r_all),
                         ("intensity", i_all)):
            rec[col] = np.interp(tt, tr_t, arr[d])
        real = np.isin(full, frames)
        rec["interpolated"] = (~real).astype(int)
        kind = np.empty(len(full), object)
        kind[:] = ""
        pos = np.searchsorted(full, frames)
        kind[pos] = det.kind.values[d]
        # для интерполированных кадров — тип ближайшей реальной детекции слева
        last = pd.Series(kind).replace("", np.nan).ffill().values
        rec["kind"] = last
        mr = np.zeros(len(full), int)
        mr[pos] = tr.merge
        rec["merge_risk"] = mr
        det_id = np.full(len(full), -1)
        det_id[pos] = d
        rec["det_id"] = det_id
        parts.append(pd.DataFrame(rec))
        pid += 1
    if not parts:
        return pd.DataFrame(columns=["particle_id", "frame", "t_s", "x_px", "y_px", "z_px", "s_px",
                                     "r_px", "intensity", "kind", "interpolated", "merge_risk", "det_id"])
    out = pd.concat(parts, ignore_index=True)
    zz, ss = cal.to_zs(out.x_px.values, out.y_px.values)
    out.insert(5, "z_px", zz)
    out.insert(6, "s_px", ss)
    log.info("сопоставление: %d треков длиной ≥ %d кадров, %d строк",
             pid, cfg.min_track_frames_keep, len(out))
    return out

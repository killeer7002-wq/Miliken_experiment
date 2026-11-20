"""Калибровка по шкале: По углу и шагу линий сетки находим оси (z, s) и масштаб px(мм).

1. Black-hat медианного кадра выделяет тёмные тонкие линии на светлом фоне.
2. Грубый угол находится преобразование Хафа (cv2.HoughLines) по бинаризованному
   black-hat; берётся медиана углов линий, близких к ожидаемому.
3. Точный угол это максимум "резкости" проекции black-hat на нормаль к линиям
   (дисперсия профиля) при переборе угла ±2 с шагом 0.05.
4. Положения линий - пики этого профиля (кластеры ρ); шаг это медиана разностей
   соседних пиков в центральной части поля (перспектива у краёв искажает шаг).

Оси: s: вдоль линий шкалы, z: перпендикулярно (вдоль поля). Начало координат
центр поля зрения. z_sign = +1 z растёт вниз по кадру; окуляр
переворачивает изображение, тогда +z это «вверх» в реальной камере, а
оседание под тяжестью даёт v_z < 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from .config import CalibrationConfig
from .roi import Roi

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    angle_deg: float            # угол линий шкалы к горизонтали (по часовой в координатах изображения, y вниз)
    step_px: float              # шаг линий
    step_spread_px: tuple[float, float]  # 10-й и 90-й перцентили шагов
    x0: float                   # начало координат (центр поля)
    y0: float
    z_sign: int
    mm_per_px: float | None
    line_rho: np.ndarray        # положения линий вдоль нормали (px, от начала координат)
    source: str                 # auto | config
    # местный шаг сетки (перспектива, если камера наклонена): step(ρ) = step_a + step_b * ρ,
    # ρ — координата вдоль нормали к линиям (px, от центра поля); step_b = 0 — без перспективы
    step_a: float = 0.0
    step_b: float = 0.0
    n_lines_used: int = 0

    def local_step(self, z: np.ndarray | float) -> np.ndarray:
        """Шаг сетки (px на деление) в точке с координатой z (px)."""
        a = self.step_a or self.step_px
        rho = self.z_sign * np.asarray(z, float)
        return np.maximum(a + self.step_b * rho, 0.3 * a)

    def to_div(self, z: np.ndarray) -> np.ndarray:
        """Координата z в делениях шкалы, от центра поля, с учётом местного шага.

        z_div = интеграл dρ / step(ρ): при линейном step(ρ) = a + bρ это (1/b)·ln(1 + bρ/a).
        """
        a = self.step_a or self.step_px
        rho = self.z_sign * np.asarray(z, float)
        b = self.step_b
        if abs(b) < 1e-9:
            return self.z_sign * rho / a
        return self.z_sign * np.log(np.maximum(1 + b * rho / a, 1e-6)) / b

    def to_zs(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th = np.deg2rad(self.angle_deg)
        dx, dy = np.asarray(x) - self.x0, np.asarray(y) - self.y0
        s = dx * np.cos(th) + dy * np.sin(th)
        z = self.z_sign * (-dx * np.sin(th) + dy * np.cos(th))
        return z, s

    def to_xy(self, z: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th = np.deg2rad(self.angle_deg)
        zz = self.z_sign * np.asarray(z)
        x = self.x0 + s * np.cos(th) - zz * np.sin(th)
        y = self.y0 + s * np.sin(th) + zz * np.cos(th)
        return x, y

    def mm(self, px: np.ndarray) -> np.ndarray:
        if self.mm_per_px is None:
            return np.full(np.shape(px), np.nan)
        return np.asarray(px) * self.mm_per_px

    def summary(self) -> str:
        sc = f"{self.mm_per_px * 1000:.3f} мкм/px" if self.mm_per_px else "масштаб не задан"
        persp = ""
        if self.step_b:
            r = 0.8 * (np.max(np.abs(self.line_rho)) if len(self.line_rho) else 250)
            lo, hi = sorted([self.step_a - self.step_b * r, self.step_a + self.step_b * r])
            persp = f", перспектива: шаг {lo:.1f}–{hi:.1f} px поперёк поля"
        return (f"угол сетки {self.angle_deg:.2f}°, шаг {self.step_px:.1f} px "
                f"(10–90%: {self.step_spread_px[0]:.1f}–{self.step_spread_px[1]:.1f}){persp}, {sc}")


def _profile(img: np.ndarray, mask: np.ndarray, angle_deg: float, x0: float, y0: float):
    """Проекция img на нормаль к линиям под углом angle_deg: (rho_bins, mean)."""
    th = np.deg2rad(angle_deg)
    ys, xs = np.nonzero(mask)
    rho = -(xs - x0) * np.sin(th) + (ys - y0) * np.cos(th)
    lo = np.floor(rho.min())
    idx = (rho - lo).astype(int)
    n = idx.max() + 1
    s = np.bincount(idx, weights=img[ys, xs], minlength=n)
    c = np.bincount(idx, minlength=n)
    prof = np.where(c > 20, s / np.maximum(c, 1), 0.0)
    return lo + np.arange(n), prof


def _period_autocorr(prof: np.ndarray, min_lag: int = 6) -> float | None:
    """Период профиля по автокорреляции (не зависит от ожидаемого шага сетки)."""
    x = prof - np.mean(prof)
    n = len(x)
    if n < 4 * min_lag:
        return None
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    pk, props = find_peaks(ac[: n // 2], height=0.1, prominence=0.05)
    pk = pk[pk >= min_lag]
    if len(pk) == 0:
        return None
    h = ac[pk]
    first = pk[np.argmax(h >= 0.6 * h.max())]
    if 1 <= first < len(ac) - 1:
        y0_, y1_, y2_ = ac[first - 1], ac[first], ac[first + 1]
        den = y0_ - 2 * y1_ + y2_
        if abs(den) > 1e-12:
            return float(first + 0.5 * (y0_ - y2_) / den)
    return float(first)


def clean_lines(line_rho: np.ndarray, step: float, tol: float = 0.25) -> np.ndarray:
    """Цепочка линий с шагом = step от центральной в обе стороны.
    """
    r = np.sort(np.asarray(line_rho, float))
    if len(r) < 2:
        return r
    c = int(np.argmin(np.abs(r)))
    chain = [r[c]]
    for direction in (1, -1):
        cur, cur_step = r[c], step
        while True:
            target = cur + direction * cur_step
            cand = r[(np.abs(r - target) < tol * cur_step)]
            if len(cand) == 0:
                target2 = cur + 2 * direction * cur_step
                cand = r[(np.abs(r - target2) < tol * cur_step)]
                if len(cand) == 0:
                    break
            nxt = cand[np.argmin(np.abs(cand - target))]
            n_steps = max(1, round(abs(nxt - cur) / cur_step))
            cur_step = 0.7 * cur_step + 0.3 * abs(nxt - cur) / n_steps   # шаг может плавно меняться
            chain.append(nxt)
            cur = nxt
    return np.sort(np.asarray(chain))


def local_step_model(line_rho: np.ndarray, step: float, radius: float, enabled: bool) -> tuple[float, float, int]:
    """Линейная модель шага поперёк поля step(ρ) = a + bρ по интервалам между соседними линиями.

    Возвращает (a, b, число линий). Если линий мало или наклон незначим b = 0.
    """
    lines = clean_lines(line_rho, step)
    if len(lines) < 2:
        return step, 0.0, len(lines)
    d = np.diff(lines)
    n = np.maximum(np.round(d / step), 1)
    sp = d / n
    mid = 0.5 * (lines[1:] + lines[:-1])
    if not enabled or len(sp) < 4:
        return float(np.median(sp)), 0.0, len(lines)
    b, a = np.polyfit(mid, sp, 1)
    resid = sp - (a + b * mid)
    se_b = np.sqrt(np.sum(resid**2) / max(len(sp) - 2, 1) / max(np.sum((mid - mid.mean()) ** 2), 1e-9))

    if abs(b) < 3 * se_b or abs(b) * radius < 0.01 * a:
        return float(np.median(sp)), 0.0, len(lines)
    if abs(b) * radius > 0.4 * a:
        log.warning("перспектива: шаг сетки меняется поперёк поля слишком сильно (b = %.4f) — проверьте калибровку", b)
    return float(a), float(b), len(lines)


def calibrate(roi: Roi, cfg: CalibrationConfig) -> Calibration:
    med = roi.median_frame.astype(np.float32)
    mask = roi.mask.copy()
    x0, y0 = roi.cx, roi.cy
    # black-hat: тёмные линии толщиной до ~5 px
    bh = cv2.morphologyEx(med, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    bh[~mask] = 0
    # нормировка на локальную яркость (виньетка ослабляет линии у края)
    loc = cv2.GaussianBlur(med, (0, 0), 25) + 1
    bhn = bh / loc
    bhn[~mask] = 0

    source = "auto"
    if cfg.grid_angle_deg == "auto":
        sub = mask.copy()
        sub[::2, :] = False
        coarse_grid = np.arange(-90.0, 90.0, 1.0)
        sharp = [np.var(_profile(bhn, sub, a_, x0, y0)[1]) for a_ in coarse_grid]
        coarse = float(coarse_grid[int(np.argmax(sharp))])
        grid = np.arange(coarse - 1.5, coarse + 1.5001, 0.05)
        sharp = [np.var(_profile(bhn, mask, a_, x0, y0)[1]) for a_ in grid]
        angle = float(grid[int(np.argmax(sharp))])
        angle = (angle + 90.0) % 180.0 - 90.0
    else:
        angle = float(cfg.grid_angle_deg)
        source = "config"

    rho, prof = _profile(bhn, mask, angle, x0, y0)
    prof_s = np.convolve(prof, np.ones(3) / 3, mode="same")
    step_est = _period_autocorr(prof_s)
    if step_est is None:
        log.warning("период сетки по автокорреляции не найден; беру ожидаемый шаг %.1f px", cfg.expected_step_px)
        step_est = cfg.expected_step_px
    pk, _ = find_peaks(prof_s, distance=max(4, int(0.6 * step_est)),
                       prominence=0.25 * np.std(prof_s))
    line_rho = rho[pk].astype(float)
    if cfg.grid_step_px == "auto":
        if len(line_rho) >= 3:
            d = np.diff(line_rho)
            mid = 0.5 * (line_rho[1:] + line_rho[:-1])
            central = np.abs(mid) < 0.6 * roi.radius
            dd = d[central] if central.sum() >= 2 else d
            # отбросить пропущенные линии (двойной шаг)
            m0 = np.median(dd)
            dd = dd[np.abs(dd - m0) < 0.35 * m0]
            step = float(np.median(dd))
            spread = (float(np.percentile(dd, 10)), float(np.percentile(dd, 90)))
        else:
            log.warning("найдено меньше 3 линий шкалы; беру ожидаемый шаг %.1f px", cfg.expected_step_px)
            step = cfg.expected_step_px
            spread = (step, step)
    else:
        step = float(cfg.grid_step_px)
        spread = (step, step)
        source = "config" if source == "config" else "mixed"

    if abs(angle - cfg.expected_angle_deg) > cfg.angle_tolerance_deg:
        log.info("угол сетки %.2f° отличается от ожидаемого %.1f° (нормально, если камера повёрнута)",
                 angle, cfg.expected_angle_deg)
    if abs(step - cfg.expected_step_px) > cfg.step_tolerance_px:
        log.info("шаг сетки %.1f px отличается от ожидаемого %.1f px (нормально, если камера ближе/дальше)",
                 step, cfg.expected_step_px)
    mm_per_px = None
    if cfg.mm_per_division:
        mm_per_px = float(cfg.mm_per_division) / step
    else:
        log.warning("calibration.mm_per_division не задан: в CSV будут только пиксели (мм = NaN)")
    a, b, n_used = local_step_model(line_rho, step, roi.radius, cfg.perspective)
    cal = Calibration(angle, step, spread, x0, y0, cfg.z_sign, mm_per_px, line_rho, source, a, b, n_used)
    log.info("калибровка: %s", cal.summary())
    return cal

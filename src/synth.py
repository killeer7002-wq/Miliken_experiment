"""Генератор синтетического видео с известными треками (ground truth).

Картинка похожа на IMG_6823: вертикальный кадр 720×1280, светлый зелёный круг
поля зрения с виньеткой и туманными пятнами, наклонная шкала (12°, шаг 51 px)
плюс почти вертикальная линия, капли-точки и капли-кольца (радиус медленно
меняется — уход из фокуса), броуновское блуждание, переключения поля и
случайные перезарядки. Кодек — mp4v (артефакты сжатия есть, как у телефона).

Ground truth:
* ``<OUT>_gt.csv``: drop_id, frame, t, x, y, z, s, r, kind, visible, field_state,
  charge_epoch;
* ``<OUT>_events.json``: switches_s, recharges [{drop_id, t, v_before, v_after}],
  параметры генерации.
Координаты (z, s) — в той же системе, что и :class:`oiltrack.calibration.Calibration`
(угол 12°, начало — центр поля, z растёт вниз по кадру).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class SynthConfig:
    seed: int = 0
    width: int = 720
    height: int = 1280
    fps: float = 30.0
    duration_s: float = 30.0
    cx: float = 380.0
    cy: float = 540.0
    radius: float = 325.0
    grid_angle_deg: float = 12.0
    grid_step_px: float = 51.0
    n_drops: int = 150
    ring_fraction: float = 0.35
    v_gravity_px_s: tuple[float, float] = (-2.5, 0.8)    # среднее, разброс (z растёт вниз по кадру)
    charged_fraction: float = 0.4
    v_field_px_s: tuple[float, float] = (6.0, 22.0)      # |v_E| диапазон у заряженных
    recharge_fraction: float = 0.25                       # доля заряженных, у которых будет перезарядка
    min_recharge_dv_px_s: float = 6.0
    diffusion_px2_s: float = 2.0
    noise_sigma: float = 2.0
    switches_s: list[float] = field(default_factory=lambda: [10.0, 20.0])
    background_level: float = 120.0


def _render_spot(img: np.ndarray, x: float, y: float, sigma: float, amp: float) -> None:
    h, w = img.shape
    rad = int(np.ceil(4 * sigma)) + 1
    x0, x1 = max(0, int(x) - rad), min(w, int(x) + rad + 2)
    y0, y1 = max(0, int(y) - rad), min(h, int(y) + rad + 2)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def _render_ring(img: np.ndarray, x: float, y: float, r: float, width: float, amp: float,
                 dot_amp: float, fill: float) -> None:
    h, w = img.shape
    rad = int(np.ceil(r + 4 * width)) + 1
    x0, x1 = max(0, int(x) - rad), min(w, int(x) + rad + 2)
    y0, y1 = max(0, int(y) - rad), min(h, int(y) + rad + 2)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    patch = amp * np.exp(-((d - r) ** 2) / (2 * width**2))
    patch += fill * amp / (1 + np.exp((d - r) / 0.8))         # слабо светлый диск внутри
    if dot_amp > 0:
        patch += dot_amp * np.exp(-(d**2) / (2 * 1.0**2))
    img[y0:y1, x0:x1] += patch


def static_background(cfg: SynthConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(фон без линий, множитель линий шкалы, маска поля)."""
    h, w = cfg.height, cfg.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rho = np.hypot(xx - cfg.cx, yy - cfg.cy) / cfg.radius
    mask = rho <= 1.0
    vign = np.clip(1 - 0.55 * rho**2, 0, 1)
    bg = cfg.background_level * vign
    for _ in range(8):  # туманные пятна
        hx, hy = rng.uniform(cfg.cx - cfg.radius, cfg.cx + cfg.radius), rng.uniform(cfg.cy - cfg.radius, cfg.cy + cfg.radius)
        s = rng.uniform(25, 70)
        bg += rng.uniform(-8, 12) * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * s * s))
    th = np.deg2rad(cfg.grid_angle_deg)
    nrm = -(xx - cfg.cx) * np.sin(th) + (yy - cfg.cy) * np.cos(th)
    phase = (nrm / cfg.grid_step_px) - np.round(nrm / cfg.grid_step_px)
    dline = np.abs(phase) * cfg.grid_step_px
    lines = 1 - 0.45 * np.exp(-(dline**2) / (2 * 0.9**2))
    # почти вертикальная линия с рисками
    vx = cfg.cx - 40 + (yy - cfg.cy) * np.tan(np.deg2rad(-6))
    lines *= 1 - 0.45 * np.exp(-((xx - vx) ** 2) / (2 * 0.9**2))
    return bg.astype(np.float32), lines.astype(np.float32), mask


def generate(out: str | Path, cfg: SynthConfig | None = None) -> dict[str, str]:
    """Сгенерировать видео и ground truth. Возвращает пути к файлам."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_fr = int(round(cfg.duration_s * cfg.fps))
    dt = 1.0 / cfg.fps
    bg, lines, mask = static_background(cfg, rng)
    th = np.deg2rad(cfg.grid_angle_deg)

    N = cfg.n_drops
    # начальные позиции равномерно в круге (в системе z, s)
    rr = cfg.radius * 0.95 * np.sqrt(rng.uniform(0, 1, N))
    ang = rng.uniform(0, 2 * np.pi, N)
    z = rr * np.sin(ang)
    s = rr * np.cos(ang)
    v_g = rng.normal(cfg.v_gravity_px_s[0], cfg.v_gravity_px_s[1], N)
    charged = rng.uniform(0, 1, N) < cfg.charged_fraction
    v_e = np.where(charged, rng.uniform(*cfg.v_field_px_s, N) * rng.choice([-1, 1], N), 0.0)
    is_ring = rng.uniform(0, 1, N) < cfg.ring_fraction
    r0 = np.where(is_ring, rng.uniform(5, 20, N), rng.uniform(1.2, 2.0, N))
    amp = np.where(is_ring, rng.uniform(30, 55, N), rng.uniform(45, 100, N))
    dot = np.where(is_ring & (rng.uniform(0, 1, N) < 0.5), rng.uniform(25, 60, N), 0.0)
    focus_w = rng.uniform(0.05, 0.3, N)
    focus_ph = rng.uniform(0, 2 * np.pi, N)
    focus_a = np.where(is_ring, rng.uniform(0.0, 0.25, N), 0.0)

    # перезарядки: не ближе 1.5 с к переключениям и к краям
    sw = np.asarray(cfg.switches_s, float)
    recharges = []
    rech_t = np.full(N, np.inf)
    rech_dv = np.zeros(N)
    for i in np.nonzero(charged)[0]:
        if rng.uniform() >= cfg.recharge_fraction:
            continue
        for _ in range(20):
            t_r = rng.uniform(2.0, cfg.duration_s - 2.0)
            if len(sw) == 0 or np.min(np.abs(sw - t_r)) > 1.5:
                break
        else:
            continue
        dv = rng.uniform(cfg.min_recharge_dv_px_s, 15.0) * rng.choice([-1, 1])
        rech_t[i] = t_r
        rech_dv[i] = dv
    rows = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, cfg.fps, (cfg.width, cfg.height))
    sqrt2D = np.sqrt(2 * cfg.diffusion_px2_s * dt)
    epoch = np.zeros(N, int)
    for f in range(n_fr):
        t = f * dt
        state = int(np.searchsorted(sw, t, side="right"))
        sign = 1.0 if state % 2 == 0 else -1.0
        # перезарядка
        hit = (rech_t <= t) & (epoch == 0)
        for i in np.nonzero(hit)[0]:
            recharges.append({"drop_id": int(i), "t": float(rech_t[i]),
                              "v_e_before": float(v_e[i]), "v_e_after": float(v_e[i] + rech_dv[i])})
            v_e[i] += rech_dv[i]
            epoch[i] = 1
        if f > 0:
            v = v_g + sign * v_e
            z = z + v * dt + rng.normal(0, sqrt2D, N)
            s = s + rng.normal(0, sqrt2D, N)
        x = cfg.cx + s * np.cos(th) - z * np.sin(th)
        y = cfg.cy + s * np.sin(th) + z * np.cos(th)
        r = r0 * (1 + focus_a * np.sin(2 * np.pi * focus_w * t + focus_ph))
        img = np.zeros((cfg.height, cfg.width), np.float32)
        vis = np.hypot(x - cfg.cx, y - cfg.cy) < cfg.radius - 3
        for i in np.nonzero(vis)[0]:
            if is_ring[i]:
                _render_ring(img, x[i], y[i], r[i], 1.0, amp[i], dot[i], 0.25)
            else:
                _render_spot(img, x[i], y[i], r[i], amp[i])
        frame = (bg + img) * lines
        frame += rng.normal(0, cfg.noise_sigma, frame.shape).astype(np.float32)
        frame[~mask] = rng.normal(2, 1, (~mask).sum())
        g = np.clip(frame, 0, 255).astype(np.uint8)
        bgr = np.zeros((cfg.height, cfg.width, 3), np.uint8)
        bgr[:, :, 1] = g
        bgr[:, :, 2] = (g * 0.05).astype(np.uint8)
        writer.write(bgr)
        st = "A" if state % 2 == 0 else "B"
        rows.append(pd.DataFrame({
            "drop_id": np.arange(N), "frame": f, "t": t, "x": x, "y": y, "z": z, "s": s, "r": r,
            "kind": np.where(is_ring, "ring", "point"), "visible": vis.astype(int),
            "field_state": st, "charge_epoch": epoch.copy(),
        }))
    writer.release()
    gt = pd.concat(rows, ignore_index=True)
    gt_path = out.with_name(out.stem + "_gt.csv")
    gt.to_csv(gt_path, index=False, float_format="%.4f")
    ev = {"switches_s": [float(x) for x in sw], "recharges": recharges,
          "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}}
    ev_path = out.with_name(out.stem + "_events.json")
    ev_path.write_text(json.dumps(ev, ensure_ascii=False, indent=2))
    return {"video": str(out), "gt": str(gt_path), "events": str(ev_path)}


# ---------------------------------------------------------------- для тестов без видео

def simulate_detections(n_drops: int = 150, duration_s: float = 30.0, fps: float = 30.0,
                        switches_s: list[float] | None = None, seed: int = 0, loc_sigma: float = 0.25,
                        miss_prob: float = 0.03, recharge_fraction: float = 0.25,
                        clutter_per_frame: float = 3.0, diffusion: float = 2.0,
                        radius: float = 325.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Быстрая синтетика на уровне детекций (без рендера): для тестов трекера и нарезки.

    Возвращает (det, gt, events): det — как у detect_video (+ столбец gt_id),
    gt — истинные позиции, events — переключения и перезарядки.
    """
    switches_s = [10.0, 20.0] if switches_s is None else switches_s
    cfg = SynthConfig(seed=seed, n_drops=n_drops, duration_s=duration_s, fps=fps,
                      switches_s=switches_s, recharge_fraction=recharge_fraction,
                      diffusion_px2_s=diffusion, radius=radius)
    rng = np.random.default_rng(seed)
    N, n_fr, dt = n_drops, int(round(duration_s * fps)), 1.0 / fps
    th = np.deg2rad(cfg.grid_angle_deg)
    rr = radius * 0.9 * np.sqrt(rng.uniform(0, 1, N))
    ang = rng.uniform(0, 2 * np.pi, N)
    z, s = rr * np.sin(ang), rr * np.cos(ang)
    v_g = rng.normal(cfg.v_gravity_px_s[0], cfg.v_gravity_px_s[1], N)
    charged = rng.uniform(0, 1, N) < cfg.charged_fraction
    v_e = np.where(charged, rng.uniform(*cfg.v_field_px_s, N) * rng.choice([-1, 1], N), 0.0)
    is_ring = rng.uniform(0, 1, N) < cfg.ring_fraction
    r0 = np.where(is_ring, rng.uniform(5, 15, N), rng.uniform(1.7, 3.0, N))
    inten = np.where(is_ring, rng.uniform(20, 50, N), rng.uniform(30, 100, N))
    sw = np.asarray(switches_s, float)
    rech_t = np.full(N, np.inf)
    rech_dv = np.zeros(N)
    for i in np.nonzero(charged)[0]:
        if rng.uniform() >= recharge_fraction:
            continue
        for _ in range(50):
            t_r = rng.uniform(2.5, duration_s - 2.5)
            if len(sw) == 0 or np.min(np.abs(sw - t_r)) > 2.0:
                rech_t[i] = t_r
                rech_dv[i] = rng.uniform(cfg.min_recharge_dv_px_s, 15.0) * rng.choice([-1, 1])
                break
    recharges = []
    det_rows, gt_rows = [], []
    epoch = np.zeros(N, int)
    sq = np.sqrt(2 * diffusion * dt)
    for f in range(n_fr):
        t = f * dt
        state = int(np.searchsorted(sw, t, side="right"))
        sign = 1.0 if state % 2 == 0 else -1.0
        for i in np.nonzero((rech_t <= t) & (epoch == 0))[0]:
            recharges.append({"drop_id": int(i), "t": float(rech_t[i]),
                              "v_e_before": float(v_e[i]), "v_e_after": float(v_e[i] + rech_dv[i])})
            v_e[i] += rech_dv[i]
            epoch[i] = 1
        if f > 0:
            z = z + (v_g + sign * v_e) * dt + rng.normal(0, sq, N)
            s = s + rng.normal(0, sq, N)
        x = cfg.cx + s * np.cos(th) - z * np.sin(th)
        y = cfg.cy + s * np.sin(th) + z * np.cos(th)
        vis = np.hypot(x - cfg.cx, y - cfg.cy) < radius - 5
        seen = vis & (rng.uniform(0, 1, N) > miss_prob)
        gt_rows.append(pd.DataFrame({"drop_id": np.arange(N), "frame": f, "t": t, "x": x, "y": y,
                                     "z": z, "s": s, "visible": vis.astype(int),
                                     "field_state": "A" if state % 2 == 0 else "B",
                                     "charge_epoch": epoch.copy()}))
        k = np.nonzero(seen)[0]
        sig = loc_sigma * np.maximum(1, r0[k] / 3)
        dx = rng.normal(0, 1, len(k)) * sig
        dy = rng.normal(0, 1, len(k)) * sig
        nc = rng.poisson(clutter_per_frame)
        ca = rng.uniform(0, 2 * np.pi, nc)
        cr = radius * 0.95 * np.sqrt(rng.uniform(0, 1, nc))
        det_rows.append(pd.DataFrame({
            "frame": f, "t": t,
            "x": np.concatenate([x[k] + dx, cfg.cx + cr * np.cos(ca)]),
            "y": np.concatenate([y[k] + dy, cfg.cy + cr * np.sin(ca)]),
            "r": np.concatenate([r0[k] * rng.uniform(0.9, 1.1, len(k)), np.full(nc, 1.7)]),
            "intensity": np.concatenate([inten[k] * rng.uniform(0.8, 1.2, len(k)), rng.uniform(5, 15, nc)]),
            "kind": np.concatenate([np.where(is_ring[k], "ring", "point"), np.full(nc, "point")]),
            "score": 5.0, "coverage": 0.0,
            "gt_id": np.concatenate([k, np.full(nc, -1)]),
        }))
    det = pd.concat(det_rows, ignore_index=True)
    gt = pd.concat(gt_rows, ignore_index=True)
    events = {"switches_s": list(map(float, sw)), "recharges": recharges,
              "t_frames": (np.arange(n_fr) * dt).tolist(), "cx": cfg.cx, "cy": cfg.cy,
              "radius": radius, "angle": cfg.grid_angle_deg}
    return det, gt, events

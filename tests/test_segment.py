"""Нарезка на синтетике: перезарядки найдены, ложных мало, момент точен (медиана < 0.3 с).

Точность момента физически ограничена броуновским шумом: излом скорости Δv
виден, когда Δv·δ²/2 превышает блуждание √(2Dδ), поэтому при Δv ≈ 10 px/с
и D ≈ 2 px²/с отдельные моменты уходят на 0.5–0.8 с. Поэтому излом
засчитывается в окне ±1 с, а требование 0.3 с проверяется по медиане.

Обнаружимость перезарядки зависит от длины кусков по обе стороны и от
диффузии: σ_v² = 12D/(5T) + 12σ²_loc/(N T²). Тест учитывает только
перезарядки, которые в принципе различимы: |Δv| ≥ 6·√(σ²_a + σ²_b), где
куски — от предыдущего до следующего события (переключение, край трека).
"""

from __future__ import annotations

import numpy as np

from helpers import field_switches_from, linked, majority_gt
from src.config import Config
from src.segment import segment_tracks, velocity_error

MATCH_S = 1.0     # излом засчитывается, если он ближе 1 с к истинной перезарядке


def _run(seed: int = 1):
    tracks, det, gt, ev, cal = linked(seed, 30.0, 150, 0.6)
    cfg = Config()
    cfg.segment.mode = "split"
    fs = field_switches_from(ev)
    t = np.asarray(ev["t_frames"])
    tr2, seg = segment_tracks(tracks, fs, cal, cfg.segment, float(t[-1]), 10 / 30, set())
    return tr2, seg, ev, cfg


def test_recharge_detection():
    tr2, seg, ev, cfg = _run(1)
    sw = np.asarray(ev["switches_s"])
    D, sl = 2.0, 0.25
    # треки → основная капля
    owner = {pid: majority_gt(g) for pid, g in tr2[tr2.interpolated == 0].groupby("particle_id")}
    span = tr2.groupby("particle_id").t_s.agg(["min", "max"])
    found = eligible = 0
    errs = []
    used_breaks = set()
    breaks = seg[seg.end_reason == "charge_change"]
    for rc in ev["recharges"]:
        i, t_r = rc["drop_id"], rc["t"]
        dv = abs(rc["v_e_after"] - rc["v_e_before"])
        pids = [p for p, o in owner.items() if o == i and span.loc[p, "min"] < t_r - 0.5 < t_r + 0.5 < span.loc[p, "max"]]
        if not pids:
            continue
        p = pids[0]
        lo = max([span.loc[p, "min"]] + [x for x in sw if x < t_r])
        hi = min([span.loc[p, "max"]] + [x for x in sw if x > t_r])
        Ta, Tb = t_r - lo, hi - t_r
        if Ta < cfg.segment.min_piece_s or Tb < cfg.segment.min_piece_s:
            continue
        r_med = float(tr2[tr2.particle_id == p].r_px.median())
        sl_i = sl * max(1.0, r_med / 3)          # как в simulate_detections
        sa = velocity_error(D, sl_i, int(Ta * 30), Ta)
        sb = velocity_error(D, sl_i, int(Tb * 30), Tb)
        if dv < 6 * np.hypot(sa, sb):
            continue
        eligible += 1
        b = breaks[breaks.particle_id == p]
        if len(b):
            k = int(np.argmin(np.abs(b.t_end.values - t_r)))
            if abs(b.t_end.values[k] - t_r) < MATCH_S:
                found += 1
                errs.append(abs(b.t_end.values[k] - t_r))
                used_breaks.add((p, int(b.segment_id.values[k])))
    # «ложные»: изломы дальше MATCH_S от истинной перезарядки этой капли. Доля
    # считается от числа кусков трека ≥ 2 с (как в нулевом тесте): доля от
    # числа изломов зависела бы от того, сколько перезарядок заложено в синтетику
    rech = {(r["drop_id"]): r["t"] for r in ev["recharges"]}
    false = 0
    n_breaks = 0
    for row in breaks.itertuples(index=False):
        o = owner.get(row.particle_id, -1)
        if o < 0:
            continue
        n_breaks += 1
        if o not in rech or abs(rech[o] - row.t_end) > MATCH_S:
            false += 1
    real = tr2[tr2.interpolated == 0]
    iv = np.searchsorted(sw, real.t_s.values, side="right")
    pieces = sum(1 for _, g in real.assign(iv=iv).groupby(["particle_id", "iv"])
                 if g.t_s.max() - g.t_s.min() >= 2.0)
    rec = found / max(1, eligible)
    fr = false / max(1, pieces)
    print(f"  различимых перезарядок: {eligible}, найдено: {found} ({100 * rec:.1f}%), "
          f"изломов всего: {n_breaks}, не у истинной перезарядки: {false} "
          f"({100 * fr:.1f}% от {pieces} кусков ≥2 с; точность {100 * (1 - false / max(1, n_breaks)):.0f}%), "
          f"ошибка момента: медиана {np.median(errs) if errs else np.nan:.3f} с")
    assert eligible >= 10
    assert rec > 0.90
    assert fr < 0.05
    assert np.median(errs) < 0.3


def test_no_false_breaks_without_recharges():
    """Нулевая гипотеза: перезарядок нет — изломов почти нет (< 5% кусков ≥ 2 с)."""
    tracks, det, gt, ev, cal = linked(2, 30.0, 150, 0.0)
    cfg = Config()
    cfg.segment.mode = "split"
    fs = field_switches_from(ev)
    t = np.asarray(ev["t_frames"])
    tr2, seg = segment_tracks(tracks, fs, cal, cfg.segment, float(t[-1]), 10 / 30, set())
    real = tr2[tr2.interpolated == 0]
    sw = np.asarray(ev["switches_s"])
    iv = np.searchsorted(sw, real.t_s.values, side="right")
    pieces = sum(1 for _, g in real.assign(iv=iv).groupby(["particle_id", "iv"])
                 if g.t_s.max() - g.t_s.min() >= 2.0)
    nb = int((seg.end_reason == "charge_change").sum())
    print(f"  кусков ≥2 с: {pieces}, ложных изломов: {nb} ({100 * nb / max(1, pieces):.2f}%)")
    assert nb / max(1, pieces) < 0.05


def test_segments_cut_at_switches():
    tr2, seg, ev, cfg = _run(1)
    sw = np.asarray(ev["switches_s"])
    # ни один участок не пересекает переключение поля
    for row in seg.itertuples(index=False):
        inside = (sw > row.t_start + 1e-6) & (sw < row.t_end - 1e-6)
        assert not inside.any(), f"участок {row.particle_id}.{row.segment_id} пересекает переключение"


def test_velocity_error_formula():
    """σ_v из формулы совпадает с разбросом МНК-наклона на чистой броуновской синтетике."""
    rng = np.random.default_rng(5)
    D, sl, T, fps = 2.0, 0.3, 3.0, 30
    n = int(T * fps)
    t = np.arange(n) / fps
    vs = []
    for _ in range(3000):
        z = np.cumsum(rng.normal(0, np.sqrt(2 * D / fps), n)) + rng.normal(0, sl, n)
        vs.append(np.polyfit(t, z, 1)[0])
    emp = np.std(vs)
    th = velocity_error(D, sl, n, t[-1])
    print(f"  σ_v эмпирически {emp:.4f}, по формуле {th:.4f}")
    assert abs(emp / th - 1) < 0.08

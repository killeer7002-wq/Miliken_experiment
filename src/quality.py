"""Флаги качества. Ничего не удаляется: флаги пишутся в CSV, отбор — при сводке.

Флаги строки трека (``flags``, через «|»):
* ``short``     — весь трек короче ``min_track_s``;
* ``defocus``   — сглаженный радиус (скользящая медиана 15 кадров) вырос
                  больше чем в ``max_radius_growth`` раз и больше чем на
                  ``defocus_min_px`` относительно начального (медиана первой
                  1 с / 10 % трека);
* ``edge``      — центр ближе ``edge_factor·r`` к краю маски поля зрения;
* ``merge_risk``— другая капля ближе r1 + r2;
* ``interpolated`` — кадр пропуска, координата интерполирована;
* ``after_charge_change`` — строка после первой перезарядки (режим stop).

Флаги участка (``quality`` в segments.csv): ``short`` (трек короче
min_track_s), ``short_segment`` (участок короче ``min_segment_s``) и флаги
строк defocus / edge / merge_risk, если ими помечено больше
``flag_fraction`` строк участка (одно касание соседа не портит участок);
если флагов нет — ``ok``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QualityConfig
from .roi import Roi


def track_flags(tracks: pd.DataFrame, roi: Roi, cfg: QualityConfig) -> pd.DataFrame:
    tr = tracks
    g = tr.groupby("particle_id")
    dur = g.t_s.transform(lambda x: x.max() - x.min())
    short = dur < cfg.min_track_s

    def r0(df: pd.DataFrame) -> float:
        t0 = df.t_s.min()
        head = df[(df.t_s <= t0 + max(1.0, 0.1 * (df.t_s.max() - t0))) & (df.interpolated == 0)]
        return float(head.r_px.median()) if len(head) else float(df.r_px.median())

    r_start = tr.particle_id.map({pid: r0(df) for pid, df in g})
    # сглаженный радиус: у точек r квантован масштабами LoG и «прыгает» 1.7↔2.5↔3.5
    r_s = g.r_px.transform(lambda v: v.rolling(15, center=True, min_periods=1).median())
    defocus = (r_s > cfg.max_radius_growth * r_start) & (r_s - r_start > cfg.defocus_min_px)
    edge = roi.dist_at(tr.x_px.values, tr.y_px.values) < cfg.edge_factor * tr.r_px.values
    flags = []
    names = [("short", short.values), ("defocus", defocus.values), ("edge", edge),
             ("merge_risk", tr.merge_risk.values.astype(bool)),
             ("interpolated", tr.interpolated.values.astype(bool))]
    if "after_charge_change" in tr:
        names.append(("after_charge_change", tr.after_charge_change.values.astype(bool)))
    arr = np.column_stack([v for _, v in names])
    labels = [n for n, _ in names]
    for row in arr:
        flags.append("|".join(l for l, b in zip(labels, row) if b))
    out = tr.copy()
    out["flags"] = flags
    out["edge"] = edge.astype(int)
    out["defocus"] = defocus.values.astype(int)
    return out


def edge_ending_tracks(tracks: pd.DataFrame, roi: Roi, cfg: QualityConfig) -> set[int]:
    """Треки, последняя реальная детекция которых у края маски (end_reason = edge)."""
    last = tracks[tracks.interpolated == 0].sort_values("t_s").groupby("particle_id").tail(1)
    d = roi.dist_at(last.x_px.values, last.y_px.values)
    return set(last.particle_id.values[d < np.maximum(cfg.edge_factor * last.r_px.values, 6.0)].tolist())


def segment_quality(seg: pd.DataFrame, tracks: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    if len(seg) == 0:
        seg = seg.copy()
        seg["quality"] = []
        return seg
    t = tracks[tracks.segment_id >= 0]
    agg = {}
    for (pid, sid), df in t.groupby(["particle_id", "segment_id"]):
        fl = set()
        f = df["flags"].astype(str)
        if f.str.contains("short").any():
            fl.add("short")
        for name in ("defocus", "edge", "merge_risk"):
            if f.str.contains(name).mean() > cfg.flag_fraction:
                fl.add(name)
        agg[(pid, sid)] = fl
    q = []
    for row in seg.itertuples(index=False):
        fl = set(agg.get((row.particle_id, row.segment_id), set()))
        if row.t_end - row.t_start < cfg.min_segment_s:
            fl.add("short_segment")
        q.append("|".join(sorted(fl)) if fl else "ok")
    seg = seg.copy()
    seg["quality"] = q
    return seg

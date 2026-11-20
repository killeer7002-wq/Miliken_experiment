"""Сопоставление на синтетике: нет перепутываний ID при переключениях поля."""

from __future__ import annotations

import numpy as np

from helpers import linked, majority_gt


def test_no_id_swaps_at_field_switch():
    tracks, det, gt, ev, cal = linked(0)
    swaps = spanning = 0
    for s0 in ev["switches_s"]:
        for pid, g in tracks.groupby("particle_id"):
            g = g[g.gt_id >= 0]
            before = g[(g.t_s > s0 - 1.0) & (g.t_s < s0)]
            after = g[(g.t_s > s0) & (g.t_s < s0 + 1.0)]
            if len(before) < 10 or len(after) < 10:
                continue
            spanning += 1
            if majority_gt(before) != majority_gt(after):
                swaps += 1
    assert spanning > 50, f"слишком мало треков через переключение: {spanning}"
    rate = swaps / spanning
    print(f"  треков через переключение: {spanning}, перепутываний: {swaps} ({100 * rate:.2f}%)")
    assert rate < 0.01, f"перепутываний ID {100 * rate:.2f}% ≥ 1%"


def test_track_purity_and_coverage():
    tracks, det, gt, ev, cal = linked(0)
    real = tracks[tracks.interpolated == 0]
    # чистота: доля детекций трека, принадлежащих его основной капле
    pur = []
    for pid, g in real.groupby("particle_id"):
        if len(g) < 30:
            continue
        m = majority_gt(g)
        pur.append(np.mean(g.gt_id == m))
    pur = np.asarray(pur)
    print(f"  треков ≥30 детекций: {len(pur)}, средняя чистота {pur.mean():.4f}")
    assert pur.mean() > 0.99
    # покрытие: доля видимых детекций настоящих капель, попавших в треки ≥ 1 с
    dur = real.groupby("particle_id").t_s.agg(lambda x: x.max() - x.min())
    in_long = real[real.particle_id.isin(dur.index[dur >= 1.0])]
    cov = in_long.gt_id.ge(0).sum() / max(1, (det.gt_id >= 0).sum())
    print(f"  покрытие детекций длинными треками: {cov:.3f}")
    assert cov > 0.95

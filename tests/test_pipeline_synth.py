"""Сквозной тест на отрендеренном синтетическом видео (медленный: ``--slow``).

Видео 12 с, 150 капель, переключение поля на 6 с. Проверяется:
* детекция: полнота > 0.95 для изолированных видимых капель, медианная
  ошибка центра < 0.3 px (точки) — по кадрам после «разгона» фона;
* весь пайплайн отрабатывает и находит переключение ±2 кадра.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.cli import Pipeline
from src.config import Config
from src.synth import SynthConfig, generate


def _make(tmp: Path) -> tuple[dict, Config]:
    sc = SynthConfig(seed=3, duration_s=12.0, n_drops=150, switches_s=[6.0])
    paths = generate(tmp / "synth.mp4", sc)
    cfg = Config()
    cfg.field.switches_s = None
    cfg.output.dir = str(tmp / "out")
    cfg.output.review_video = False
    cfg.output.plots = False
    cfg.output.per_particle_csv = False
    cfg.background.window_s = 8.0
    return paths, cfg


def test_synthetic_video_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths, cfg = _make(tmp)
        pl = Pipeline(paths["video"], cfg, progress=False)
        det = pl.detections()
        gt = pd.read_csv(paths["gt"])
        recalls, errs_p, errs_r = [], [], []
        for f in range(30, int(gt.frame.max()) + 1, 10):
            g = gt[(gt.frame == f) & (gt.visible == 1)]
            dd = det[det.frame == f]
            if len(dd) == 0:
                recalls.append(0.0)
                continue
            gxy = g[["x", "y"]].values
            # изолированные: ближайшая другая капля дальше r1 + r2 + 4 px, не у края поля
            tg = cKDTree(gxy)
            dist, idx = tg.query(gxy, k=2)
            r = g.r.values
            iso = dist[:, 1] > r + r[idx[:, 1]] + 4
            iso &= np.hypot(g.x - 380, g.y - 540).values < 325 - 2 * r - 15
            td = cKDTree(dd[["x", "y"]].values)
            dmin, _ = td.query(gxy[iso])
            tol = np.maximum(2.0, 0.3 * r[iso])
            recalls.append(float(np.mean(dmin < tol)))
            kinds = g.kind.values[iso]
            ok = dmin < tol
            errs_p.extend(dmin[ok & (kinds == "point")])
            errs_r.extend(dmin[ok & (kinds == "ring")])
        rec = float(np.mean(recalls))
        print(f"  полнота {rec:.3f}, ошибка центра: точки медиана {np.median(errs_p):.3f} px, "
              f"кольца медиана {np.median(errs_r):.3f} px")
        assert rec > 0.95
        assert np.median(errs_p) < 0.3
        assert np.median(errs_r) < 0.5
        res = pl.run(review=False)
        s = res["summary"]
        print(f"  переключения: {s['field_switches_s']}, треков ≥2 с: {s['tracks_min_len']}")
        assert len(s["field_switches_s"]) == 1
        assert abs(s["field_switches_s"][0] - 6.0) <= 2 / 30 + 1e-6


test_synthetic_video_end_to_end.slow = True

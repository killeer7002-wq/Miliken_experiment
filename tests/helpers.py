"""Общие функции тестов: синтетика на уровне детекций и сопоставление с истиной."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from src.calibration import Calibration
from src.config import Config
from src.field_switch import FieldSwitches
from src.link import link_detections
from src.synth import simulate_detections


def synth_calibration(ev: dict) -> Calibration:
    return Calibration(ev["angle"], 51.0, (51.0, 51.0), ev["cx"], ev["cy"], 1, None,
                       np.arange(-300, 301, 51.0), "synthetic")


@functools.lru_cache(maxsize=4)
def simulated(seed: int = 0, duration: float = 30.0, n: int = 150, recharge: float = 0.25):
    det, gt, ev = simulate_detections(n_drops=n, duration_s=duration, seed=seed, recharge_fraction=recharge)
    return det, gt, ev


@functools.lru_cache(maxsize=4)
def linked(seed: int = 0, duration: float = 30.0, n: int = 150, recharge: float = 0.25):
    det, gt, ev = simulated(seed, duration, n, recharge)
    cfg = Config()
    cal = synth_calibration(ev)
    t = np.asarray(ev["t_frames"])
    tracks = link_detections(det, cal, cfg.link, ev["switches_s"], t)
    tracks["gt_id"] = np.where(tracks.det_id >= 0, det.gt_id.values[np.maximum(tracks.det_id.values, 0)], -2)
    return tracks, det, gt, ev, cal


def field_switches_from(ev: dict) -> FieldSwitches:
    t = np.asarray(ev["t_frames"])
    sw = ev["switches_s"]
    states = ["A" if i % 2 == 0 else "B" for i in range(len(sw) + 1)]
    return FieldSwitches(list(sw), [int(np.searchsorted(t, x)) for x in sw], ["config"] * len(sw), states)


def majority_gt(g: pd.DataFrame) -> int:
    v = g.gt_id[g.gt_id >= 0]
    return int(v.mode().iloc[0]) if len(v) else -1

"""Авто-поиск переключений поля (+ уточнение по трекам): ошибка < 2 кадров."""

from __future__ import annotations

import numpy as np

from helpers import linked, synth_calibration
from src.changepoint import pelt_mean
from src.config import Config
from src.field_switch import (apply_refinement, auto_switches_tracks, determine_switches,
                                   replace_times, track_change_series)


def test_pelt_simple():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 80), rng.normal(-1, 1, 120)])
    bps = pelt_mean(y, penalty=3 * np.log(len(y)) * 2, min_size=5)
    assert len(bps) == 2 and abs(bps[0] - 100) <= 3 and abs(bps[1] - 180) <= 3


def test_auto_switches_within_two_frames():
    """Синтетика со сменой полярности (заряды обоих знаков поровну).

    Популяционная статистика здесь ничего не видит (распределение скоростей
    симметрично), поэтому работает знаконезависимый признак по трекам,
    затем уточнение до кадра.
    """
    tracks, det, gt, ev, cal = linked(0)
    cfg = Config()
    cfg.field.switches_s = None
    t = np.asarray(ev["t_frames"])
    fs, stats = determine_switches(det, synth_calibration(ev), t, cfg.field, 0, len(t))
    series = track_change_series(tracks, t, cfg.field)
    auto_tr = auto_switches_tracks(series, cfg.field)
    true = np.asarray(ev["switches_s"])
    assert len(auto_tr) == len(true), f"авто по трекам: {auto_tr}, истинные {true.tolist()}"
    fs = replace_times(fs, auto_tr, t, cfg.field, "auto_tracks")
    fs = apply_refinement(fs, tracks, t, cfg.field)
    err_frames = np.abs(np.asarray(fs.times) - true) * 30
    print(f"  авто по статистике: {np.round(fs.auto_times, 3)} (ожидаемо пусто), по трекам: "
          f"{np.round(auto_tr, 3)}, уточнено: {np.round(fs.times, 3)}, ошибка {np.round(err_frames, 2)} кадра")
    assert np.all(err_frames < 2)

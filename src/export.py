"""Экспорт CSV в output/<имя_видео>/.

Формат: разделитель — запятая, десятичная точка, UTF-8, без индекса pandas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibration
from .config import QualityConfig
from .field_switch import FieldSwitches

log = logging.getLogger(__name__)

TRACK_COLS = ["particle_id", "segment_id", "frame", "t_s", "x_px", "y_px", "z_px", "s_px",
              "z_div", "z_mm", "s_mm", "r_px", "intensity", "kind", "field_state", "interpolated",
              "merge_risk", "flags"]

SEGMENT_COLS = ["particle_id", "segment_id", "t_start", "t_end", "n_frames", "n_real", "field_state",
                "v_z_mm_s", "v_z_err", "v_s_mm_s", "v_s_err", "D_mm2_s",
                "v_z_div_s", "v_z_err_div_s", "v_s_div_s", "v_s_err_div_s", "D_z_div2_s",
                "v_z_px_s", "v_z_err_px_s", "v_s_px_s", "v_s_err_px_s", "D_z_px2_s", "sigma_loc_px",
                "step_local_px", "z_mean_px",
                "r_mean_px", "end_reason", "quality"]

PAIR_COLS = ["particle_id", "segment_a", "segment_b", "state_a", "state_b", "t_switch",
             "v_a", "v_b", "err_a", "err_b", "v_a_div_s", "v_b_div_s", "err_a_div_s", "err_b_div_s",
             "v_a_px_s", "v_b_px_s", "err_a_px_s", "err_b_px_s",
             "dur_a", "dur_b", "quality_a", "quality_b"]


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.6g")


def make_pairs(seg: pd.DataFrame, fs: FieldSwitches, cfg: QualityConfig) -> pd.DataFrame:
    """Пары соседних участков одной капли, разделённых переключением поля без перезарядки."""
    rows = []
    if len(seg) == 0:
        return pd.DataFrame(columns=PAIR_COLS)
    sw = np.asarray(fs.times)
    for pid, g in seg.sort_values(["particle_id", "segment_id"]).groupby("particle_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.end_reason != "field_switch":
                continue
            # переключение между ними
            between = sw[(sw >= a.t_end - 0.1) & (sw <= b.t_start + 0.1)]
            if len(between) == 0:
                continue
            da, db = a.t_end - a.t_start, b.t_end - b.t_start
            if da < cfg.min_segment_s or db < cfg.min_segment_s:
                continue
            rows.append({
                "particle_id": int(pid), "segment_a": int(a.segment_id), "segment_b": int(b.segment_id),
                "state_a": a.field_state, "state_b": b.field_state, "t_switch": float(between[0]),
                "v_a": a.v_z_mm_s, "v_b": b.v_z_mm_s, "err_a": a.v_z_err, "err_b": b.v_z_err,
                "v_a_div_s": a.get("v_z_div_s", np.nan), "v_b_div_s": b.get("v_z_div_s", np.nan),
                "err_a_div_s": a.get("v_z_err_div_s", np.nan), "err_b_div_s": b.get("v_z_err_div_s", np.nan),
                "v_a_px_s": a.v_z_px_s, "v_b_px_s": b.v_z_px_s,
                "err_a_px_s": a.v_z_err_px_s, "err_b_px_s": b.v_z_err_px_s,
                "dur_a": da, "dur_b": db, "quality_a": a.quality, "quality_b": b.quality,
            })
    return pd.DataFrame(rows, columns=PAIR_COLS)


def export_all(out_dir: Path, tracks: pd.DataFrame, seg: pd.DataFrame, fs: FieldSwitches,
               cal: Calibration, qcfg: QualityConfig, per_particle: bool) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = tracks.copy()
    tr["z_div"] = cal.to_div(tr.z_px.values)
    mmd = cal.mm_per_px * cal.step_px if cal.mm_per_px else None
    tr["z_mm"] = tr["z_div"] * mmd if mmd else np.nan
    tr["s_mm"] = cal.mm(tr.s_px.values)
    tr = tr[TRACK_COLS].sort_values(["particle_id", "frame"])
    paths = {}
    paths["tracks"] = out_dir / "tracks.csv"
    _write(tr, paths["tracks"])
    seg_out = seg.reindex(columns=SEGMENT_COLS)
    paths["segments"] = out_dir / "segments.csv"
    _write(seg_out, paths["segments"])
    pairs = make_pairs(seg, fs, qcfg)
    paths["pairs"] = out_dir / "pairs.csv"
    _write(pairs, paths["pairs"])
    paths["field_switches"] = out_dir / "field_switches.csv"
    _write(fs.to_frame(), paths["field_switches"])
    if per_particle:
        pdir = out_dir / "particles"
        pdir.mkdir(exist_ok=True)
        for old in pdir.glob("particle_*.csv"):
            old.unlink()
        # только треки не короче quality.min_track_s (короткие — в основном шум детектора)
        dur = tr.groupby("particle_id").t_s.agg(lambda x: x.max() - x.min())
        keep = set(dur.index[dur >= qcfg.min_track_s])
        for pid, g in tr.groupby("particle_id"):
            if pid in keep:
                _write(g, pdir / f"particle_{int(pid):04d}.csv")
        paths["particles"] = pdir
    log.info("экспорт: %d строк треков, %d участков, %d пар → %s", len(tr), len(seg_out), len(pairs), out_dir)
    return paths

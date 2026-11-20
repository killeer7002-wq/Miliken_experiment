"""Проверка результата глазами: review.mp4, графики z(t) по каплям, сводные графики."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .field_switch import FieldSwitches
from .io_video import iter_frames

log = logging.getLogger(__name__)

_PALETTE = [(66, 135, 245), (245, 66, 230), (66, 245, 200), (245, 170, 66), (160, 66, 245),
            (66, 245, 92), (245, 66, 92), (66, 220, 245), (235, 245, 66), (245, 120, 180)]
_STATE_COL = {"A": "#dfe9f7", "B": "#fbe3d6"}


def select_tracks(tracks: pd.DataFrame, n: int) -> list[int]:
    """n самых длинных (по времени) треков."""
    dur = tracks.groupby("particle_id").t_s.agg(lambda x: x.max() - x.min())
    return dur.sort_values(ascending=False).index[:n].tolist()


def render_review(video: str, tracks: pd.DataFrame, fs: FieldSwitches, out: Path,
                  max_tracks: int, tail_s: float, i0: int, i1: int, scale: float = 1.0) -> Path:
    """Видео: исходный кадр + кружки радиуса r, ID, хвост за tail_s; цвет — по segment_id."""
    ids = select_tracks(tracks, max_tracks)
    tr = tracks[tracks.particle_id.isin(ids)].sort_values("frame")
    by_frame = {f: g for f, g in tr.groupby("frame")}
    hist: dict[int, list[tuple[float, float, float, int]]] = {}
    sw_frames = {f: k for k, f in enumerate(fs.frames)}
    tmp = out.with_suffix(".tmp.mp4")
    writer = None
    flash = 0
    flash_txt = ""
    for i, t, img in iter_frames(video, "bgr", start=i0, stop=i1):
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tmp), fourcc, 30.0, (w, h))
        g = by_frame.get(i)
        if g is not None:
            for row in g.itertuples(index=False):
                hist.setdefault(row.particle_id, []).append((t, row.x_px * scale, row.y_px * scale,
                                                            int(row.segment_id)))
        for pid in list(hist):
            pts = [p for p in hist[pid] if t - p[0] <= tail_s]
            if not pts:
                del hist[pid]
                continue
            hist[pid] = pts
            if pts[-1][0] < t - 1e-6:   # трек уже закончился — хвост гаснет
                continue
            seg = pts[-1][3]
            col = (128, 128, 128) if seg < 0 else _PALETTE[(pid * 3 + seg) % len(_PALETTE)]
            poly = np.array([[p[1], p[2]] for p in pts], np.int32)
            if len(poly) > 1:
                cv2.polylines(img, [poly], False, col, 1, cv2.LINE_AA)
        if g is not None:
            for row in g.itertuples(index=False):
                seg = int(row.segment_id)
                col = (128, 128, 128) if seg < 0 else _PALETTE[(row.particle_id * 3 + seg) % len(_PALETTE)]
                c = (int(round(row.x_px * scale)), int(round(row.y_px * scale)))
                rad = int(round(max(row.r_px, 2.5) * scale)) + 3
                cv2.circle(img, c, rad, col, 1 if row.interpolated else 2, cv2.LINE_AA)
                cv2.putText(img, f"{row.particle_id}.{seg}" if seg >= 0 else f"{row.particle_id}x",
                            (c[0] + rad + 1, c[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * max(scale, 1),
                            col, 1, cv2.LINE_AA)
        if i in sw_frames:
            k = sw_frames[i]
            flash = 30
            flash_txt = f"FIELD SWITCH {fs.states[k]} -> {fs.states[k + 1]}"
        state = str(fs.state_at(t))
        cv2.putText(img, f"t={t:6.2f}s  frame {i}  field {state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if flash > 0:
            cv2.putText(img, flash_txt, (20, img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 255), 3, cv2.LINE_AA)
            flash -= 1
        writer.write(img)
    if writer is not None:
        writer.release()
    # перекодировать в H.264, чтобы видео открывалось в браузере
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast", str(out)])
        if r.returncode == 0:
            tmp.unlink()
            return out
    tmp.replace(out)
    return out


def _shade_states(ax, fs: FieldSwitches, t0: float, t1: float) -> None:
    edges = [t0] + [x for x in fs.times if t0 < x < t1] + [t1]
    for a, b in zip(edges[:-1], edges[1:]):
        st = str(fs.state_at(0.5 * (a + b)))
        ax.axvspan(a, b, color=_STATE_COL.get(st, "#eeeeee"), zorder=0, lw=0)
    for x in fs.times:
        if t0 <= x <= t1:
            ax.axvline(x, color="#c0392b", lw=1, ls="--", zorder=1)


def plot_particles(tracks: pd.DataFrame, seg: pd.DataFrame, fs: FieldSwitches, out_dir: Path,
                   n: int) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("particle_*.png"):
        old.unlink()
    ids = select_tracks(tracks, n)
    segs = {pid: g for pid, g in seg.groupby("particle_id")}
    for pid in ids:
        g = tracks[tracks.particle_id == pid]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        t0, t1 = g.t_s.min(), g.t_s.max()
        _shade_states(ax, fs, t0, t1)
        real = g[g.interpolated == 0]
        interp = g[g.interpolated == 1]
        for sid, gs in real.groupby("segment_id"):
            col = "#7f8c8d" if sid < 0 else plt.cm.tab10(int(sid) % 10)
            ax.plot(gs.t_s, gs.z_px, ".", ms=2.5, color=col, zorder=2,
                    label="после перезарядки" if sid < 0 else None)
        if len(interp):
            ax.plot(interp.t_s, interp.z_px, ".", ms=1.5, color="#bbbbbb", zorder=2, label="интерполяция")
        sp = segs.get(pid)
        if sp is not None:
            for row in sp.itertuples(index=False):
                gs = real[real.segment_id == row.segment_id]
                if len(gs) < 2 or not np.isfinite(row.v_z_px_s):
                    continue
                tt = np.array([row.t_start, row.t_end])
                zc = gs.z_px.mean() + row.v_z_px_s * (tt - gs.t_s.mean())
                ax.plot(tt, zc, "-", color="k", lw=1.3, zorder=3)
                ax.annotate(f"{row.v_z_px_s:+.2f}±{row.v_z_err_px_s:.2f}", (tt.mean(), zc.mean()),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
                if row.end_reason == "charge_change":
                    zz = gs.z_px.mean() + row.v_z_px_s * (row.t_end - gs.t_s.mean())
                    ax.plot([row.t_end], [zz], "x", color="#c0392b", ms=10, mew=2, zorder=4)
        ax.set_xlabel("t, с")
        ax.set_ylabel("z, px")
        ax.set_title(f"капля {pid}: {len(real)} детекций, {t1 - t0:.1f} с "
                     f"(скорости в px/с; × — перезарядка; фон — режим поля)", fontsize=9)
        ax.grid(alpha=0.3)
        h, lab = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"particle_{int(pid):04d}.png", dpi=110)
        plt.close(fig)
    return len(ids)


def plot_summary(seg: pd.DataFrame, fs: FieldSwitches, stats: pd.DataFrame | None, out_dir: Path,
                 min_dur: float) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    good = seg[(seg.t_end - seg.t_start >= min_dur) & np.isfinite(seg.v_z_px_s)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo, hi = np.percentile(good.v_z_px_s, [0.5, 99.5]) if len(good) else (-1, 1)
    bins = np.linspace(lo, hi, 80)
    for st, g in good.groupby("field_state"):
        ax.hist(g.v_z_px_s, bins=bins, alpha=0.55, label=f"поле {st} (n={len(g)})")
    ax.set_xlabel("v_z, px/с")
    ax.set_ylabel("число участков")
    ax.set_title(f"скорости участков длиннее {min_dur:g} с по режимам поля")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "v_z_histogram.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    if stats is not None and len(stats):
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax, col, lab in zip(axs, ["p5", "p95", "frac_fast"],
                                ["5-й перц. v_z, px/с", "95-й перц. v_z, px/с", "доля быстрых"]):
            _shade_states(ax, fs, stats.t.min(), stats.t.max())
            ax.plot(stats.t, stats[col], ".-", ms=3, lw=0.8, color="k")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            if fs.auto_times:
                for x in fs.auto_times:
                    ax.axvline(x, color="#2980b9", lw=0.8, ls=":")
        axs[-1].set_xlabel("t, с")
        axs[0].set_title("грубая статистика скоростей; красный пунктир — используемые моменты "
                         "переключения, синий — авто", fontsize=9)
        fig.tight_layout()
        p = out_dir / "field_switches.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths

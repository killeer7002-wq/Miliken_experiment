"""Командная строка и сборка пайплайна с кэшированием этапов.

    oiltrack run VIDEO -c config.yaml        # весь пайплайн
    oiltrack detect VIDEO -c config.yaml     # только маска, фон, детекции (в кэш)
    oiltrack switches VIDEO -c config.yaml   # моменты переключения: конфиг vs авто vs уточнённые
    oiltrack review VIDEO -c config.yaml     # review.mp4 и графики по готовым трекам
    oiltrack synth OUT.mp4                   # синтетическое видео с известными треками

Кэш лежит в output/<имя_видео>/cache/. Ключ каждого этапа — хэш входного
видео (имя, размер, время изменения), диапазона кадров и всех секций конфига,
от которых этап зависит; при изменении параметра пересчитывается только этот
этап и следующие.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .background import build_background
from .calibration import calibrate
from .config import ChargeConfig, Config, deep_merge, load_config, parse_set
from .charges import load_settings, run_charges, save_settings
from .detect import detect_video, load_detections, save_detections
from .export import export_all, make_pairs
from .field_switch import (apply_refinement, auto_switches_tracks, determine_switches,
                           replace_times, track_change_series)
from .io_video import frame_range, read_timestamps
from .link import link_detections
from .quality import edge_ending_tracks, segment_quality, track_flags
from .review import plot_particles, plot_summary, render_review
from .roi import compute_roi
from .segment import segment_tracks

log = logging.getLogger("oiltrack")


# ------------------------------------------------------------------ кэш
class Cache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def _meta(self, name: str) -> Path:
        return self.root / f"{name}.key"

    def valid(self, name: str, key: str) -> bool:
        m = self._meta(name)
        return self.enabled and m.exists() and m.read_text().strip() == key

    def mark(self, name: str, key: str) -> None:
        self._meta(name).write_text(key)

    def load_pickle(self, name: str, key: str) -> Any | None:
        p = self.root / f"{name}.pkl"
        if self.valid(name, key) and p.exists():
            with open(p, "rb") as fh:
                return pickle.load(fh)
        return None

    def save_pickle(self, name: str, key: str, obj: Any) -> None:
        with open(self.root / f"{name}.pkl", "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.mark(name, key)


def _video_id(path: str) -> dict:
    p = Path(path)
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)}


# ------------------------------------------------------------------ этапы
class Pipeline:
    def __init__(self, video: str, cfg: Config, out_root: Path | None = None, use_cache: bool = True,
                 progress: bool = True):
        self.video = str(video)
        self.cfg = cfg
        name = Path(video).stem
        self.out = (out_root or Path(cfg.output.dir)) / name
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(self.out / "cache", use_cache)
        self.progress = progress
        self.t = read_timestamps(self.video)
        self.i0, self.i1 = frame_range(self.t, cfg.t_start, cfg.t_end)
        if self.i1 - self.i0 < 10:
            raise ValueError("в выбранном интервале меньше 10 кадров")
        self.vid = _video_id(self.video)
        self.timings: dict[str, float] = {}

    def _k(self, *sections: str, extra: Any = None) -> str:
        d = asdict(self.cfg)
        return Cache.key(__version__, self.vid, self.i0, self.i1, [d[s] for s in sections], extra)

    def roi_bg(self):
        k = self._k("roi", "background", extra=self.cfg.detect.channel)
        obj = self.cache.load_pickle("roi_bg", k)
        if obj is None:
            t0 = time.time()
            roi = compute_roi(self.video, len(self.t), self.cfg.roi, self.cfg.detect.channel, self.i0, self.i1)
            bg = build_background(self.video, roi, self.cfg.background, self.cfg.detect.channel,
                                  self.i0, self.i1, self.t)
            obj = (roi, bg)
            self.cache.save_pickle("roi_bg", k, obj)
            self.timings["фон"] = time.time() - t0
        else:
            log.info("маска и фон: из кэша")
        return obj

    def detections(self) -> pd.DataFrame:
        k = self._k("roi", "background", "detect")
        stem = str(self.out / "detections")
        if self.cache.valid("detections", k):
            det = load_detections(stem)
            if det is not None:
                log.info("детекции: из кэша (%d)", len(det))
                return det
        roi, bg = self.roi_bg()
        t0 = time.time()
        det = detect_video(self.video, bg, self.cfg.detect, self.i0, self.i1, self.progress)
        path = save_detections(det, stem)
        self.cache.mark("detections", k)
        self.timings["детекция"] = time.time() - t0
        n_fr = self.i1 - self.i0
        log.info("детекции: %d (в среднем %.0f на кадр, колец %.0f%%) → %s", len(det), len(det) / n_fr,
                 100 * (det.kind == "ring").mean() if len(det) else 0, Path(path).name)
        return det

    def calibration(self):
        roi, _bg = self.roi_bg()
        return calibrate(roi, self.cfg.calibration)

    def switches_and_tracks(self, det: pd.DataFrame):
        cal = self.calibration()
        k = self._k("roi", "background", "detect", "calibration", "field", "link")
        obj = self.cache.load_pickle("tracks", k)
        if obj is not None:
            log.info("переключения и треки: из кэша")
            return (cal,) + tuple(obj)
        t0 = time.time()
        fs, stats = determine_switches(det, cal, self.t, self.cfg.field, self.i0, self.i1)
        tracks = link_detections(det, cal, self.cfg.link, fs.times, self.t, (self.i0, self.i1))
        # знаконезависимый авто-поиск по трекам
        series = track_change_series(tracks, self.t, self.cfg.field)
        fs.auto_track_times = auto_switches_tracks(series, self.cfg.field)
        log.info("переключения поля (авто по трекам): %s", [round(x, 2) for x in fs.auto_track_times])
        self.track_series = series
        series.to_csv(self.out / "switch_series.csv", index=False)
        if self.cfg.field.switches_s is None and fs.auto_track_times and \
                (len(fs.auto_track_times) != len(fs.times) or
                 np.max(np.abs(np.asarray(fs.auto_track_times) - np.asarray(fs.times))) > 0.5):
            log.info("авто по трекам расходится с авто по статистике — беру по трекам и пересвязываю")
            fs = replace_times(fs, fs.auto_track_times, self.t, self.cfg.field, "auto_tracks")
            tracks = link_detections(det, cal, self.cfg.link, fs.times, self.t, (self.i0, self.i1))
        if self.cfg.field.auto_refine and fs.times:
            old = list(fs.times)
            fs = apply_refinement(fs, tracks, self.t, self.cfg.field)
            # если момент сдвинулся больше чем на полсекунды сброса — пересвязать
            if any(abs(a - b) > 0.5 * self.cfg.link.switch_reset_s for a, b in zip(old, fs.times)):
                log.info("моменты переключения заметно сдвинулись — повторное сопоставление")
                tracks = link_detections(det, cal, self.cfg.link, fs.times, self.t, (self.i0, self.i1))
        self.cache.save_pickle("tracks", k, (fs, stats, tracks))
        self.timings["переключения+треки"] = time.time() - t0
        return cal, fs, stats, tracks

    def run(self, review: bool = True) -> dict[str, Any]:
        cfg = self.cfg
        t_start = time.time()
        det = self.detections()
        roi, _bg = self.roi_bg()
        cal, fs, stats, tracks = self.switches_and_tracks(det)
        t0 = time.time()
        edge_ids = edge_ending_tracks(tracks, roi, cfg.quality)
        max_gap_s = cfg.link.max_gap_frames * float(np.median(np.diff(self.t)))
        tracks, seg = segment_tracks(tracks, fs, cal, cfg.segment, float(self.t[self.i1 - 1]),
                                     max_gap_s, edge_ids)
        tracks = track_flags(tracks, roi, cfg.quality)
        seg = segment_quality(seg, tracks, cfg.quality)
        self.timings["нарезка+качество"] = time.time() - t0
        t0 = time.time()
        paths = export_all(self.out, tracks, seg, fs, cal, cfg.quality, cfg.output.per_particle_csv)
        stats.to_csv(self.out / "velocity_stats.csv", index=False)
        self.timings["экспорт"] = time.time() - t0
        with open(self.out / "cache" / "final.pkl", "wb") as fh:
            pickle.dump((tracks, seg, fs, stats, cal), fh, protocol=pickle.HIGHEST_PROTOCOL)
        if review:
            self.review(tracks, seg, fs, stats)
        summary = self.summary(tracks, seg, fs, cal)
        (self.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if cfg.charge.enabled:
            t0 = time.time()
            try:
                save_settings(self.out, cfg.charge, cfg.calibration.mm_per_division)
                summary["charges"] = run_charges([self.out], cfg.charge, cfg.quality.min_segment_s,
                                                 cfg.calibration.mm_per_division, self.out)
            except Exception as exc:   # заряды — последний шаг, не роняем весь прогон
                log.warning("заряды не посчитаны: %s", exc)
            self.timings["заряды"] = time.time() - t0
        summary["time_total_s"] = round(time.time() - t_start, 1)
        (self.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        self.print_summary(summary)
        return {"paths": paths, "summary": summary}

    def review(self, tracks=None, seg=None, fs=None, stats=None) -> None:
        cfg = self.cfg
        if tracks is None:
            p = self.out / "cache" / "final.pkl"
            if not p.exists():
                raise FileNotFoundError("нет готовых треков: сначала `oiltrack run`")
            with open(p, "rb") as fh:
                tracks, seg, fs, stats, _cal = pickle.load(fh)
        t0 = time.time()
        if cfg.output.plots:
            n = plot_particles(tracks, seg, fs, self.out / "plots", cfg.output.max_plots)
            plot_summary(seg, fs, stats, self.out / "plots", cfg.quality.min_segment_s)
            log.info("графики: %d капель → %s", n, self.out / "plots")
        if cfg.output.review_video:
            p = render_review(self.video, tracks, fs, self.out / "review.mp4", cfg.output.max_review_tracks,
                              cfg.output.review_tail_s, self.i0, self.i1, cfg.output.review_scale)
            log.info("review-видео → %s", p)
        self.timings["review"] = time.time() - t0

    def summary(self, tracks, seg, fs, cal) -> dict[str, Any]:
        dur = tracks.groupby("particle_id").t_s.agg(lambda x: x.max() - x.min())
        good = dur[dur >= self.cfg.quality.min_track_s]
        pairs = make_pairs(seg, fs, self.cfg.quality)
        return {
            "video": self.video,
            "frames": [int(self.i0), int(self.i1)],
            "calibration": cal.summary(),
            "grid_angle_deg": round(cal.angle_deg, 3),
            "grid_step_px": round(cal.step_px, 3),
            "field_switches_s": [round(x, 3) for x in fs.times],
            "field_switches_auto_s": [round(x, 3) for x in (fs.auto_times or [])],
            "tracks_total": int(len(dur)),
            "tracks_min_len": int(len(good)),
            "min_track_s": self.cfg.quality.min_track_s,
            "median_track_s": round(float(good.median()), 2) if len(good) else 0.0,
            "tracks_over_10s": int((dur > 10).sum()),
            "tracks_over_30s": int((dur > 30).sum()),
            "segments": int(len(seg)),
            "segments_ok": int((seg.quality == "ok").sum()) if len(seg) else 0,
            "charge_changes": int((seg.end_reason == "charge_change").sum()) if len(seg) else 0,
            "pairs": int(len(pairs)),
            "timings_s": {k: round(v, 1) for k, v in self.timings.items()},
        }

    @staticmethod
    def print_summary(s: dict[str, Any]) -> None:
        lines = [
            "=" * 60,
            f"видео: {s['video']}  кадры {s['frames'][0]}–{s['frames'][1]}",
            f"калибровка: {s['calibration']}",
            f"переключения поля: {s['field_switches_s']} (авто: {s['field_switches_auto_s']})",
            f"треков всего: {s['tracks_total']}, не короче {s['min_track_s']} с: {s['tracks_min_len']}, "
            f"медианная длина {s['median_track_s']} с, >10 с: {s['tracks_over_10s']}, >30 с: {s['tracks_over_30s']}",
            f"участков: {s['segments']} (без флагов: {s['segments_ok']}), перезарядок: {s['charge_changes']}, "
            f"пар поле±: {s['pairs']}",
            (f"заряды: капель {s['charges']['particles']}, значимых {s['charges']['significant']} "
             f"(charges.csv, charges_hist.png)") if s.get("charges") else "заряды: —",
            f"время: {s.get('time_total_s', '?')} с  {s['timings_s']}",
            "=" * 60,
        ]
        print("\n".join(lines))


# ------------------------------------------------------------------ CLI
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", help="путь к видео")
    p.add_argument("-c", "--config", action="append", default=None,
                   help="config.yaml; можно несколько раз — каждый следующий накладывается на предыдущий")
    p.add_argument("--set", action="append", default=None, metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                   help="переопределить параметр, например --set charge.voltage_V=700")
    p.add_argument("-o", "--out", default=None, help="корневая папка вывода (по умолчанию output.dir)")
    p.add_argument("--t-start", type=float, default=None, help="начало куска, с")
    p.add_argument("--t-end", type=float, default=None, help="конец куска, с")
    p.add_argument("--no-cache", action="store_true", help="пересчитать всё")
    p.add_argument("-q", "--quiet", action="store_true")


def _make(args) -> Pipeline:
    over = deep_merge({"t_start": args.t_start, "t_end": args.t_end, "video": args.video},
                      parse_set(args.set))
    cfg = load_config(args.config, over)
    return Pipeline(args.video, cfg, Path(args.out) if args.out else None, not args.no_cache,
                    progress=not args.quiet)


def _per_dir_charge_cfg(dirs: list[Path], cfg: Config, sets: dict) -> dict:
    """Константы зарядов для каждой папки.

    Приоритет (от слабого к сильному): общий конфиг (-c) → значения, с которыми
    это видео прогоняли (charge_settings.json в папке: там его напряжение) →
    --set из командной строки.
    """
    from dataclasses import asdict

    out = {}
    for d in dirs:
        saved = load_settings(d) or {}
        mmd = saved.pop("_mm_per_division", None)
        merged = deep_merge(deep_merge(asdict(cfg.charge), saved), sets.get("charge", {}))
        mm = sets.get("calibration", {}).get("mm_per_division", mmd if mmd is not None
                                             else cfg.calibration.mm_per_division)
        out[d] = (ChargeConfig(**{k: v for k, v in merged.items() if k in ChargeConfig.__dataclass_fields__}), mm)
    return out


def _print_charges(res: dict, dest: Path) -> None:
    print(f"пар: {res.get('pairs', 0)} (годных {res.get('pairs_used', 0)}), капель: {res.get('particles', 0)}, "
          f"значимых зарядов: {res.get('significant', 0)}; гистограмма: {res.get('histogram', '—')} → {dest}")


def run_batch(manifest: Path, use_cache: bool = True, only_charges: bool = False, progress: bool = True,
              review: bool | None = None) -> int:
    """Серия видео по списку videos.yaml: свои параметры у каждого + общая гистограмма зарядов.

    Формат::

        config: config.yaml              # общий конфиг (пути — относительно videos.yaml)
        out: output                      # корневая папка вывода (необязательно)
        charges_out: output/charges_all  # куда общие заряды (необязательно)
        review: false                    # рисовать ли review.mp4 и графики
        set:                             # общие переопределения для всех видео (необязательно)
          charge: {rho_oil_kg_m3: 875}
        videos:
          - path: 700_1.mov
            set: {charge: {voltage_V: 700}}
          - path: 500_1.mov
            set:
              charge: {voltage_V: 500}
              field: {switches_s: [12.0, 30.5]}
    """
    import yaml

    m = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    base_dir = manifest.parent
    rel = lambda p: p if Path(p).is_absolute() else str(base_dir / p)  # noqa: E731
    base_cfgs = m.get("config")
    base_cfgs = [rel(c) for c in (base_cfgs if isinstance(base_cfgs, list) else [base_cfgs])] if base_cfgs else None
    common = m.get("set") or {}
    out_root = Path(rel(m["out"])) if m.get("out") else None
    dirs, per_dir = [], {}
    cfg0 = None
    for i, v in enumerate(m.get("videos") or []):
        if isinstance(v, str):
            v = {"path": v}
        path = rel(v["path"])
        over = deep_merge(deep_merge(common, v.get("set") or {}), {"video": path})
        cfg = load_config(base_cfgs, over)
        cfg0 = cfg0 or cfg
        log.info("=== [%d/%d] %s ===", i + 1, len(m["videos"]), path)
        pl = Pipeline(path, cfg, out_root, use_cache, progress)
        if not only_charges:
            cfg.charge.enabled = False      # общие заряды посчитаем ниже по всем видео
            pl.run(review=bool(m.get("review", False)) if review is None else review)
            cfg.charge.enabled = True
        save_settings(pl.out, cfg.charge, cfg.calibration.mm_per_division)
        dirs.append(pl.out)
        per_dir[pl.out] = (cfg.charge, cfg.calibration.mm_per_division)
    if not dirs:
        print("в videos.yaml нет видео")
        return 1
    # заряды по каждому видео в его папку и общие
    for d in dirs:
        c, mm = per_dir[d]
        run_charges([d], c, cfg0.quality.min_segment_s, mm, d, {d: per_dir[d]})
    dest = Path(rel(m["charges_out"])) if m.get("charges_out") else (out_root or Path(cfg0.output.dir)) / "charges_all"
    res = run_charges(dirs, cfg0.charge, cfg0.quality.min_segment_s, cfg0.calibration.mm_per_division, dest, per_dir)
    _print_charges(res, dest)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oiltrack", description="трекер капель масла (опыт Милликена)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("run", "весь пайплайн"), ("detect", "маска, фон и детекции"),
                      ("switches", "моменты переключения поля"), ("review", "review.mp4 и графики")):
        p = sub.add_parser(name, help=hlp)
        _common(p)
        if name == "run":
            p.add_argument("--no-review", action="store_true", help="не рисовать видео и графики")
            p.add_argument("--mode", choices=["stop", "split"], default=None, help="segment.mode")
    pc = sub.add_parser("charges", help="заряды по одной или нескольким папкам output/<видео>")
    pc.add_argument("dirs", nargs="+", help="папки с pairs.csv и segments.csv")
    pc.add_argument("-c", "--config", action="append", default=None)
    pc.add_argument("--set", action="append", default=None, metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                    help="применяется ко всем папкам, например --set charge.rho_oil_kg_m3=875")
    pc.add_argument("-o", "--out", default=None, help="куда писать (по умолчанию — в папку, если она одна, "
                                                       "иначе output/charges_all)")
    pc.add_argument("--off-state", default=None, help="режим «поле выключено» (по умолчанию auto)")
    pb = sub.add_parser("batch", help="серия видео со своими параметрами (videos.yaml) + общие заряды")
    pb.add_argument("manifest", help="videos.yaml")
    pb.add_argument("--no-cache", action="store_true")
    pb.add_argument("--only-charges", action="store_true", help="не прогонять видео, только пересчитать заряды")
    g = pb.add_mutually_exclusive_group()
    g.add_argument("--review", dest="review", action="store_true", default=None,
                   help="рисовать review.mp4 и графики (перекрывает review: в videos.yaml)")
    g.add_argument("--no-review", dest="review", action="store_false",
                   help="не рисовать review.mp4 и графики")
    pb.add_argument("-q", "--quiet", action="store_true")
    ps = sub.add_parser("synth", help="синтетическое видео с известными треками")
    ps.add_argument("out", help="OUT.mp4")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--n-drops", type=int, default=150)
    ps.add_argument("--duration", type=float, default=30.0)
    ps.add_argument("--switches", type=float, nargs="*", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "synth":
        from .synth import SynthConfig, generate

        sc = SynthConfig(seed=args.seed, n_drops=args.n_drops, duration_s=args.duration)
        if args.switches is not None:
            sc.switches_s = list(args.switches)
        gt = generate(args.out, sc)
        print(f"синтетика: {args.out}, ground truth: {gt}")
        return 0
    if args.cmd == "charges":
        sets = parse_set(args.set)
        if args.off_state:
            sets = deep_merge(sets, {"charge": {"off_state": args.off_state}})
        cfg = load_config(args.config, sets)
        dirs = [Path(d) for d in args.dirs]
        per_dir = _per_dir_charge_cfg(dirs, cfg, sets)
        dest = Path(args.out) if args.out else (dirs[0] if len(dirs) == 1 else Path(cfg.output.dir) / "charges_all")
        res = run_charges(dirs, cfg.charge, cfg.quality.min_segment_s, cfg.calibration.mm_per_division, dest, per_dir)
        _print_charges(res, dest)
        return 0
    if args.cmd == "batch":
        return run_batch(Path(args.manifest), use_cache=not args.no_cache,
                         only_charges=args.only_charges, progress=not args.quiet, review=args.review)
    pl = _make(args)
    if args.cmd == "run":
        if args.mode:
            pl.cfg.segment.mode = args.mode
        pl.run(review=not args.no_review)
    elif args.cmd == "detect":
        det = pl.detections()
        print(f"детекций: {len(det)}, на кадр: {len(det) / (pl.i1 - pl.i0):.1f}")
    elif args.cmd == "switches":
        det = pl.detections()
        cal, fs, stats, _tr = pl.switches_and_tracks(det)
        out = pl.out / "field_switches.csv"
        fs.to_frame().to_csv(out, index=False)
        plot_summary(pd.DataFrame(columns=["t_start", "t_end", "v_z_px_s", "field_state"]), fs, stats,
                     pl.out / "plots", 1.0)
        print(fs.to_frame().to_string(index=False))
        print(f"→ {out}")
    elif args.cmd == "review":
        pl.review()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

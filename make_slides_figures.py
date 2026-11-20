#!/usr/bin/env python3
"""Графики и числа для слайдов «Обработка данных» (опыт Милликена).

Берёт готовые результаты oiltrack batch:

    python3 make_slides_figures.py output/charges_all -m videos.yaml -o figs

* пары скоростей и отбор — из output/charges_all/charge_pairs.csv (то, что
  посчитал batch по всем видео);
* константы каждого видео (U, d, ρ, η, цена деления, давление) — из videos.yaml
  (-m) так же, как их собирает batch; без -m — из output/<видео>/charge_settings.json,
  который batch сохраняет; ещё без него — из config.yaml (-c);
* траектория, распределение скоростей и кадр с треками — из папок output/<видео>/
  (рядом с charges_all).

Заряды считаются дважды из одних и тех же пар: по формуле Стокса и с поправкой
Каннингема; капли для обеих гистограмм — одни и те же.

Что получается в папке figs/:
  fig_tracking.png      кадр review.mp4 с треками (если есть review.mp4)
  fig_zt.png            z(t) одной капли: падение / подъём, прямые и скорости
  fig_velocities.png    распределение скоростей в режимах «поле выкл / вкл»
  fig_q_raw.png         заряды по формуле Стокса — гистограмма |q|/e
  fig_qn_r_raw.png      |q|/n в зависимости от радиуса капли (без поправки)
  fig_q_corr.png        заряды с поправкой Каннингема
  fig_qn_r_compare.png  |q|/n от радиуса: до и после поправки
  results.tex           числа для слайдов (\\newcommand)
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import json

from src.charges import E_CHARGE, absolute_charge
from src.config import ChargeConfig, deep_merge, load_config, parse_set

import re
import yaml


class _YamlLoader(yaml.SafeLoader):
    """Читает числа вида 5e-3 как числа, а не как текст."""


_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?
        |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def yaml_load(text):
    return yaml.load(text, Loader=_YamlLoader)

MAIN = "#952A1F"      # maincolor из mytheme.sty
GREY = "#9a9a9a"
BLUE = "#2f5d8a"


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ------------------------------------------------------------------ e по зарядам

def coherence(q: np.ndarray, x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """|⟨exp(2πi q/x)⟩| для каждого x: 1 — все q точно кратны x, ~0 — нет кратности."""
    w = np.ones_like(q) if w is None else w
    ph = np.exp(2j * np.pi * q[None, :] / x[:, None])
    return np.abs((ph * w[None, :]).sum(axis=1)) / w.sum()


def quantum_scan(q: np.ndarray, lo: float = 0.6, hi: float = 1.8) -> tuple[float, float]:
    """Шаг, на котором стоят заряды (в единицах e), и когерентность на нём.

    Берётся самый высокий пик когерентности; если на удвоенном шаге есть пик не
    ниже 85 % — значит, найдена половина настоящего шага (кратные доли x/2 тоже
    дают высокую когерентность), и берётся удвоенный. Диапазон 0.6–1.8 e
    предполагает, что константы известны хотя бы с точностью до ~1.7 раза.
    """
    from scipy.signal import find_peaks

    x = np.linspace(lo, hi, 2401)
    S = coherence(q, x)
    pk, _ = find_peaks(S)
    if len(pk) == 0:
        k = int(np.argmax(S))
        return float(x[k]), float(S[k])
    best = pk[np.argmax(S[pk])]
    x0, s0 = x[best], S[best]
    near2 = pk[np.abs(x[pk] - 2 * x0) < 0.06 * x0]
    if len(near2) and S[near2].max() >= 0.85 * s0:
        j = near2[np.argmax(S[near2])]
        return float(x[j]), float(S[j])
    return float(x0), float(s0)


def fit_e(q: np.ndarray, w: np.ndarray, e0: float, n_max: int = 6, frac_max: float = 0.3) -> dict:
    """МНК через начало координат q = n·e: n = round(q/e), итерации; отбрасываются
    капли между пиками (|q/e − n| > frac_max) и с n > n_max."""
    e = e0
    w0 = w
    for _ in range(5):
        n = np.round(np.abs(q) / e)
        frac = np.abs(np.abs(q) / e - n)
        sel = (n >= 1) & (n <= n_max) & (frac <= frac_max)
        # погрешность заряда ∝ самому заряду; вес считаем от номинального n·e, а не от
        # измеренного q — иначе случайно заниженные заряды получают больший вес (смещение вниз)
        w = w0 * (np.abs(q) / np.maximum(n * e, 1e-9)) ** 2
        e = float(np.sum(w[sel] * n[sel] * np.abs(q[sel])) / np.sum(w[sel] * n[sel] ** 2))
    n = np.round(np.abs(q) / e)
    frac = np.abs(np.abs(q) / e - n)
    sel = (n >= 1) & (n <= n_max) & (frac <= frac_max)
    r = np.abs(q[sel]) - n[sel] * e
    dof = max(sel.sum() - 1, 1)
    # погрешность: по разбросу остатков (учитывает и неучтённые ошибки)
    s2 = np.sum(w[sel] * r**2) / dof
    err = float(np.sqrt(s2 / np.sum(w[sel] * n[sel] ** 2)))
    return {"e": e, "err": err, "n": n, "sel": sel, "n_used": int(sel.sum()), "n_total": len(q)}


# ------------------------------------------------------------------ графики

def plot_hist(plt, q_e: np.ndarray, sel: np.ndarray, e_fit: float, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    qa = np.abs(q_e)
    bins = np.arange(0, min(7.0, qa.max() + 0.5), 0.1)
    ax.hist(qa, bins=bins, color=GREY, edgecolor="white", label="все значимые капли")
    ax.hist(qa[sel], bins=bins, color=MAIN, edgecolor="white", label="использованы в оценке")
    top = ax.get_ylim()[1]
    for k in range(1, int(bins[-1]) + 1):
        ax.axvline(k, color="k", lw=1.0, ls="-", alpha=0.55)
        ax.axvline(k * e_fit, color=BLUE, lw=1.2, ls="--", alpha=0.9)
    ax.plot([], [], color="k", lw=1, label=r"$n\,e$ (табличное $e$)")
    ax.plot([], [], color=BLUE, lw=1.2, ls="--", label=fr"$n\,e^*$, $e^* = {e_fit:.2f}\,e$")
    ax.set_ylim(0, top)
    ax.set_xlim(0, bins[-1])
    ax.set_xlabel(r"$|q|\,/\,e$")
    ax.set_ylabel("число капель")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_qn_r(plt, axes, df: pd.DataFrame, fit: dict, label: str, color: str):
    s = fit["sel"]
    r = df.radius_um.values[s]
    qn = np.abs(df.q_over_e.values[s]) / fit["n"][s]
    qn_err = df.q_over_e_err.values[s] / fit["n"][s]
    axes.errorbar(r, qn, yerr=qn_err, fmt="o", ms=3, lw=0.6, color=color, alpha=0.75, label=label)
    # скользящая медиана по радиусу — тренд
    o = np.argsort(r)
    if len(o) > 15:
        k = max(5, len(o) // 8)
        rm = pd.Series(r[o]).rolling(k, center=True).median()
        qm = pd.Series(qn[o]).rolling(k, center=True).median()
        axes.plot(rm, qm, color=color, lw=2.2)


def pick_example(pairs: pd.DataFrame) -> tuple[str, int] | None:
    ok = pairs[(pairs.use == "ok") & (pairs.dur_off >= 4) & (pairs.dur_on >= 4)].copy()
    if len(ok) == 0:
        ok = pairs[pairs.use == "ok"].copy()
    if len(ok) == 0:
        return None
    ok["sig"] = np.abs(ok.q_rel) / ok.q_rel_err
    ok = ok[ok.v2 > 0] if (ok.v2 > 0).any() else ok      # нагляднее: капля поднимается в поле
    best = ok.sort_values("sig", ascending=False).iloc[0]
    return str(best.video), int(best.particle_id)


def plot_zt(plt, out_dir: Path, pid: int, path: Path, off_state: str):
    tr = []
    for chunk in pd.read_csv(out_dir / "tracks.csv", usecols=["particle_id", "segment_id", "t_s", "z_div",
                                                               "field_state", "interpolated"],
                             chunksize=500_000):
        tr.append(chunk[chunk.particle_id == pid])
    tr = pd.concat(tr)
    seg = pd.read_csv(out_dir / "segments.csv")
    seg = seg[seg.particle_id == pid]
    real = tr[(tr.interpolated == 0) & (tr.segment_id >= 0)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    t0, t1 = tr.t_s.min(), tr.t_s.max()
    # фон — режимы поля
    st = tr.field_state.values
    edges = np.nonzero(st[1:] != st[:-1])[0]
    bounds = [t0] + list(tr.t_s.values[edges + 1]) + [t1]
    states = [st[0]] + list(st[edges + 1])
    for a, b, s_ in zip(bounds[:-1], bounds[1:], states):
        ax.axvspan(a, b, color="#e8eef6" if s_ == off_state else "#f8e3df", lw=0, zorder=0)
    ax.plot(real.t_s, real.z_div, ".", ms=2, color="k", zorder=2)
    for row in seg.itertuples(index=False):
        g = real[real.segment_id == row.segment_id]
        if len(g) < 5 or not np.isfinite(row.v_z_div_s):
            continue
        tt = np.array([g.t_s.min(), g.t_s.max()])
        zc = g.z_div.mean() + row.v_z_div_s * (tt - g.t_s.mean())
        ax.plot(tt, zc, color=MAIN, lw=2, zorder=3)
        lab = (r"$v_\downarrow$" if row.field_state == off_state else r"$v_\uparrow$")
        ax.annotate(f"{lab} = {abs(row.v_z_div_s):.3f} дел/с", (tt.mean(), zc.mean()),
                    textcoords="offset points", xytext=(0, 16), ha="center", fontsize=9, color=MAIN,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("t, с")
    ax.set_ylabel("z, деления шкалы")
    ax.set_title("Траектория одной капли (синий фон — поле выключено, красный — включено)", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_velocities(plt, dirs: list[Path], off_by_video: dict, path: Path):
    rows = []
    for d in dirs:
        s = pd.read_csv(d / "segments.csv")
        s = s[(s.quality == "ok") & (s.t_end - s.t_start >= 2) & np.isfinite(s.v_z_div_s)]
        off = off_by_video.get(d.name)
        s = s.assign(mode=np.where(s.field_state == off, "off", "on"))
        rows.append(s)
    s = pd.concat(rows)
    # знак: «вверх» положительно (оседание при выключенном поле — отрицательно)
    med_off = np.median(s[s["mode"] == "off"].v_z_div_s)
    sign = -1.0 if med_off > 0 else 1.0
    v = s.v_z_div_s * sign
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lo, hi = np.percentile(v, [1, 99])
    bins = np.linspace(lo, hi, 70)
    ax.hist(v[s["mode"] == "off"], bins=bins, color=BLUE, alpha=0.7, label="поле выключено")
    ax.hist(v[s["mode"] == "on"], bins=bins, color=MAIN, alpha=0.6, label="поле включено")
    ax.set_xlabel("скорость вдоль поля, дел/с (вверх > 0)")
    ax.set_ylabel("число участков")
    ax.legend(fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return int((s["mode"] == "off").sum()), int((s["mode"] == "on").sum())


def tracking_frame(plt, dirs: list[Path], path: Path) -> bool:
    import cv2

    for d in dirs:
        p = d / "review.mp4"
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 3)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        g = img.max(axis=2)
        ys, xs = np.nonzero(g > 25)
        if len(xs):
            y0, y1 = max(0, ys.min() - 5), min(img.shape[0], ys.max() + 5)
            x0, x1 = max(0, xs.min() - 5), min(img.shape[1], xs.max() + 5)
            img = img[y0:y1, x0:x1]
        cv2.imwrite(str(path), img)
        return True
    return False


# ------------------------------------------------------------------ данные

def settings_from_manifest(manifest: Path) -> dict[str, tuple[ChargeConfig, float | None, str]]:
    """{имя видео: (константы зарядов, цена деления, папка вывода)} — так же, как собирает batch."""
    m = yaml_load(manifest.read_text(encoding="utf-8")) or {}
    base = manifest.parent
    rel = lambda p_: p_ if Path(p_).is_absolute() else str(base / p_)  # noqa: E731
    cfgs = m.get("config")
    cfgs = [rel(c) for c in (cfgs if isinstance(cfgs, list) else [cfgs])] if cfgs else None
    common = m.get("set") or {}
    out = {}
    for v in m.get("videos") or []:
        v = {"path": v} if isinstance(v, str) else v
        path = rel(v["path"])
        cfg = load_config(cfgs, deep_merge(deep_merge(common, v.get("set") or {}), {"video": path}))
        root = Path(rel(m["out"])) if m.get("out") else Path(cfg.output.dir)
        name = Path(path).stem
        out[name] = (cfg.charge, cfg.calibration.mm_per_division, str(root / name))
    return out


def settings_from_saved(video: str, vdir: Path, cfg) -> tuple[ChargeConfig, float | None, str]:
    p = vdir / "charge_settings.json"
    if p.exists():
        d = json.loads(p.read_text())
        mmd = d.pop("_mm_per_division", None)
        fields = ChargeConfig.__dataclass_fields__
        return ChargeConfig(**{k: v for k, v in d.items() if k in fields}), mmd, str(vdir)
    return cfg.charge, cfg.calibration.mm_per_division, str(vdir)


def charges_both(pairs: pd.DataFrame, settings: dict, b_default: float) -> pd.DataFrame:
    """Заряды по Стоксу и с поправкой Каннингема для каждой пары (одни и те же пары)."""
    rows = []
    for video, g in pairs.groupby("video"):
        if video not in settings:
            raise SystemExit(f"нет констант для видео {video}: добавьте его в videos.yaml или запустите batch")
        c, mmd, _ = settings[video]
        if (g.unit != "div").any():
            raise SystemExit(f"{video}: скорости не в делениях шкалы (сетка не найдена) — задайте grid_step_px")
        if not (c.has_constants() and mmd):
            raise SystemExit(f"{video}: не заданы константы (calibration.mm_per_division, charge.plate_gap_m, "
                             "charge.voltage_V, charge.rho_oil_kg_m3)")
        k = mmd * 1e-3                                    # м на деление
        v1, v2 = g.v1.values * k, g.v2.values * k
        e1, e2 = g.err_off.values * k, g.err_on.values * k
        out = g[["video", "particle_id"]].copy()
        b = c.cunningham_b_Pa_m or b_default
        for tag, bb in (("raw", None), ("corr", b)):
            cc = ChargeConfig(**{**c.__dict__, "cunningham_b_Pa_m": bb})
            q, a = absolute_charge(v1, v2, cc)
            q2, _ = absolute_charge(v1, v2 + e2, cc)
            q1, _ = absolute_charge(v1 + e1, v2, cc)
            out[f"q_{tag}"] = q / E_CHARGE
            out[f"err_{tag}"] = np.hypot(q2 - q, q1 - q) / E_CHARGE
            out[f"r_{tag}"] = a * 1e6
        out["b"] = b
        out["p"] = c.pressure_Pa
        out["voltage_V"] = c.voltage_V
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def per_drop(q: pd.DataFrame) -> pd.DataFrame:
    """Средневзвешенный заряд капли по её парам (оба варианта)."""
    rows = []
    for (video, pid), g in q.groupby(["video", "particle_id"]):
        row = {"video": video, "particle_id": pid, "n_pairs": len(g)}
        for tag in ("raw", "corr"):
            w = 1 / np.maximum(g[f"err_{tag}"].values, 1e-6) ** 2
            row[f"q_{tag}"] = float(np.sum(w * g[f"q_{tag}"]) / np.sum(w))
            row[f"err_{tag}"] = float(1 / np.sqrt(np.sum(w)))
            row[f"r_{tag}"] = float(g[f"r_{tag}"].mean())
        row["b"], row["p"] = g.b.iloc[0], g.p.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def as_fit_frame(d: pd.DataFrame, tag: str) -> pd.DataFrame:
    return pd.DataFrame({"q_over_e": d[f"q_{tag}"].values, "q_over_e_err": d[f"err_{tag}"].values,
                         "radius_um": d[f"r_{tag}"].values})


def near_integer_fraction(q: np.ndarray, e: float, tol: float = 0.2) -> float:
    x = np.abs(q) / e
    n = np.round(x)
    return float(np.mean((n >= 1) & (np.abs(x - n) <= tol)))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("charges_dir", nargs="?", default="output/charges_all",
                    help="папка общих зарядов batch (по умолчанию output/charges_all)")
    ap.add_argument("-m", "--manifest", default=None, help="videos.yaml — константы каждого видео")
    ap.add_argument("-c", "--config", action="append", default=None, help="config.yaml (если нет videos.yaml)")
    ap.add_argument("-o", "--out", default="figs")
    ap.add_argument("--b", type=float, default=8.2e-3,
                    help="постоянная Каннингема b, Па·м, если не задана в настройках (8.2e-3)")
    ap.add_argument("--n-max", type=int, default=6, help="наибольший n в оценке e (6)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    plt = setup_mpl()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.charges_dir)
    pairs_all = pd.read_csv(cdir / "charge_pairs.csv")
    videos = sorted(pairs_all.video.unique())
    if args.manifest:
        settings = settings_from_manifest(Path(args.manifest))
    else:
        cfg = load_config(args.config)
        settings = {v: settings_from_saved(v, cdir.parent / v, cfg) for v in videos}
    vdirs = [Path(settings[v][2]) if v in settings else cdir.parent / v for v in videos]

    pairs = pairs_all[pairs_all.use == "ok"]
    q = charges_both(pairs, settings, args.b)
    drops = per_drop(q)
    # одни и те же капли в обеих гистограммах: заряд значимо отличен от нуля
    drops = drops[np.abs(drops.q_raw) >= 3 * drops.err_raw].reset_index(drop=True)
    raw, cor = as_fit_frame(drops, "raw"), as_fit_frame(drops, "corr")
    e0_raw, S_raw = quantum_scan(np.abs(raw.q_over_e.values))
    e0_cor, S_cor = quantum_scan(np.abs(cor.q_over_e.values))
    w_raw = 1 / np.maximum(raw.q_over_e_err.values, 1e-3) ** 2
    w_cor = 1 / np.maximum(cor.q_over_e_err.values, 1e-3) ** 2
    fit_raw = fit_e(raw.q_over_e.values, w_raw, e0_raw, n_max=args.n_max)
    fit_cor = fit_e(cor.q_over_e.values, w_cor, e0_cor, n_max=args.n_max)
    if fit_raw["n_used"] < 5 or fit_cor["n_used"] < 5:
        med = float(np.median(np.abs(raw.q_over_e)))
        raise SystemExit(f"заряды не похожи на кратные e (медиана |q| = {med:.3g} e): проверьте константы")
    fr_raw = near_integer_fraction(raw.q_over_e.values, fit_raw["e"])
    fr_cor = near_integer_fraction(cor.q_over_e.values, fit_cor["e"])

    plot_hist(plt, raw.q_over_e.values, fit_raw["sel"], fit_raw["e"], out / "fig_q_raw.png",
              "Заряды капель по формуле Стокса")
    plot_hist(plt, cor.q_over_e.values, fit_cor["sel"], fit_cor["e"], out / "fig_q_corr.png",
              "Заряды капель с поправкой Каннингема")
    for fname, items in (("fig_qn_r_raw.png", [(raw, fit_raw, "по формуле Стокса", GREY)]),
                         ("fig_qn_r_compare.png", [(raw, fit_raw, "по формуле Стокса", GREY),
                                                   (cor, fit_cor, "с поправкой Каннингема", MAIN)])):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for d_, f_, lab, col in items:
            plot_qn_r(plt, ax, d_, f_, lab, col)
        ax.axhline(1, color="k", lw=1)
        ax.set_xlabel(r"радиус капли $r$, мкм")
        ax.set_ylabel(r"$|q|/n$, в единицах $e$")
        ax.legend(fontsize=8)
        fig.savefig(out / fname)
        plt.close(fig)

    off_by_video = pairs_all.groupby("video").off_state.first().to_dict()
    existing = [d for d in vdirs if (d / "segments.csv").exists()]
    n_off = n_on = 0
    if existing:
        n_off, n_on = plot_velocities(plt, existing, off_by_video, out / "fig_velocities.png")
    ex = pick_example(pairs_all)
    if ex:
        vid, pid = ex
        d = next((x for x in existing if x.name == vid), None)
        if d is not None and (d / "tracks.csv").exists():
            plot_zt(plt, d, pid, out / "fig_zt.png", off_by_video[vid])
    has_frame = tracking_frame(plt, existing, out / "fig_tracking.png")

    b, p = float(drops.b.iloc[0]), float(drops.p.iloc[0])
    r_med = float(np.median(cor.radius_um))
    fac = (1 + b / (p * cor.radius_um.values * 1e-6)) ** 1.5
    e_raw, e_cor = fit_raw["e"] * E_CHARGE, fit_cor["e"] * E_CHARGE
    macros = {
        "Nvideos": len(videos), "Npairs": int(len(pairs_all)), "Npairsok": int(len(pairs)),
        "Ndrops": int(len(drops)), "Nused": fit_raw["n_used"], "Nusedcorr": fit_cor["n_used"],
        "Nsegoff": n_off, "Nsegon": n_on,
        "eraw": f"{e_raw / 1e-19:.2f}", "erawerr": f"{fit_raw['err'] * E_CHARGE / 1e-19:.2f}",
        "erawdev": f"{(fit_raw['e'] - 1) * 100:+.0f}",
        "ecorr": f"{e_cor / 1e-19:.2f}", "ecorrerr": f"{fit_cor['err'] * E_CHARGE / 1e-19:.2f}",
        "ecorrdev": f"{(fit_cor['e'] - 1) * 100:+.1f}",
        "rmed": f"{r_med:.2f}", "rlo": f"{np.percentile(cor.radius_um, 10):.2f}",
        "rhi": f"{np.percentile(cor.radius_um, 90):.2f}",
        "bconst": f"{b * 1e3:.1f}", "pconst": f"{p / 1e3:.1f}",
        "corrfac": f"{np.median(fac):.2f}", "corrfaclo": f"{np.percentile(fac, 10):.2f}",
        "corrfachi": f"{np.percentile(fac, 90):.2f}",
        "cohraw": f"{S_raw:.2f}", "cohcorr": f"{S_cor:.2f}",
        "nearraw": f"{100 * fr_raw:.0f}", "nearcorr": f"{100 * fr_cor:.0f}",
    }
    lines = ["% сгенерировано make_slides_figures.py — не править вручную"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (out / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    drops.to_csv(out / "drops_both.csv", index=False, float_format="%.5g")

    print(f"готово → {out}/  (видео: {', '.join(videos)})")
    print(f"  пар всего {len(pairs_all)}, годных {len(pairs)}; капель со значимым зарядом {len(drops)}")
    print(f"  константы: " + "; ".join(f"{v}: U={settings[v][0].voltage_V} В, d={settings[v][0].plate_gap_m} м, "
                                       f"цена дел.={settings[v][1]} мм" for v in videos if v in settings))
    print(f"  по Стоксу:     шаг e* = {e_raw:.3e} Кл ({(fit_raw['e'] - 1) * 100:+.0f}% от e); "
          f"когерентность {S_raw:.2f}; у целых кратных (±0.2) {100 * fr_raw:.0f}% капель; в оценке {fit_raw['n_used']}")
    print(f"  с поправкой:   e  = {e_cor:.3e} ± {fit_cor['err'] * E_CHARGE:.1e} Кл ({(fit_cor['e'] - 1) * 100:+.1f}%); "
          f"когерентность {S_cor:.2f}; у целых кратных {100 * fr_cor:.0f}% капель; в оценке {fit_cor['n_used']}")
    print(f"  радиусы: медиана {r_med:.2f} мкм; множитель (1+b/pr)^1.5: медиана {np.median(fac):.2f} "
          f"(10–90%: {np.percentile(fac, 10):.2f}–{np.percentile(fac, 90):.2f})")
    if S_cor < 0.8 * S_raw:
        print("  ВНИМАНИЕ: поправка Каннингема РАЗМЫВАЕТ квантование (когерентность упала). Значит, расхождение "
              "с e вызвано не (только) ею — проверьте цену деления (в плоскости капель!), d, U, ρ.")
    elif S_cor > 1.1 * S_raw:
        print("  Поправка Каннингема делает квантование чётче — это подтверждает её.")
    if not has_frame:
        print("  fig_tracking.png не создан: нет review.mp4 (oiltrack review <видео> -c config.yaml)")


if __name__ == "__main__":
    main()

r"""Заряды капель из пар «поле выключено / поле включено» одной и той же капли.

Физика (метод Милликена, плоский конденсатор, закон Стокса).
Поле выключено: капля оседает со скоростью v₁, mg = k·v₁, k = 6πηa.
Поле включено: скорость v₂ против тяжести (положительная — вверх),
qE − mg = k·v₂  ⇒  qE = k·(v₁ + v₂).
Радиус из v₁: a = √(9ηv₁ / (2Δρ g)), поэтому

    q = 6πη·a·(v₁ + v₂)/E = 9π·(d/U)·√(2η³/(Δρ g))·√v₁·(v₁ + v₂).

Всё, кроме скоростей, одинаково для капель одной установки, поэтому
**q_rel = √v₁·(v₁ + v₂)** пропорционален заряду. Скорости — в делениях
шкалы окуляра в секунду (дел/с): шкала лежит в одной плоскости с
изображением капель, поэтому её шаг в пикселях меняется вместе с каплями,
когда камера стоит ближе или дальше, и q_rel из разных видео сравнимы
(при одинаковом поле — иначе см. приведение к U_ref ниже). Угол камеры
не важен: ось z для каждого видео строится перпендикулярно его линиям шкалы. Если заданы константы (d, U, η, ρ, масштаб px→м), считается и
абсолютный заряд, и q/e; по желанию — с поправкой Каннингема
η_eff = η / (1 + b/(p·a)) (у капель радиусом ~1 мкм она ~5–10 %).

Знак: q_rel > 0 — сила поля направлена вверх (капля в поле поднимается или
падает медленнее), q_rel < 0 — поле тянет вниз. Незаряженная капля даёт
v₂ ≈ −v₁, q ≈ 0.

Отличие от попарного сопоставления разных треков «вверх/вниз» по близости x:
здесь пара — это два участка ОДНОГО трека по разные стороны переключения
поля (трекер сохраняет ID капли через переключение), без перезарядки между
ними; поэтому ошибочных пар из разных капель почти нет.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ChargeConfig

log = logging.getLogger(__name__)

E_CHARGE = 1.602176634e-19


def off_state_auto(seg: pd.DataFrame, min_dur: float) -> str:
    """Режим «поле выключено» — с самым узким распределением скоростей участков (IQR)."""
    g = seg[(seg.t_end - seg.t_start >= min_dur) & np.isfinite(seg.v_z_px_s)]
    iqr = g.groupby("field_state").v_z_px_s.agg(lambda v: np.subtract(*np.percentile(v, [75, 25])))
    if len(iqr) == 0:
        raise ValueError("нет участков для определения режима «поле выключено»")
    return str(iqr.idxmin())


def fall_sign_auto(seg: pd.DataFrame, off: str, min_dur: float) -> float:
    """Знак v_z при оседании: знак медианы скоростей в режиме «поле выключено»."""
    g = seg[(seg.field_state == off) & (seg.t_end - seg.t_start >= min_dur)]
    med = float(np.median(g.v_z_px_s)) if len(g) else -1.0
    return -1.0 if med < 0 else 1.0


def q_rel(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Заряд с точностью до константы: √v₁·(v₁ + v₂); v₁ > 0 — оседание."""
    return np.sqrt(np.maximum(v1, 0)) * (v1 + v2)


def q_rel_err(v1, v2, e1, e2) -> np.ndarray:
    """Погрешность q_rel линеаризацией: ∂q/∂v₁ = (v₁+v₂)/(2√v₁) + √v₁, ∂q/∂v₂ = √v₁."""
    s = np.sqrt(np.maximum(v1, 1e-12))
    d1 = (v1 + v2) / (2 * s) + s
    d2 = s
    return np.sqrt((d1 * e1) ** 2 + (d2 * e2) ** 2)


def absolute_charge(v1_m: np.ndarray, v2_m: np.ndarray, cfg: ChargeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Абсолютный заряд, Кл, и радиус капли, м (скорости в м/с)."""
    drho = cfg.rho_oil_kg_m3 - cfg.rho_air_kg_m3
    E = cfg.voltage_V / cfg.plate_gap_m
    v1c = np.maximum(v1_m, 1e-15)
    eta = np.full_like(v1c, cfg.eta_air_Pa_s)
    a = np.sqrt(9 * eta * v1c / (2 * drho * cfg.g))
    if cfg.cunningham_b_Pa_m:
        for _ in range(20):        # итерации: η_eff зависит от a, a — от η_eff
            eta = cfg.eta_air_Pa_s / (1 + cfg.cunningham_b_Pa_m / (cfg.pressure_Pa * a))
            a = np.sqrt(9 * eta * v1c / (2 * drho * cfg.g))
    q = 6 * np.pi * eta * a * (v1_m + v2_m) / E
    return q, a


def _grid_step_from_summary(out_dir: Path) -> float | None:
    p = out_dir / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    if "grid_step_px" in s:
        return float(s["grid_step_px"])
    m = re.search(r"шаг ([\d.]+) px", s.get("calibration", ""))
    return float(m.group(1)) if m else None


def charges_for_run(out_dir: Path, cfg: ChargeConfig, min_segment_s: float,
                    mm_per_division: float | None = None) -> pd.DataFrame:
    """Заряды по одной папке вывода (pairs.csv + segments.csv). Строка = пара участков."""
    pairs = pd.read_csv(out_dir / "pairs.csv")
    seg = pd.read_csv(out_dir / "segments.csv")
    off = cfg.off_state if cfg.off_state != "auto" else off_state_auto(seg, cfg.min_segment_s)
    fall = fall_sign_auto(seg, off, cfg.min_segment_s) if cfg.fall_sign == "auto" else float(cfg.fall_sign)
    states = set(seg.field_state.astype(str))
    if off not in states:
        raise ValueError(f"режима «{off}» нет в данных ({sorted(states)})")
    # Единица скорости: деления шкалы/с. Шкала окуляра лежит в одной плоскости с
    # изображением капель, поэтому её шаг в пикселях масштабируется вместе с
    # каплями, и скорости в делениях/с сравнимы между видео, снятыми с разного
    # расстояния. Если таблицы старые (без столбцов *_div_s) — делим на средний
    # шаг сетки из summary.json; если нет и его — остаёмся в px/с с предупреждением.
    unit, div = "div", 1.0
    if "v_a_div_s" in pairs and pairs[["v_a_div_s", "v_b_div_s"]].notna().all().all():
        va, vb, ea, eb = "v_a_div_s", "v_b_div_s", "err_a_div_s", "err_b_div_s"
    else:
        va, vb, ea, eb = "v_a_px_s", "v_b_px_s", "err_a_px_s", "err_b_px_s"
        step = _grid_step_from_summary(out_dir)
        if step:
            div = float(step)
            log.info("заряды: %s — старые таблицы, скорости переведены в деления/с по среднему шагу %.1f px "
                     "(без учёта перспективы)", out_dir.name, step)
        else:
            unit = "px"
            log.warning("заряды: %s — шаг сетки неизвестен, скорости в px/с; с другими видео "
                        "(другое расстояние камеры) их сравнивать нельзя", out_dir.name)
    rows = []
    for r in pairs.itertuples(index=False):
        sa, sb = str(r.state_a), str(r.state_b)
        if (sa == off) == (sb == off):
            continue                        # оба участка в одном режиме — не пара «выкл/вкл»
        rv = r._asdict()
        A = (rv[va] / div, rv[ea] / div, r.quality_a, r.dur_a)
        B = (rv[vb] / div, rv[eb] / div, r.quality_b, r.dur_b)
        (v_off, e_off, q_off, d_off), (v_on, e_on, q_on, d_on) = (A, B) if sa == off else (B, A)
        rows.append({"video": out_dir.name, "particle_id": int(r.particle_id),
                     "segment_off": int(r.segment_a if sa == off else r.segment_b),
                     "segment_on": int(r.segment_b if sa == off else r.segment_a),
                     "t_switch": r.t_switch, "v_off": v_off, "v_on": v_on,
                     "err_off": e_off, "err_on": e_on, "dur_off": d_off, "dur_on": d_on,
                     "quality_off": q_off, "quality_on": q_on})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["off_state"] = off
    df["unit"] = unit              # div — деления шкалы/с; px — пиксели/с (несравнимо между видео)
    df["voltage_V"] = np.nan if cfg.voltage_V is None else float(cfg.voltage_V)
    df["plate_gap_m"] = np.nan if cfg.plate_gap_m is None else float(cfg.plate_gap_m)
    # v1 > 0 — оседание; v2 > 0 — движение против тяжести
    df["v1"] = fall * df.v_off
    df["v2"] = -fall * df.v_on
    reasons = []
    for r in df.itertuples(index=False):
        why = []
        if not (r.quality_off == "ok" and r.quality_on == "ok") and cfg.require_ok:
            why.append("quality")
        if r.dur_off < cfg.min_segment_s or r.dur_on < cfg.min_segment_s:
            why.append("short")
        if not np.isfinite(r.v1) or r.v1 <= 0 or r.v1 < cfg.min_v1_snr * r.err_off:
            why.append("v1_not_significant")   # оседание не измерено надёжно → радиус не определён
        reasons.append("|".join(why) if why else "ok")
    df["use"] = reasons
    df["q_rel"] = q_rel(df.v1.values, df.v2.values)
    df["q_rel_err"] = q_rel_err(df.v1.values, df.v2.values, df.err_off.values, df.err_on.values)
    # абсолютный заряд, если заданы константы и масштаб (м на единицу скорости)
    k = None
    if unit == "div" and (mm_per_division or cfg.mm_per_px):
        if mm_per_division:
            k = mm_per_division * 1e-3
        else:
            step = _grid_step_from_summary(out_dir)
            k = cfg.mm_per_px * step * 1e-3 if step else None
    elif unit == "px" and cfg.mm_per_px:
        k = cfg.mm_per_px * 1e-3
    if k and cfg.has_constants():
        v1, v2, e1, e2 = df.v1.values, df.v2.values, df.err_off.values, df.err_on.values
        q, a = absolute_charge(v1 * k, v2 * k, cfg)
        q_hi, _ = absolute_charge(v1 * k, (v2 + e2) * k, cfg)
        q1, _ = absolute_charge((v1 + e1) * k, v2 * k, cfg)
        df["radius_um"] = a * 1e6
        df["q_C"] = q
        df["q_err_C"] = np.hypot(q_hi - q, q1 - q)
        df["q_over_e"] = q / E_CHARGE
        df["q_over_e_err"] = df.q_err_C / E_CHARGE
    return df


def per_particle(pairs: pd.DataFrame) -> pd.DataFrame:
    """Средневзвешенный (1/σ²) заряд капли по всем её парам с use == ok."""
    ok = pairs[pairs.use == "ok"]
    rows = []
    for (video, pid), g in ok.groupby(["video", "particle_id"]):
        w = 1 / np.maximum(g.q_rel_err.values, 1e-9) ** 2
        q = float(np.sum(w * g.q_rel) / np.sum(w))
        sc = float(g.field_scale.iloc[0]) if "field_scale" in g else 1.0
        row = {"video": video, "particle_id": int(pid), "n_pairs": len(g),
               "voltage_V": g.voltage_V.iloc[0] if "voltage_V" in g else np.nan,
               "unit": g.unit.iloc[0],
               "v1": float(np.average(g.v1, weights=1 / g.err_off**2)),
               "q_rel": q, "q_rel_err": float(1 / np.sqrt(np.sum(w))),
               "q_rel_norm": q * sc, "q_rel_norm_err": sc / float(np.sqrt(np.sum(w)))}
        if len(g) > 1:   # согласованность пар одной капли
            row["chi2_pairs"] = float(np.sum(w * (g.q_rel - q) ** 2))
        if "q_C" in g:
            wc = 1 / np.maximum(g.q_err_C.values, 1e-30) ** 2
            qc = float(np.sum(wc * g.q_C) / np.sum(wc))
            row.update({"q_C": qc, "q_err_C": float(1 / np.sqrt(np.sum(wc))),
                        "q_over_e": qc / E_CHARGE, "q_over_e_err": float(1 / np.sqrt(np.sum(wc))) / E_CHARGE,
                        "radius_um": float(g.radius_um.mean())})
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["significant"] = (np.abs(df.q_rel) >= 3 * df.q_rel_err).astype(int)
    return df


def plot_charges(parts: pd.DataFrame, path: Path, bins: int, label: str = "q_rel") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_e = "q_over_e" in parts and parts.q_over_e.notna().all()
    col = "q_over_e" if use_e else "q_rel_norm"
    sig = parts[parts.significant == 1]
    neu = parts[parts.significant == 0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1.3]})
    ax = axes[0]
    vals = parts[col].values
    lo, hi = (np.percentile(vals, [1, 99]) if len(vals) > 10 else (vals.min(), vals.max())) if len(vals) else (-1, 1)
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist([sig[col], neu[col]], bins=edges, stacked=True, color=["#4C78A8", "#BBBBBB"],
            edgecolor="white", label=[f"заряд значим (|q| ≥ 3σ), {len(sig)}", f"не отличим от нуля, {len(neu)}"])
    ax.plot(sig[col], np.full(len(sig), 0), "|", color="k", ms=12, clip_on=False)
    if use_e:
        for n in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
            ax.axvline(n, color="#c0392b", lw=0.6, alpha=0.5)
        ax.set_xlabel("q / e")
    else:
        u = "дел/с" if (parts.unit == "div").all() else "px/с"
        ax.set_xlabel(r"$q_{rel}=\sqrt{v_1}\,(v_1+v_2)$, (" + u + r")$^{3/2}$" +
                      ("" if label == "q_rel" else "\n" + label.replace("q_rel, ", "")))
    ax.set_ylabel("число капель")
    nv = parts.video.nunique()
    ax.set_title(f"Распределение зарядов, N = {len(parts)} капель" + (f" из {nv} видео" if nv > 1 else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = axes[1]   # каждая значимая капля с погрешностью
    s = sig.sort_values(col)
    err = s["q_over_e_err"] if use_e else s["q_rel_norm_err"]
    ax2.errorbar(s[col], np.arange(len(s)), xerr=err, fmt=".", ms=3, lw=0.6, color="#4C78A8")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_yticks([])
    ax2.set_xlabel(ax.get_xlabel())
    ax2.set_title("значимые заряды с погрешностью (по возрастанию)", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


SETTINGS_FILE = "charge_settings.json"


def save_settings(out_dir: Path, cfg: ChargeConfig, mm_per_division: float | None) -> None:
    """Константы, с которыми считались заряды этого видео (напряжение и т. п.)."""
    d = dataclasses.asdict(cfg)
    d["_mm_per_division"] = mm_per_division
    (Path(out_dir) / SETTINGS_FILE).write_text(json.dumps(d, ensure_ascii=False, indent=2))


def load_settings(out_dir: Path) -> dict | None:
    p = Path(out_dir) / SETTINGS_FILE
    return json.loads(p.read_text()) if p.exists() else None


def _field_scale(pairs: pd.DataFrame, cfg: ChargeConfig) -> tuple[np.ndarray, str]:
    """Множитель, приводящий q_rel разных видео к одному полю E_ref.

    q = C·(d/U)·q_rel, поэтому при разных напряжениях сравнимо q_rel·(E_ref/E),
    E = U/d (если d не задан — считается одинаковым, E ∝ U). Возвращает
    (множитель на каждую строку, подпись для графиков).
    """
    U = pairs.voltage_V.values.astype(float)
    d = pairs.plate_gap_m.values.astype(float)
    per_video = pairs.groupby("video").voltage_V.first()
    known = per_video.notna()
    if known.sum() == 0 or per_video[known].nunique() <= 1 and known.all():
        return np.ones(len(pairs)), "q_rel"
    if not known.all():
        missing = ", ".join(per_video.index[~known])
        raise ValueError("напряжение задано не у всех видео, а у заданных оно разное — q_rel "
                         f"нельзя сравнивать. Укажите charge.voltage_V для: {missing}")
    d_eff = np.where(np.isfinite(d), d, 1.0)
    E = U / d_eff
    if cfg.reference_voltage_V is not None:
        d_ref = cfg.plate_gap_m if cfg.plate_gap_m and np.isfinite(d).all() else 1.0
        E_ref = cfg.reference_voltage_V / d_ref
        ref_txt = f"{cfg.reference_voltage_V:g} В"
    else:
        E_ref = E[0]
        ref_txt = f"{U[0]:g} В (как у {pairs.video.iloc[0]})"
    return E_ref / E, f"q_rel, приведённый к напряжению {ref_txt}"


def run_charges(out_dirs: list[Path], cfg: ChargeConfig, min_segment_s: float,
                mm_per_division: float | None, dest: Path,
                per_dir_cfg: dict | None = None) -> dict:
    """Заряды по одной или нескольким папкам вывода → charge_pairs.csv, charges.csv, гистограмма.

    ``per_dir_cfg`` — {папка: (ChargeConfig, mm_per_division)} для видео с
    разными константами (напряжением); иначе для всех берётся ``cfg``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    allp = []
    for d in out_dirs:
        c, mmd = (per_dir_cfg or {}).get(Path(d), (cfg, mm_per_division))
        p = charges_for_run(Path(d), c, min_segment_s, mmd)
        if len(p):
            log.info("заряды: %s — %d пар, режим «поле выключено» = %s, годных %d, U = %s",
                     Path(d).name, len(p), p.off_state.iloc[0], int((p.use == "ok").sum()),
                     "?" if c.voltage_V is None else f"{c.voltage_V:g} В")
            allp.append(p)
    if not allp:
        log.warning("заряды: нет пар «поле выключено / включено»")
        return {"pairs": 0, "particles": 0}
    pairs = pd.concat(allp, ignore_index=True)
    px_videos = sorted(pairs.loc[pairs.unit == "px", "video"].unique())
    if px_videos and pairs.video.nunique() > 1:
        raise ValueError("у видео " + ", ".join(px_videos) + " не найден шаг сетки, скорости остались в px/с; "
                         "камера на разных видео стоит по-разному, поэтому смешивать их с другими видео "
                         "нельзя. Задайте calibration.grid_step_px для этих видео (--set или videos.yaml)")
    scale, label = _field_scale(pairs, cfg)
    pairs["field_scale"] = scale
    pairs["q_rel_norm"] = pairs.q_rel * scale
    pairs["q_rel_norm_err"] = pairs.q_rel_err * scale
    parts = per_particle(pairs)
    pairs.to_csv(dest / "charge_pairs.csv", index=False, float_format="%.6g")
    parts.sort_values("q_rel_norm").to_csv(dest / "charges.csv", index=False, float_format="%.6g")
    if len(parts):
        plot_charges(parts, dest / "charges_hist.png", cfg.hist_bins, label)
    res = {"pairs": int(len(pairs)), "pairs_used": int((pairs.use == "ok").sum()),
           "particles": int(len(parts)), "significant": int(parts.significant.sum()) if len(parts) else 0,
           "absolute": bool("q_C" in parts and parts.q_C.notna().all()), "histogram": label}
    log.info("заряды: %s → %s", res, dest)
    return res

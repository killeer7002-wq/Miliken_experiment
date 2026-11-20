"""Заряды: формула q_rel, абсолютный заряд и сборка пар из pairs.csv/segments.csv."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.charges import E_CHARGE, absolute_charge, q_rel, q_rel_err, run_charges
from src.config import ChargeConfig


def _cfg() -> ChargeConfig:
    return ChargeConfig(plate_gap_m=6e-3, voltage_V=300.0, rho_oil_kg_m3=900.0, mm_per_px=1e-3)


def test_absolute_charge_roundtrip():
    """Капля с известными радиусом и зарядом → скорости → обратно тот же заряд."""
    c = _cfg()
    a, n = 0.8e-6, 3
    drho = c.rho_oil_kg_m3 - c.rho_air_kg_m3
    k = 6 * np.pi * c.eta_air_Pa_s * a
    v1 = (4 / 3) * np.pi * a**3 * drho * c.g / k                 # оседание
    v2 = n * E_CHARGE * c.voltage_V / c.plate_gap_m / k - v1      # движение в поле
    q, a_est = absolute_charge(np.array([v1]), np.array([v2]), c)
    assert abs(q[0] / E_CHARGE - n) < 1e-6 and abs(a_est[0] - a) / a < 1e-6


def test_q_rel_matches_formula_and_sign():
    assert np.isclose(q_rel(np.array([4.0]), np.array([5.0]))[0], 2.0 * 9.0)
    assert q_rel(np.array([4.0]), np.array([-4.0]))[0] == 0          # незаряженная: v2 = −v1
    assert q_rel(np.array([4.0]), np.array([-9.0]))[0] < 0           # поле тянет вниз
    e = q_rel_err(np.array([4.0]), np.array([5.0]), np.array([0.1]), np.array([0.0]))[0]
    num = (q_rel(np.array([4.1]), np.array([5.0]))[0] - q_rel(np.array([3.9]), np.array([5.0]))[0]) / 2
    assert abs(e - num) < 1e-3


def test_run_charges_on_tables():
    """Пары из таблиц: режим «выключено» и знак оседания определяются сами."""
    rng = np.random.default_rng(0)
    seg_rows, pair_rows = [], []
    for pid in range(60):
        v1 = rng.uniform(1.5, 4.0)
        n = rng.integers(-3, 4)
        v_on_up = n * 10.0 / np.sqrt(v1) - v1        # q_rel = 10·n
        v_off, v_on = -v1, v_on_up                    # z вверх: оседание отрицательно
        seg_rows += [dict(particle_id=pid, segment_id=0, t_start=0, t_end=8, field_state="A", v_z_px_s=v_off),
                     dict(particle_id=pid, segment_id=1, t_start=8, t_end=16, field_state="B", v_z_px_s=v_on)]
        pair_rows.append(dict(particle_id=pid, segment_a=0, segment_b=1, state_a="A", state_b="B", t_switch=8.0,
                              v_a_px_s=v_off, v_b_px_s=v_on, err_a_px_s=0.05, err_b_px_s=0.05,
                              dur_a=8.0, dur_b=8.0, quality_a="ok", quality_b="ok"))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d) / "vid"
        d.mkdir()
        pd.DataFrame(seg_rows).to_csv(d / "segments.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(d / "pairs.csv", index=False)
        res = run_charges([d], ChargeConfig(), 1.0, None, d)
        ch = pd.read_csv(d / "charges.csv")
        cp = pd.read_csv(d / "charge_pairs.csv")
    assert res["particles"] == 60 and cp.off_state.iloc[0] == "A"
    assert np.allclose(np.round(ch.q_rel / 10), ch.q_rel / 10, atol=0.05)


def _make_dir(root: Path, name: str, U: float, seed: int) -> Path:
    """Таблицы видео при напряжении U: одинаковые заряды n·q₀, v₁ + v₂ ∝ q·U."""
    rng = np.random.default_rng(seed)
    seg_rows, pair_rows = [], []
    for pid in range(50):
        v1 = rng.uniform(1.5, 4.0)
        n = int(rng.integers(1, 4)) * int(rng.choice([-1, 1]))
        v_on = n * 0.02 * U / np.sqrt(v1) - v1
        seg_rows += [dict(particle_id=pid, segment_id=0, t_start=0, t_end=8, field_state="A", v_z_px_s=-v1),
                     dict(particle_id=pid, segment_id=1, t_start=8, t_end=16, field_state="B", v_z_px_s=v_on)]
        pair_rows.append(dict(particle_id=pid, segment_a=0, segment_b=1, state_a="A", state_b="B", t_switch=8.0,
                              v_a_px_s=-v1, v_b_px_s=v_on, err_a_px_s=0.05, err_b_px_s=0.05,
                              dur_a=8.0, dur_b=8.0, quality_a="ok", quality_b="ok"))
    d = root / name
    d.mkdir()
    pd.DataFrame(seg_rows).to_csv(d / "segments.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(d / "pairs.csv", index=False)
    (d / "summary.json").write_text('{"grid_step_px": 1.0}')
    return d


def test_different_voltages_are_normalized():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = _make_dir(root, "v500", 500, 1), _make_dir(root, "v1000", 1000, 2)
        per = {a: (ChargeConfig(voltage_V=500), None), b: (ChargeConfig(voltage_V=1000), None)}
        res = run_charges([a, b], ChargeConfig(), 1.0, None, root / "all", per)
        ch = pd.read_csv(root / "all" / "charges.csv")
    # приведено к 500 В: q_rel_norm = 10·n для обоих видео
    assert "500" in res["histogram"]
    assert np.allclose(ch.q_rel_norm / 10, np.round(ch.q_rel_norm / 10), atol=0.05)
    raw_b = ch[ch.video == "v1000"].q_rel.abs().min()
    assert raw_b > 15        # без приведения у второго видео q_rel вдвое больше


def test_mixed_unknown_voltage_is_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b, c = _make_dir(root, "a", 500, 1), _make_dir(root, "b", 1000, 2), _make_dir(root, "c", 700, 3)
        per = {a: (ChargeConfig(voltage_V=500), None), b: (ChargeConfig(voltage_V=1000), None),
               c: (ChargeConfig(), None)}
        try:
            run_charges([a, b, c], ChargeConfig(), 1.0, None, root / "all", per)
        except ValueError as e:
            assert "c" in str(e)
        else:
            raise AssertionError("ожидалась ошибка: у одного видео не задано напряжение")


def test_camera_distance_does_not_change_charges():
    """Одни и те же капли, камера ближе (шаг сетки 80 px) и дальше (40 px): q_rel совпадают."""
    rng = np.random.default_rng(7)
    v1_div = rng.uniform(0.03, 0.08, 40)           # деления/с
    n = rng.integers(1, 4, 40) * rng.choice([-1, 1], 40)
    v_on_div = n * 0.01 / np.sqrt(v1_div) - v1_div
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, step in (("near", 80.0), ("far", 40.0)):
            seg_rows, pair_rows = [], []
            for pid in range(40):
                vo, vn = -v1_div[pid] * step, v_on_div[pid] * step       # px/с на этом видео
                seg_rows += [dict(particle_id=pid, segment_id=0, t_start=0, t_end=8, field_state="A", v_z_px_s=vo),
                             dict(particle_id=pid, segment_id=1, t_start=8, t_end=16, field_state="B", v_z_px_s=vn)]
                pair_rows.append(dict(particle_id=pid, segment_a=0, segment_b=1, state_a="A", state_b="B",
                                      t_switch=8.0, v_a_px_s=vo, v_b_px_s=vn, err_a_px_s=0.01 * step,
                                      err_b_px_s=0.01 * step, v_a_div_s=vo / step, v_b_div_s=vn / step,
                                      err_a_div_s=0.01, err_b_div_s=0.01,
                                      dur_a=8.0, dur_b=8.0, quality_a="ok", quality_b="ok"))
            d = root / name
            d.mkdir()
            pd.DataFrame(seg_rows).to_csv(d / "segments.csv", index=False)
            pd.DataFrame(pair_rows).to_csv(d / "pairs.csv", index=False)
            run_charges([d], ChargeConfig(min_v1_snr=0), 1.0, None, d)
            out[name] = pd.read_csv(d / "charges.csv").sort_values("particle_id").q_rel.values
    assert np.allclose(out["near"], out["far"])


def test_perspective_step_model():
    """Шаг сетки линейно меняется поперёк поля — модель находит наклон, z_div считает деления."""
    from src.calibration import Calibration, local_step_model

    a_true, b_true = 50.0, 0.03                    # шаг 50 px в центре, ±9 px на краях
    pos = [0.0]
    for _ in range(6):
        pos.append(pos[-1] + a_true + b_true * (pos[-1] + 25))
    neg = [0.0]
    for _ in range(6):
        neg.append(neg[-1] - (a_true + b_true * (neg[-1] - 25)))
    lines = np.array(sorted(set(neg + pos)))
    a, b, n = local_step_model(lines, 50.0, 330.0, True)
    assert n == len(lines) and abs(b - b_true) < 0.005 and abs(a - a_true) < 1.0
    cal = Calibration(12.0, 50.0, (50, 50), 0.0, 0.0, 1, None, lines, "test", a, b, n)
    zdiv = cal.to_div(lines)
    assert np.allclose(np.diff(zdiv), 1.0, atol=0.03)      # между соседними линиями — одно деление

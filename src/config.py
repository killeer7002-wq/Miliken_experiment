"""Конфигурация: dataclass-структура, загрузка и проверка YAML.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class RoiConfig:
    auto: bool = True
    erode_px: int = 15
    n_frames: int = 30            # сколько кадров брать для медианы
    # ручное задание круга: [cx, cy, R] (используется, если auto: false)
    circle: list[float] | None = None


@dataclass
class BackgroundConfig:
    window_s: float = 10.0        # ширина окна скользящей статистики, с
    sample_every: int = 5         # брать в фон каждый N-й кадр
    update_every_s: float = 1.0   # шаг пересчёта фона, между узлами линейная интерполяция
    percentile: float = 20.0


@dataclass
class DetectConfig:
    channel: str = "green"
    sigmas: list[float] = dc_field(default_factory=lambda: [1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0, 16.0, 20.0])
    threshold: str | float = "auto"   # Если "auto", то k_mad * MAD
    k_mad: float = 7.0
    ring_radii: list[float] = dc_field(default_factory=lambda: [8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 26.0])
    width: float = 1.0                # ширина кольца у кольцевого фильтра
    k_mad_annulus: float = 6.0        # порог кольцевого фильтра (в робастных σ)
    ring_min_snr: float = 3.0         # яркость обода >= ring_min_snr * шум пикселя
    k_mad_ring: float = 3.5           # порог для масштабов колец (r = σ√2 >= ring_min_r)
    merge_factor: float = 1.0
    coarse_sigma: float = 7.0         # σ >= этого считаются на изображении, уменьшенном вдвое
    ring_min_r: float = 4.5           # если r ≥ этого, то kind = 'ring', иначе 'point'
    min_intensity: float = 3.0        # минимальная яркость остатка в центре/на кольце (уровни серого)
    max_edge_ratio: float = 3.0       # отношение главных кривизн гессиана; больше — «линия», не капля
    min_contrast: float = 6.0         # точка: центр минус медиана кольца [r+2, r+4] вокруг, уровни серого
    border_px: int = 3                # не искать ближе к краю маски
    workers: int = 0


@dataclass
class CalibrationConfig:
    grid_angle_deg: str | float = "auto"
    grid_step_px: str | float = "auto"
    expected_angle_deg: float = 12.0
    expected_step_px: float = 51.0
    angle_tolerance_deg: float = 6.0
    step_tolerance_px: float = 15.0
    mm_per_division: float | None = None
    perspective: bool = True          # учитывать изменение шага сетки поперёк поля (наклон камеры)
    # направление оси z: +1 — z растёт «вниз по кадру» (окуляр переворачивает
    # изображение, поэтому это «вверх» в камере; оседание даёт v_z < 0),
    # -1 — z растёт «вверх по кадру».
    z_sign: int = 1


@dataclass
class FieldConfig:
    switches_s: list[float] | None = None
    states: list[str] | None = None
    auto_refine: bool = True
    # авто-поиск
    auto_window_s: float = 0.33
    auto_lag_frames: int = 3
    auto_gate_px: float = 6.0
    auto_fast_px_s: float = 12.0
    auto_penalty: float | str = "auto"
    auto_min_size_s: float = 2.0
    auto_min_jump_sigma: float = 8.0   # скачок медиан по обе стороны ≥ стольких σ шума ряда
    track_window_s: float = 0.7        # окна слева/справа для знаконезависимого признака по трекам
    auto_track_k_mad: float = 12.0     # порог пика: медиана + k·MAD
    refine_max_shift_s: float = 1.0
    refine_min_tracks: int = 5


@dataclass
class LinkConfig:
    gate_px: float = 4.0
    gate_after_switch_px: float = 6.0
    switch_reset_s: float = 0.5
    max_gap_frames: int = 10
    gate_r_factor: float = 0.4         # гейт = gate_px + gate_r_factor·r (кольца и смена точка↔кольцо)
    stitch_max_gap: int = 60           # склейка обрывков: макс. разрыв, кадров (капля под линией шкалы до ~2 с)
    stitch_gate_px: float = 4.0        # склейка обрывков: рассогласование экстраполяций при нулевом разрыве
    stitch_gate_px_per_s: float = 2.0  # ... плюс столько px на каждую секунду разрыва
    stitch_fit_points: int = 15        # точек на краю обрывка для оценки скорости
    w_r: float = 2.0
    w_i: float = 1.0
    sigma_loc_px: float = 0.5          # ошибка измерения координаты (Калман R)
    diffusion_px2_s: float = 3.0       # D для шума процесса (Калман Q, позиция)
    accel_px_s2: float = 5.0           # шум процесса по скорости (медленные изменения)
    v_init_sigma_px_s: float = 15.0    # неопределённость скорости у нового трека и после переключения
    min_track_frames_keep: int = 5     # короче — трек не сохраняется вовсе (шум детектора)


@dataclass
class SegmentConfig:
    mode: str = "stop"                 # stop | split
    min_piece_s: float = 1.0
    k_sigma: float | str = "auto"     # порог z смены скорости; auto — по нулевой координате s
    k_sigma_floor: float = 4.0         # нижняя граница автопорога
    null_fa_rate: float = 0.01         # доля ложных изломов на кусок для автопорога
    msd_max_lag: int = 5               # лаги (кадров) для D участка в segments.csv
    d_floor_frac: float = 0.5          # D трека не ниже этой доли от медианного D по трекам
    noise_max_lag_s: float = 1.0       # лаги до стольких секунд для шума при поиске перезарядок
    max_breaks_per_interval: int = 5
    skip_after_switch_s: float = 0.0   # отбросить из подгонки N секунд после переключения (переходный процесс)


@dataclass
class QualityConfig:
    min_track_s: float = 2.0
    max_radius_growth: float = 1.5
    defocus_min_px: float = 2.0        # и рост радиуса не меньше стольких px
    flag_fraction: float = 0.2         # флаг строки переносится на участок, если им помечено > 20% строк
    edge_factor: float = 2.0           # edge: ближе edge_factor * r к краю маски
    min_segment_s: float = 1.0         # для pairs.csv/сводки


@dataclass
class ChargeConfig:
    enabled: bool = True
    off_state: str = "auto"             # режим «поле выключено»; auto — с самым узким распределением скоростей
    fall_sign: str | float = "auto"     # знак v_z при оседании; auto — по медиане в режиме «поле выключено»
    min_segment_s: float = 2.0          # оба участка пары не короче, с
    require_ok: bool = True             # оба участка без флагов качества
    min_v1_snr: float = 2.0             # скорость оседания ≥ min_v1_snr·σ (иначе радиус не определён)
    hist_bins: int = 30
    # константы установки для абсолютного заряда (null → только q_rel)
    mm_per_px: float | None = None      # null → calibration.mm_per_division / шаг сетки
    plate_gap_m: float | None = None    # расстояние между пластинами конденсатора d, м
    voltage_V: float | None = None      # напряжение U, В (своё у каждого видео — через --set или batch)
    reference_voltage_V: float | None = None  # к какому U приводить q_rel при объединении (null — U первого видео)
    eta_air_Pa_s: float = 1.81e-5
    rho_oil_kg_m3: float | None = None
    rho_air_kg_m3: float = 1.2
    g: float = 9.81
    cunningham_b_Pa_m: float | None = None   # например 8.2e-3; null — без поправки
    pressure_Pa: float = 101325.0

    def has_constants(self) -> bool:
        return None not in (self.plate_gap_m, self.voltage_V, self.rho_oil_kg_m3)


@dataclass
class OutputConfig:
    dir: str = "output"
    per_particle_csv: bool = True
    plots: bool = True
    max_plots: int = 200
    review_video: bool = True
    max_review_tracks: int = 60
    review_tail_s: float = 2.0
    review_scale: float = 1.0


@dataclass
class Config:
    video: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    roi: RoiConfig = dc_field(default_factory=RoiConfig)
    background: BackgroundConfig = dc_field(default_factory=BackgroundConfig)
    detect: DetectConfig = dc_field(default_factory=DetectConfig)
    calibration: CalibrationConfig = dc_field(default_factory=CalibrationConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    link: LinkConfig = dc_field(default_factory=LinkConfig)
    segment: SegmentConfig = dc_field(default_factory=SegmentConfig)
    quality: QualityConfig = dc_field(default_factory=QualityConfig)
    charge: ChargeConfig = dc_field(default_factory=ChargeConfig)
    output: OutputConfig = dc_field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _fill(cls: type, data: dict[str, Any] | None, path: str) -> Any:
    """Рекурсивно создать dataclass ``cls`` из словаря, ругаясь на лишние ключи."""
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"секция '{path}' должна быть словарём, а не {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"неизвестный параметр '{path}.{key}'")
        sub = _SECTIONS.get(key) if cls is Config else None
        if sub is not None:
            kwargs[key] = _fill(sub, value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_SECTIONS: dict[str, type] = {
    "charge": ChargeConfig,
    "roi": RoiConfig,
    "background": BackgroundConfig,
    "detect": DetectConfig,
    "calibration": CalibrationConfig,
    "field": FieldConfig,
    "link": LinkConfig,
    "segment": SegmentConfig,
    "quality": QualityConfig,
    "output": OutputConfig,
}


def validate(cfg: Config) -> None:
    """Проверить согласованность параметров. Бросает ValueError."""
    if cfg.segment.k_sigma != "auto" and not isinstance(cfg.segment.k_sigma, (int, float)):
        raise ValueError("segment.k_sigma: число или 'auto'")
    if cfg.segment.mode not in ("stop", "split"):
        raise ValueError("segment.mode должен быть 'stop' или 'split'")
    if cfg.field.switches_s is not None:
        sw = list(cfg.field.switches_s)
        if sorted(sw) != sw:
            raise ValueError("field.switches_s должен быть отсортирован по возрастанию")
        if cfg.field.states is not None and len(cfg.field.states) != len(sw) + 1:
            raise ValueError("field.states должен содержать len(switches_s)+1 подписей")
    if not cfg.detect.sigmas or min(cfg.detect.sigmas) <= 0:
        raise ValueError("detect.sigmas: нужен непустой список положительных σ")
    if not 0 < cfg.background.percentile <= 100:
        raise ValueError("background.percentile должен быть в (0, 100]")
    if cfg.calibration.z_sign not in (1, -1):
        raise ValueError("calibration.z_sign: 1 или -1")
    if cfg.link.gate_px <= 0 or cfg.link.gate_after_switch_px < cfg.link.gate_px:
        raise ValueError("link: нужно 0 < gate_px <= gate_after_switch_px")


def deep_merge(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно наложить upd на base (секции сливаются, значения заменяются)."""
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_set(items: list[str] | None) -> dict[str, Any]:
    """``["charge.voltage_V=700", "field.switches_s=[10, 20]"]`` → вложенный словарь.

    Значение разбирается как YAML: числа, списки, null, true/false.
    """
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set ожидает ключ=значение, получено {it!r}")
        key, val = it.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p_ in parts[:-1]:
            node = node.setdefault(p_, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(path: str | Path | list[str | Path] | None,
                overrides: dict[str, Any] | None = None) -> Config:
    """Загрузить один или несколько YAML (каждый следующий накладывается на
    предыдущий) и применить переопределения (вложенный словарь; None-значения
    верхнего уровня пропускаются)."""
    data: dict[str, Any] = {}
    paths = [] if path is None else (path if isinstance(path, (list, tuple)) else [path])
    for p_ in paths:
        with open(p_, encoding="utf-8") as fh:
            data = deep_merge(data, yaml.safe_load(fh) or {})
    data = deep_merge(data, {k: v for k, v in (overrides or {}).items() if v is not None})
    cfg = _fill(Config, data, "")
    validate(cfg)
    return cfg


def field_states(cfg: FieldConfig, n_switches: int) -> list[str]:
    """Подписи режимов поля: из конфига или по умолчанию A, B, A, B, ..."""
    if cfg.states is not None and len(cfg.states) == n_switches + 1:
        return [str(s) for s in cfg.states]
    return ["A" if i % 2 == 0 else "B" for i in range(n_switches + 1)]

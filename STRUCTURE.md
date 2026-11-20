## Структура проекта

```

Miliken_experiments/
  ├── pyproject.toml - Зависимости
  ├── config.yaml - Все переменные
  ├── README.md
  ├── STRUCTURE.md
  ├── src/
  │   ├── config.py
  │   ├── io_video.py         # Разделение видео на кадры
  │   ├── roi.py              # Ограничение интересующей нас области.
  │   ├── detect.py           # Детекция капель масла на видео
  │   ├── calibration.py      # Поиск угла и шага сетки, затем поворот осей и масштаб 
  │   ├── background.py       # Удаление фона
  │   ├── changepoints.py     # Поиск точек излома
  │   ├── charges.py          # Поиск зарядов
  │   ├── export.py
  │   ├── field_switch.py     # Поиск моментов переключения полей
  │   ├── link.py             # Сопоставление детекций в треки
  │   ├── quality.py          # Флаги качества
  │   ├── review.py           # Графики и генерация видео для ревью
  │   ├── synth.py            # Генерация синтетических данных
  ├── tests/
  ├── outputs/<name_video>/
  ├── media/
  │

```
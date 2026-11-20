"""Простой раннер тестов без pytest: ``python tests/run_tests.py [-k подстрока] [--slow]``.

Тесты написаны в стиле pytest (функции test_*), поэтому ``pytest tests`` тоже
работает. Медленные тесты (с рендером синтетического видео) помечены
атрибутом ``slow = True`` и запускаются только с ``--slow``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("--slow", action="store_true")
    args = ap.parse_args()
    failed = passed = skipped = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(f.stem)
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if not (name.startswith("test_") and callable(fn)):
                continue
            full = f"{f.stem}::{name}"
            if args.k and args.k not in full:
                continue
            if getattr(fn, "slow", False) and not args.slow:
                skipped += 1
                print(f"SKIP  {full} (медленный, --slow)")
                continue
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"PASS  {full}  ({time.time() - t0:.1f} с)")
            except Exception:
                failed += 1
                print(f"FAIL  {full}  ({time.time() - t0:.1f} с)")
                traceback.print_exc()
    print(f"\nитого: {passed} прошло, {failed} упало, {skipped} пропущено")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

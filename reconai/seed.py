"""Load demo data. `python -m reconai.seed [--force]`"""
from __future__ import annotations

import argparse

from . import db
from .generate_data import generate


def ensure_demo_data(engine=None, force: bool = False) -> bool:
    e = engine or db.get_engine()
    if not force and db.has_data(e):
        return False
    frames, _ = generate()
    db.load_frames(frames, e)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    seeded = ensure_demo_data(force=ap.parse_args().force)
    print("Demo data loaded." if seeded else "Data already present (use --force to regenerate).")

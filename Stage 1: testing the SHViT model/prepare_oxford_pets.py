"""prepare_oxford_pets.py (Stage 1 thin wrapper)

Stage 1 used to ship its own Food-101 preparation script. The Oxford Pets
preparation lives at the repo root (../prepare_oxford_pets.py) and is shared by
every stage; this file is kept as a thin convenience entry point so the Stage 1
notebook can call `python prepare_oxford_pets.py` without leaving its folder.

Usage:
    python prepare_oxford_pets.py [--root data] [--seed 1] [--force-generate-split]
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Delegate to the root script.
from prepare_oxford_pets import main  # noqa: E402


if __name__ == "__main__":
    main()

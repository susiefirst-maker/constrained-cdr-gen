"""End-to-end Phase 5 demo on Trastuzumab CDR-H3.

Runs the full sampler + oracle + metrics pipeline and prints a
human-readable summary. A thin wrapper around `cdr_gen.cli` kept for
quick manual inspection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cdr_gen.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main([
        "--parent", "trastuzumab",
        "--n-designs", "200",
        "--n-sweeps", "10",
        "--strategies", "random", "unguided", "guided",
        "--lambda", "1.0",
        "--out-dir", "reports",
    ]))

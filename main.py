from __future__ import annotations

from pathlib import Path
import sys

# Keep src layout runnable from root without requiring packaging steps.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from not_an_llm.cli import main


if __name__ == "__main__":
    main()

"""Command-line entry point; run from the project directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mnist_assignment.cli import main

if __name__ == "__main__":
    main()

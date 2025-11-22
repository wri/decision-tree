import os
from pathlib import Path


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    # Start from the directory of the current file
    return str(Path(os.path.dirname(__file__)).parent)

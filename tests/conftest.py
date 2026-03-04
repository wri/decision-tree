# conftest.py
from pathlib import Path

# Block any collection under paths containing these names
BLOCKED = {"tm-api-utils", "terramatch-researcher-api"}

def pytest_ignore_collect(path=None, config=None, collection_path=None):
    """
    Compatible with pytest <8 (path: py.path.local) and pytest >=8
    (collection_path: pathlib.Path). Will work when pytest 9 removes 'path'.
    """
    # Prefer the new argument when available (pytest >= 8)
    if collection_path is not None:
        p = collection_path  # already a pathlib.Path
    else:
        # Fallback for pytest < 8
        # 'path' is a py.path.local and can be cast to str safely
        p = Path(str(path)) if path is not None else None

    if p is None:
        return False  # nothing to decide on

    # If any of the blocked folder names appear in the path, skip collecting
    parts = set(p.parts)
    return bool(BLOCKED & parts)

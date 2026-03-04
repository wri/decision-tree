# conftest.py
from pathlib import Path

# Block any directory that contains the remote repo checkout
BLOCKED = {"terramatch-researcher-api", "tm-api-utils"}

def pytest_ignore_collect(path=None, config=None, collection_path=None):
    # pytest >= 8 passes collection_path: Path
    if collection_path is not None:
        p = collection_path
    else:
        # pytest < 8 passes path: py.path.local (stringify it)
        p = Path(str(path)) if path else None

    if not p:
        return False

    # Skip collecting tests if the repo name appears anywhere in the path
    return any(part in BLOCKED for part in p.parts)

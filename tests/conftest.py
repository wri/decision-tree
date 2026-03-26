# conftest.py
import os
from pathlib import Path

from shared_library.os_tools import get_project_root_dir

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

PROJECT_ROOT = get_project_root_dir()
SECRETS_FILE_PATH = os.path.join(PROJECT_ROOT, "secrets.yaml")
DT_TEST_DATA_SUB_PATH = os.path.join('tests', 'data', 'decision_tree')
DT_TEST_DATA_DIR = os.path.join(PROJECT_ROOT, DT_TEST_DATA_SUB_PATH)
DT_TEST_PROJECTS = os.path.join(DT_TEST_DATA_DIR, "test_projects")
DT_TEST_PARAMS_DIR = os.path.join(DT_TEST_DATA_DIR, "param_files")

TEST_01_GRI_PROJECT_ID = '1826cc5f-0d4d-4427-b5b3-fe244deba919'

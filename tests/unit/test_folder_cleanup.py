"""Tests for the safety guard in tests/tools.py::folder_cleanup()."""

import os
import shutil

import pytest
import yaml
from gri_shared_library.os_tools import get_project_root_dir

from tools import folder_cleanup

GUARD_TEST_FOLDER = os.path.join("tests", "data", "_folder_cleanup_guard_test")


def _params_file(tmp_path, project_data_folder):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"outfile": {"project_data_folder": project_data_folder}}))
    return str(path)


@pytest.mark.parametrize("project_data_folder", [
    ".",                      # project root
    "data",                   # real data folder
    "tests",                  # parent of tests/data
    os.path.join("tests", "data"),                         # tests/data itself
    os.path.join("tests", "data", "..", "..", "data"),     # escapes via ..
])
def test_refuses_folders_outside_test_data(tmp_path, project_data_folder):
    with pytest.raises(ValueError, match="Refusing to clean"):
        folder_cleanup(_params_file(tmp_path, project_data_folder))


def test_cleans_scratch_and_keeps_keeper_folders(tmp_path):
    data_dir = os.path.join(get_project_root_dir(), GUARD_TEST_FOLDER)
    try:
        for name in ["imagery_availability", "tree_output", "decision_scores", "feats"]:
            os.makedirs(os.path.join(data_dir, name), exist_ok=True)

        folder_cleanup(_params_file(tmp_path, GUARD_TEST_FOLDER))

        assert sorted(os.listdir(data_dir)) == ["imagery_availability", "tree_output"]
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)

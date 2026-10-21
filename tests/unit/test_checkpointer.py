"""Unit tests for the Checkpointer class in run_decision_tree.py."""

import pandas as pd
import pytest

from decision_tree.run_decision_tree import Checkpointer


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def _cp(tmp_path, enabled):
    return Checkpointer(enabled=enabled, paths={"k": str(tmp_path / "nested" / "dir" / "k.csv")})


def test_disabled_save_writes_nothing(tmp_path, frame):
    cp = _cp(tmp_path, enabled=False)
    cp.save("k", frame)
    assert not cp.exists("k")


def test_disabled_save_with_always_writes(tmp_path, frame):
    cp = _cp(tmp_path, enabled=False)
    cp.save("k", frame, always=True)
    assert cp.exists("k")


def test_enabled_save_creates_folders_and_round_trips(tmp_path, frame):
    cp = _cp(tmp_path, enabled=True)
    cp.save("k", frame)
    pd.testing.assert_frame_equal(cp.load("k"), frame)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No file found for 'k'"):
        _cp(tmp_path, enabled=True).load("k")


def test_exists_unknown_key_is_false(tmp_path):
    assert _cp(tmp_path, enabled=True).exists("unknown") is False

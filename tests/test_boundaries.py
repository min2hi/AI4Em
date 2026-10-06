import pytest

from src.config import validate_config
from src.datasets.acquisition import download_member, select_subset


@pytest.mark.parametrize("config", [
    {"yawn_enter": float("nan"), "yawn_exit": 0.2},
    {"yawn_enter": 0.5, "yawn_exit": -0.1},
    {"smoothing_samples": 2.5},
    {"num_layers": 1.5},
])
def test_invalid_event_threshold_or_integer_setting_is_rejected(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_acquisition_does_not_allow_paths_that_escape_project_raw_directory(tmp_path):
    # Windows drive syntax must be rejected even on non-Windows CI.
    record = {"relative_path": "01/C:\\outside.mp4"}
    with pytest.raises(ValueError):
        download_member(record, tmp_path, reserve_bytes=0)


def test_subset_limit_cannot_be_increased_above_one_quarter():
    with pytest.raises(ValueError):
        select_subset([], subjects_per_fold=4)

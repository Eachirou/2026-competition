"""Unit tests for the pure parts of nb_task.

Run with ``pytest benchmark_utils``.

The loading pipeline itself needs the neuralbench stack and real data — it is
validated on the cluster; these tests cover the subset-filter composition,
which is plain dict/string logic, and the windows' sampling rate.
"""

import pytest

from benchmark_utils.nb_task import _window_sfreq, build_test_only_filter


def test_predefined_split_with_query_and_existing_filter():
    # xu2025alljoined shape: explicit test query + a shipped filter_stimuli.
    cfg = {
        "study": {
            "split": {"name": "PredefinedSplit",
                      "test_split_query": "label == 'stim_test'"},
            "filter_stimuli": {
                "name": "QueryEvents",
                "query": "type != 'Image' or label in "
                         "['stim_train', 'stim_test']",
            },
        },
        "trigger_event_type": "Image",
    }
    out = build_test_only_filter(cfg)
    assert out["name"] == "QueryEvents"
    assert out["query"] == (
        "(type != 'Image' or label in ['stim_train', 'stim_test']) and "
        "(type not in ['Image'] or (label == 'stim_test'))"
    )


def test_predefined_split_on_data_column():
    # gifford shape: the split column ships with the source data.
    cfg = {
        "study": {"split": {"name": "PredefinedSplit",
                            "test_split_query": None, "col_name": "split"}},
        "trigger_event_type": "Image",
    }
    assert build_test_only_filter(cfg)["query"] == (
        "type not in ['Image'] or (split == 'test')"
    )


def test_runtime_split_is_rejected():
    cfg = {
        "study": {"split": {"name": "SklearnSplit"}},
        "trigger_event_type": "Stimulus",
    }
    with pytest.raises(ValueError, match="PredefinedSplit"):
        build_test_only_filter(cfg)


def test_missing_split_is_rejected():
    with pytest.raises(ValueError, match="PredefinedSplit"):
        build_test_only_filter({"study": {}, "trigger_event_type": "Stimulus"})


def test_window_sfreq_is_the_extractor_rate():
    assert _window_sfreq(120.0, 480, 4.0) == 120.0


def test_window_sfreq_at_native_rate_is_read_off_the_window():
    # Muse: 5-s windows of 640 samples at the recorded 128 Hz.
    assert _window_sfreq("native", 640, 5.0) == 128.0


def test_window_sfreq_at_native_rate_needs_a_duration():
    with pytest.raises(ValueError, match="duration"):
        _window_sfreq("native", 640, None)

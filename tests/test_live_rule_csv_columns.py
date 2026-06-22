"""Recording-CSV binary column tests for ROI/behavior trigger rules.

Setting a rule must add a clearly-labelled binary column to the recording export
that is 1 on frames where the rule's gated condition is ON and 0 otherwise, and
the column must only exist when at least one rule is present (so it appears in a
recording's CSV only when active during that recording).
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYKABOO_SHOW_SIMULATED_CAMERAS", "1")

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    pass

QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication


@pytest.fixture(scope="module")
def main_window():
    app = QApplication.instance() or QApplication([])
    try:
        import main_window_enhanced as mwe
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GUI deps unavailable: {exc}")
    try:
        window = mwe.MainWindow()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MainWindow could not be constructed headless: {exc}")
    window._startup_autoconnect_done = True
    yield window
    try:
        window.close()
    except Exception:
        pass


def _rules():
    from live_detection_types import LiveTriggerRule

    return [
        LiveTriggerRule(rule_id="r1", rule_type="roi_occupancy", output_id="DO1",
                        roi_name="Left Chamber", mouse_id=1),
        LiveTriggerRule(rule_id="r2", rule_type="behavior_class", output_id="DO2",
                        behavior_name="anogenital", behavior_subject_id=2),
        LiveTriggerRule(rule_id="r3", rule_type="proximity", output_id="DO3",
                        mouse_id=1, peer_mouse_id=2),
    ]


def test_rule_column_names_are_clear_and_unique(main_window):
    from live_detection_types import BehaviorROI

    w = main_window
    w.live_rois["Left Chamber"] = BehaviorROI(
        name="Left Chamber", roi_type="rectangle", data=[(0.0, 0.0, 100.0, 100.0)]
    )
    w.live_rules = _rules()
    cm = w._live_rule_export_column_map()
    assert cm["r1"] == "rule_in_zone_left_chamber_m1"
    assert cm["r2"] == "rule_behavior_anogenital_m2"
    assert cm["r3"] == "rule_proximity_m1_m2"
    assert len(set(cm.values())) == 3  # unique columns


def test_rule_binary_reflects_active_set(main_window):
    w = main_window
    w.live_rules = _rules()
    vals = w._live_rule_binary_export_values(["r2"])
    assert vals["rule_behavior_anogenital_m2"] == 1
    assert vals["rule_in_zone_left_chamber_m1"] == 0
    assert vals["rule_proximity_m1_m2"] == 0
    # all three rule columns are always present (every frame), as 0 or 1
    assert set(vals) == {
        "rule_in_zone_left_chamber_m1",
        "rule_behavior_anogenital_m2",
        "rule_proximity_m1_m2",
    }


def test_no_rules_means_no_columns(main_window):
    w = main_window
    w.live_rules = []
    assert w._live_rule_binary_export_values(["anything"]) == {}


def test_duplicate_rule_targets_get_distinct_columns(main_window):
    from live_detection_types import LiveTriggerRule

    w = main_window
    # Two rules with the same condition signature must not collide into one column.
    w.live_rules = [
        LiveTriggerRule(rule_id="a", rule_type="behavior_class", output_id="DO1",
                        behavior_name="rearing", behavior_subject_id=1),
        LiveTriggerRule(rule_id="b", rule_type="behavior_class", output_id="DO2",
                        behavior_name="rearing", behavior_subject_id=1),
    ]
    cm = w._live_rule_export_column_map()
    assert len(set(cm.values())) == 2

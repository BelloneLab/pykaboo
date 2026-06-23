"""Panel-level ROI tests: the Trigger Rules State cell surfaces the live ROI zone
status the user asked for ("in <ROI> zone"), and the ROI save/load signals exist."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    import live_detection_panel as ldp

    widget = ldp.LiveDetectionPanel()
    yield widget
    try:
        widget.deleteLater()
    except Exception:
        pass


def _roi_rule(rule_id, roi_name):
    from live_detection_types import LiveTriggerRule

    return LiveTriggerRule(
        rule_id=rule_id, rule_type="roi_occupancy", output_id="DO1",
        mode="gate", roi_name=roi_name, mouse_id=1,
    )


def test_active_roi_rule_shows_zone_status(panel):
    panel.set_rules([_roi_rule("r1", "left_chamber")], active_rule_ids=["r1"])
    assert panel.rule_table.item(0, 1).text() == "in left_chamber zone"


def test_idle_roi_rule_shows_idle(panel):
    panel.set_rules([_roi_rule("r2", "left_chamber")], active_rule_ids=[])
    assert panel.rule_table.item(0, 1).text() == "idle"


def test_active_non_roi_rule_shows_plain_active(panel):
    from live_detection_types import LiveTriggerRule

    rule = LiveTriggerRule(
        rule_id="b1", rule_type="behavior_class", output_id="DO1",
        mode="gate", behavior_name="nose2nose",
    )
    panel.set_rules([rule], active_rule_ids=["b1"])
    assert panel.rule_table.item(0, 1).text() == "ACTIVE"


def test_roi_save_load_signals_exist(panel):
    assert hasattr(panel, "save_rois_requested")
    assert hasattr(panel, "load_rois_requested")
    # select_roi_row is a no-op for an unknown name (must not raise).
    panel.select_roi_row("does-not-exist")

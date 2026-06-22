"""Interactive ROI editing regression tests (rectangle + polygon handles).

These exercise the real MainWindow ROI graphics-item machinery: an ROI registered
in ``live_rois`` must spawn a movable/resizable pyqtgraph handle, dragging or
resizing it must write the new geometry back into ``live_rois``, an external edit
must re-seed the handle (without fighting an in-progress drag), a polygon must
accept an added vertex, and deleting an ROI must drop its handle.

The MainWindow is heavy to build (loads Qt + torch), so it is constructed once per
module and the whole module skips if a headless GUI cannot be created here.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYKABOO_SHOW_SIMULATED_CAMERAS", "1")

try:  # torch first for the Windows DLL-order workaround, mirroring main.py
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch optional for collection
    pass

QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication


@pytest.fixture(scope="module")
def main_window():
    app = QApplication.instance() or QApplication([])
    try:
        import main_window_enhanced as mwe
        from live_detection_types import BehaviorROI  # noqa: F401

        window = mwe.MainWindow()
    except Exception as exc:  # pragma: no cover - environment without full GUI deps
        pytest.skip(f"MainWindow could not be constructed headless: {exc}")
    window._startup_autoconnect_done = True
    window.last_frame_size = (400, 300)
    window.live_detection_enabled = True  # force ROI overlays/handles visible
    yield window
    try:
        window.close()
    except Exception:
        pass


def _roi(name, roi_type, data):
    from live_detection_types import BehaviorROI

    return BehaviorROI(name=name, roi_type=roi_type, data=data)


def test_rectangle_drag_writes_back(main_window):
    w = main_window
    w.live_rois["RtA"] = _roi("RtA", "rectangle", [(50.0, 60.0, 180.0, 140.0)])
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    assert "RtA" in w.live_rect_roi_items

    item = w.live_rect_roi_items["RtA"]
    item.setPos([70.0, 80.0])
    item.setSize([100.0, 50.0])
    w._on_live_rect_roi_item_changed("RtA")

    x1, y1, x2, y2 = w.live_rois["RtA"].data[0]
    assert abs(x1 - 70.0) < 1e-3 and abs(y1 - 80.0) < 1e-3
    assert abs(x2 - 170.0) < 1e-3 and abs(y2 - 130.0) < 1e-3


def test_rectangle_external_edit_reseeds_item(main_window):
    w = main_window
    w.live_rois["RtB"] = _roi("RtB", "rectangle", [(50.0, 60.0, 180.0, 140.0)])
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    # An external edit (numeric dialog / load) must move the handle on the next sync.
    w.live_rois["RtB"].data = [(10.0, 10.0, 60.0, 40.0)]
    w._sync_live_circle_roi_items()
    pos = w.live_rect_roi_items["RtB"].pos()
    assert abs(float(pos.x()) - 10.0) < 1e-3 and abs(float(pos.y()) - 10.0) < 1e-3


def test_polygon_add_vertex_and_move(main_window):
    w = main_window
    poly = [(100.0, 100.0), (220.0, 110.0), (200.0, 210.0), (110.0, 200.0)]
    w.live_rois["PgA"] = _roi("PgA", "polygon", list(poly))
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    assert "PgA" in w.live_poly_roi_items

    item = w.live_poly_roi_items["PgA"]
    assert len(item.getHandles()) == 4

    # Add a vertex between the first two handles (double-click-on-edge equivalent).
    handles = item.getHandles()
    mid = (handles[0].pos() + handles[1].pos()) / 2.0
    item.addFreeHandle(mid)
    w._on_live_poly_roi_item_changed("PgA")
    assert len(w.live_rois["PgA"].data) >= 5

    # Move the whole zone and confirm every vertex tracks the translation.
    before = list(w.live_rois["PgA"].data)
    item.setPos(item.pos().x() + 15, item.pos().y() - 7)
    w._on_live_poly_roi_item_changed("PgA")
    after = w.live_rois["PgA"].data
    assert all(
        abs(a[0] - (b[0] + 15)) < 1.0 and abs(a[1] - (b[1] - 7)) < 1.0
        for a, b in zip(after, before)
    )


def test_deleting_roi_removes_handle(main_window):
    w = main_window
    w.live_rois["RtC"] = _roi("RtC", "rectangle", [(20.0, 20.0, 90.0, 70.0)])
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    assert "RtC" in w.live_rect_roi_items
    del w.live_rois["RtC"]
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    assert "RtC" not in w.live_rect_roi_items

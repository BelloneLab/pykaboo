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


def test_draw_mode_toggles_crosshair_cursor(main_window):
    from PySide6.QtCore import Qt

    w = main_window
    gv = w.live_image_view.ui.graphicsView
    w._set_live_roi_draw_mode("rectangle")
    assert w.live_roi_draw_mode == "rectangle"
    assert gv.viewport().cursor().shape() == Qt.CrossCursor
    w._set_live_roi_draw_mode("")
    assert w.live_roi_draw_mode == ""
    assert gv.viewport().cursor().shape() != Qt.CrossCursor


def test_selection_reveals_handles_only_for_selected_roi(main_window):
    w = main_window
    w.live_rois.clear()
    w.live_rois["S1"] = _roi("S1", "rectangle", [(10.0, 10.0, 60.0, 60.0)])
    w.live_rois["S2"] = _roi("S2", "rectangle", [(100.0, 100.0, 160.0, 160.0)])
    w.live_rule_engine.set_rois(w.live_rois)
    w.live_selected_roi_name = None
    w._sync_live_circle_roi_items()

    def handles(name):
        return w.live_rect_roi_items[name].getHandles()

    # Nothing selected -> every handle hidden for a clean preview.
    assert all(not h.isVisible() for h in handles("S1") + handles("S2"))
    w._select_live_roi("S1")
    assert all(h.isVisible() for h in handles("S1"))
    assert all(not h.isVisible() for h in handles("S2"))
    w._select_live_roi("")  # click empty space -> deselect
    assert all(not h.isVisible() for h in handles("S1"))


def test_polygon_item_is_double_click_editable(main_window):
    from main_window_enhanced import _EditablePolyLineROI

    w = main_window
    w.live_rois.clear()
    w.live_rois["PolyX"] = _roi("PolyX", "polygon", [(100.0, 100.0), (200.0, 100.0), (150.0, 200.0)])
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    item = w.live_poly_roi_items["PolyX"]
    assert isinstance(item, _EditablePolyLineROI)

    # A single (non-double) edge click selects without scattering a vertex.
    n_before = len(item.getHandles())
    fired = []
    item._pkb_on_select = lambda: fired.append(1)

    class _SingleClick:
        def double(self):
            return False

        def pos(self):
            from PySide6.QtCore import QPointF
            return QPointF(0.0, 0.0)

    item.segmentClicked(object(), _SingleClick())
    assert fired == [1]
    assert len(item.getHandles()) == n_before


def test_merge_rois_renames_on_name_collision(main_window):
    w = main_window
    w.live_rois.clear()
    w.live_rois["Arena"] = _roi("Arena", "rectangle", [(0.0, 0.0, 50.0, 50.0)])
    added = w._merge_live_rois(
        [
            {"name": "Arena", "roi_type": "rectangle", "data": [(5, 5, 9, 9)], "color": [1, 2, 3]},
            {"name": "Nest", "roi_type": "circle", "data": [(20, 20, 5)], "color": [4, 5, 6]},
        ]
    )
    assert added == 2
    assert "Arena (2)" in w.live_rois  # collision renamed, original kept
    assert "Nest" in w.live_rois
    assert w.live_rois["Arena"].data == [(0.0, 0.0, 50.0, 50.0)]


def test_first_zone_uses_insertion_order(main_window):
    w = main_window
    w.live_rois.clear()
    w.live_rois["outer"] = _roi("outer", "rectangle", [(0.0, 0.0, 100.0, 100.0)])
    w.live_rois["inner"] = _roi("inner", "rectangle", [(10.0, 10.0, 40.0, 40.0)])
    # Overlapping zones: the first-inserted ROI wins the single current_zone string.
    assert w._first_live_zone_for_point(20, 20) == "outer"
    assert w._first_live_zone_for_point(80, 80) == "outer"
    assert w._first_live_zone_for_point(500, 500) == ""


def test_cancel_live_roi_draw_clears_mode_and_cursor(main_window):
    from PySide6.QtCore import Qt

    w = main_window
    gv = w.live_image_view.ui.graphicsView
    w._set_live_roi_draw_mode("polygon")
    w.live_roi_draw_points = [(10.0, 10.0), (20.0, 20.0)]
    assert gv.viewport().cursor().shape() == Qt.CrossCursor
    w._cancel_live_roi_draw()
    assert w.live_roi_draw_mode == ""
    assert w.live_roi_draw_points == []
    assert w.live_roi_circle_center is None
    assert gv.viewport().cursor().shape() != Qt.CrossCursor


def test_escape_key_cancels_roi_draw(main_window):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    w = main_window
    w._set_live_roi_draw_mode("rectangle")
    w.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert w.live_roi_draw_mode == ""


def test_in_progress_draw_overlay_updates_instantly_on_click(main_window):
    w = main_window
    w.live_rois.clear()
    w._set_live_roi_draw_mode("polygon")
    w.live_roi_draw_points = []
    # Each placed point must show immediately via the vector overlay (no camera frame).
    w._handle_live_roi_click(40.0, 50.0)
    w._handle_live_roi_click(120.0, 60.0)
    scatter_pts = w._roi_draw_scatter.getData()
    assert len(scatter_pts[0]) == 2
    assert w._roi_draw_scatter.isVisible()
    # A rubber-band segment to the cursor extends the preview curve live.
    w._update_roi_draw_overlay(cursor=(130.0, 160.0))
    cx, cy = w._roi_draw_curve.getData()
    assert len(cx) == 3 and float(cx[-1]) == 130.0 and float(cy[-1]) == 160.0
    # Ending draw mode tears the preview down.
    w._set_live_roi_draw_mode("")
    assert not w._roi_draw_curve.isVisible()
    assert not w._roi_draw_scatter.isVisible()


def test_rectangle_rubber_band_preview_is_closed_rect(main_window):
    w = main_window
    w._set_live_roi_draw_mode("rectangle")
    w.live_roi_draw_points = []
    w._handle_live_roi_click(10.0, 10.0)  # first corner placed
    w._update_roi_draw_overlay(cursor=(60.0, 40.0))
    cx, cy = w._roi_draw_curve.getData()
    # 5-point closed rectangle from corner to cursor.
    assert len(cx) == 5 and float(cx[0]) == float(cx[-1]) and float(cy[0]) == float(cy[-1])
    w._set_live_roi_draw_mode("")


def test_circle_resync_does_not_snap_back_during_drag(main_window):
    w = main_window
    w.live_rois.clear()
    w.live_rois["C1"] = _roi("C1", "circle", [(100.0, 100.0, 30.0)])
    w.live_rule_engine.set_rois(w.live_rois)
    w._sync_live_circle_roi_items()
    item = w.live_circle_roi_items["C1"]

    # Emulate an in-progress drag: move the handle without committing to live_rois
    # (suppress the writeback exactly like a mid-drag resync would see it).
    w._syncing_live_circle_roi_item = True
    try:
        item.setPos([200.0, 200.0])
        item.setSize([80.0, 80.0])
    finally:
        w._syncing_live_circle_roi_item = False

    # A per-frame resync must NOT snap the circle back to the stale live_rois geometry.
    w._sync_live_circle_roi_items()
    pos = item.pos()
    assert abs(float(pos.x()) - 200.0) < 1e-3 and abs(float(pos.y()) - 200.0) < 1e-3

    # An external edit (load / numeric dialog) DOES re-seed the handle.
    w.live_rois["C1"].data = [(50.0, 50.0, 10.0)]
    w._sync_live_circle_roi_items()
    pos2 = item.pos()
    assert abs(float(pos2.x()) - 40.0) < 1e-3 and abs(float(pos2.y()) - 40.0) < 1e-3

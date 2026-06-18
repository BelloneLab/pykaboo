import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QTableWidget
except Exception:  # pragma: no cover
    QApplication = None
    QTableWidget = None

if QApplication is not None:
    from main_window_enhanced import MainWindow
else:  # pragma: no cover
    MainWindow = None


@unittest.skipIf(QApplication is None, "PySide6 is required")
class PlannerStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _planner_window(self):
        window = MainWindow.__new__(MainWindow)
        window.planner_default_columns = ["Status", "Trial"]
        window.planner_custom_columns = []
        window.planner_table = QTableWidget(1, 2)
        window.planner_table.setHorizontalHeaderLabels(window._planner_headers())
        window._syncing_planner_recording_statuses = False
        window.worker = None
        window.current_recording_filepath = None
        window.active_planner_row = None
        window._set_planner_cell(0, "Trial", "1")
        return window

    def test_manual_pending_survives_recording_reconciliation(self):
        window = self._planner_window()
        window._planner_row_has_recording = lambda row: True

        window._set_planner_row_status(0, "Acquired")
        window._set_planner_row_status(0, "Pending", manual_pending=True)
        changed = window._sync_planner_recording_statuses()

        self.assertFalse(changed)
        self.assertEqual(window._planner_row_payload(0)["Status"], "Pending")
        self.assertTrue(window._planner_row_manual_pending(0))

    def test_ordinary_pending_still_autodetects_existing_recording(self):
        window = self._planner_window()
        window._planner_row_has_recording = lambda row: True

        window._set_planner_row_status(0, "Pending")
        changed = window._sync_planner_recording_statuses()

        self.assertTrue(changed)
        self.assertEqual(window._planner_row_payload(0)["Status"], "Acquired")
        self.assertFalse(window._planner_row_manual_pending(0))

    def test_new_recording_status_clears_manual_pending_override(self):
        window = self._planner_window()

        window._set_planner_row_status(0, "Pending", manual_pending=True)
        window._set_planner_row_status(0, "Acquiring")

        self.assertEqual(window._planner_row_payload(0)["Status"], "Acquiring")
        self.assertFalse(window._planner_row_manual_pending(0))

    def test_manual_acquired_survives_recording_reconciliation(self):
        window = self._planner_window()
        window._planner_row_has_recording = lambda row: False

        window._set_planner_row_status(0, "Pending")
        window._set_planner_row_status(0, "Acquired", manual_acquired=True)
        changed = window._sync_planner_recording_statuses()

        self.assertFalse(changed)
        self.assertEqual(window._planner_row_payload(0)["Status"], "Acquired")
        self.assertTrue(window._planner_row_manual_acquired(0))

    def test_ordinary_acquired_reverts_without_recording(self):
        window = self._planner_window()
        window._planner_row_has_recording = lambda row: False

        window._set_planner_row_status(0, "Acquired")
        changed = window._sync_planner_recording_statuses()

        self.assertTrue(changed)
        self.assertEqual(window._planner_row_payload(0)["Status"], "Pending")
        self.assertFalse(window._planner_row_manual_acquired(0))

    def test_new_recording_status_clears_manual_acquired_override(self):
        window = self._planner_window()

        window._set_planner_row_status(0, "Acquired", manual_acquired=True)
        window._set_planner_row_status(0, "Acquiring")

        self.assertEqual(window._planner_row_payload(0)["Status"], "Acquiring")
        self.assertFalse(window._planner_row_manual_acquired(0))


if __name__ == "__main__":
    unittest.main()

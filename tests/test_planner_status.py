import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QLineEdit, QTableWidget, QTextEdit
except Exception:  # pragma: no cover
    QApplication = None
    QLineEdit = None
    QTableWidget = None
    QTextEdit = None

if QApplication is not None:
    from main_window_enhanced import MainWindow
else:  # pragma: no cover
    MainWindow = None


@unittest.skipIf(QApplication is None, "PySide6 is required")
class PlannerStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _planner_window(self, rows=1):
        window = MainWindow.__new__(MainWindow)
        window.planner_default_columns = ["Status", "Trial", "Animal ID", "Session"]
        window.planner_custom_columns = []
        window.planner_variable_lists = {}
        window.planner_next_trial_number = rows + 1
        window.planner_table = QTableWidget(rows, len(window._planner_headers()))
        window.planner_table.setHorizontalHeaderLabels(window._planner_headers())
        window._syncing_planner_recording_statuses = False
        window._syncing_planner_to_recording = False
        window._syncing_recording_to_planner = False
        window._planner_default_duration_text = lambda: "00:00:00"
        window.worker = None
        window.current_recording_filepath = None
        window.active_planner_row = None
        window._fit_planner_columns = lambda: None
        window._update_planner_summary = lambda: None
        window._on_status_update = lambda message: None
        window._on_error_occurred = lambda message: None
        for row in range(rows):
            window._set_planner_cell(row, "Trial", str(row + 1))
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

    def test_identity_table_edit_resets_acquired_row_to_manual_pending(self):
        for header in ("Animal ID", "Session"):
            with self.subTest(header=header):
                window = self._planner_window()
                window._set_planner_cell(0, header, "old")
                window._set_planner_row_status(0, "Acquired")
                window._set_planner_row_recording_base_path(0, "d:/recordings/old")

                column = window._planner_headers().index(header)
                item = window.planner_table.item(0, column)
                item.setText("new")
                window._on_planner_item_changed(item)

                self.assertEqual(window._planner_row_payload(0)["Status"], "Pending")
                self.assertTrue(window._planner_row_manual_pending(0))
                self.assertEqual(window._planner_row_recording_base_path(0), "")

    def test_metadata_sync_identity_edit_resets_acquired_row(self):
        window = self._planner_window()
        window.active_planner_row = 0
        window.meta_trial = QLineEdit("1")
        window.meta_arena = QLineEdit("Arena 1")
        window.meta_animal_id = QLineEdit("mouse-new")
        window.meta_session = QLineEdit("session-old")
        window.meta_experiment = QLineEdit("")
        window.meta_condition = QLineEdit("")
        window.meta_notes = QTextEdit()
        window.custom_metadata_fields = {}
        window._set_planner_cell(0, "Animal ID", "mouse-old")
        window._set_planner_cell(0, "Session", "session-old")
        window._set_planner_row_status(0, "Acquired")

        window._sync_active_trial_metadata_cells()

        self.assertEqual(window._planner_row_payload(0)["Animal ID"], "mouse-new")
        self.assertEqual(window._planner_row_payload(0)["Status"], "Pending")
        self.assertTrue(window._planner_row_manual_pending(0))

    def test_duplicate_acquired_row_is_manual_pending(self):
        window = self._planner_window()
        window._set_planner_cell(0, "Animal ID", "mouse-old")
        window._set_planner_cell(0, "Session", "session-old")
        window._set_planner_row_status(0, "Acquired")
        window.planner_table.selectRow(0)

        window._duplicate_selected_planner_trials()

        self.assertEqual(window.planner_table.rowCount(), 2)
        self.assertEqual(window._planner_row_payload(1)["Status"], "Pending")
        self.assertTrue(window._planner_row_manual_pending(1))

    def test_new_trial_uses_saved_variable_list_defaults(self):
        window = self._planner_window(rows=0)
        window.planner_next_trial_number = 1
        window.planner_variable_lists = {
            "Animal ID": ["mouse-a", "mouse-b"],
            "Session": ["session-1"],
        }

        window._append_planner_trial()
        window._append_planner_trial()

        self.assertEqual(window._planner_row_payload(0)["Animal ID"], "mouse-a")
        self.assertEqual(window._planner_row_payload(1)["Animal ID"], "mouse-b")
        self.assertEqual(window._planner_row_payload(1)["Session"], "session-1")


if __name__ == "__main__":
    unittest.main()

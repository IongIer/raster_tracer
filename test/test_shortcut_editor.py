"""The Controls editor previews, validates and persists tool-local shortcuts."""

import tempfile
import unittest
from pathlib import Path

from qgis.PyQt import sip
from qgis.PyQt.QtCore import QCoreApplication, QEvent, QSettings, Qt
from qgis.PyQt.QtGui import QKeyEvent, QKeySequence
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtWidgets import QAction, QDialogButtonBox, QWidget

from ..controls import create_controls_dialog
from ..shortcuts import DEFAULT_BINDINGS, ShortcutSettings, sequence_label
from .utilities import get_qgis_app


class ShortcutEditorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, _, _, _ = get_qgis_app()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.settings = QSettings(
            str(Path(self.directory.name) / "shortcuts.ini"), QSettings.Format.IniFormat
        )
        self.shortcuts = ShortcutSettings(settings=self.settings)
        self.window = QWidget()
        self.dialog = create_controls_dialog(self.window, self.shortcuts, self.window)
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        sip.delete(self.window)
        self.directory.cleanup()

    def set_binding(self, action, key):
        self.dialog.editors[action].setKeySequence(QKeySequence(key))

    def test_apply_saves_keys_and_modifier_together(self):
        self.set_binding("toggle_straight", "Ctrl+K")
        self.dialog.finish_modifier_combo.setCurrentIndex(
            self.dialog.finish_modifier_combo.findData("Alt")
        )
        self.assertEqual(self.shortcuts.bindings, DEFAULT_BINDINGS)
        self.assertEqual(self.shortcuts.finish_modifier, "Shift")
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.dialog.apply_button.click()
        loaded = ShortcutSettings(settings=self.settings)
        self.assertEqual(loaded.bindings["toggle_straight"], "Ctrl+K")
        self.assertEqual(loaded.finish_modifier, "Alt")
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn(sequence_label("Ctrl+K"), self.dialog.browser.toPlainText())
        self.assertIn("Alt+right-click", self.dialog.browser.toPlainText())
        self.assertNotIn("Shift+right-click", self.dialog.browser.toPlainText())

    def test_duplicates_block_apply_and_clear_unassigns(self):
        self.set_binding("toggle_straight", "D")
        self.assertFalse(self.dialog.apply_button.isEnabled())
        for action in ("toggle_straight", "toggle_dense_straight"):
            self.assertIn(
                "another tracing action", self.dialog.conflict_labels[action].text()
            )
        self.dialog.editors["toggle_dense_straight"].clear()
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.dialog.apply_button.click()
        self.assertEqual(self.shortcuts.bindings["toggle_straight"], "D")
        self.assertEqual(self.shortcuts.bindings["toggle_dense_straight"], "")
        self.assertIn("Unassigned", self.dialog.browser.toPlainText())

    def test_close_discards_edits_and_restore_defaults_needs_apply(self):
        self.set_binding("sample_color", "F9")
        self.dialog.close()
        self.dialog.show()
        self.assertEqual(self.dialog.proposed_bindings()["sample_color"], "T")
        self.set_binding("sample_color", "F9")
        self.dialog.apply_button.click()
        self.dialog.buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).click()
        self.assertEqual(self.dialog.proposed_bindings(), DEFAULT_BINDINGS)
        self.assertEqual(self.shortcuts.bindings["sample_color"], "F9")
        self.dialog.apply_button.click()
        self.assertEqual(self.shortcuts.bindings, DEFAULT_BINDINGS)

    def test_qgis_conflicts_warn_but_allow_apply_and_refresh(self):
        action = QAction("&QGIS test action", self.window)
        action.setShortcut(QKeySequence("Ctrl+K"))
        self.window.addAction(action)
        self.set_binding("sample_color", "Ctrl+K")
        warning = self.dialog.conflict_labels["sample_color"]
        self.assertIn("QGIS test action", warning.text())
        self.assertTrue(self.dialog.apply_button.isEnabled())
        self.dialog.apply_button.click()
        self.assertEqual(self.shortcuts.bindings["sample_color"], "Ctrl+K")
        self.assertEqual(action.shortcut().toString(), "Ctrl+K")
        action.setShortcut(QKeySequence("Ctrl+L"))
        self.dialog.conflict_timer.timeout.emit()
        self.assertEqual(warning.text(), "")
        self.dialog.close()
        self.assertFalse(self.dialog.conflict_timer.isActive())
        action.setShortcut(QKeySequence("Ctrl+K"))
        self.dialog.show()
        self.assertTrue(self.dialog.conflict_timer.isActive())
        self.assertIn("QGIS test action", warning.text())

    def test_disabled_finish_override_removes_modifier_reference(self):
        self.dialog.finish_modifier_combo.setCurrentIndex(
            self.dialog.finish_modifier_combo.findData("None")
        )
        self.dialog.apply_button.click()
        self.assertEqual(self.shortcuts.finish_modifier, "None")
        self.assertNotIn("+right-click", self.dialog.browser.toPlainText())
        self.assertIn("Right-click", self.dialog.browser.toPlainText())

    def test_capture_is_one_stroke_and_escape_does_not_close_dialog(self):
        editor = self.dialog.editors["sample_color"]
        editor.setFocus()
        QTest.keyClick(editor, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(editor.keySequence().toString(), "Ctrl+K")
        QTest.keyClick(editor, Qt.Key.Key_L, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(editor.keySequence().toString(), "Alt+L")
        self.assertEqual(editor.keySequence().count(), 1)
        QTest.keyClick(editor, Qt.Key.Key_Escape)
        self.assertTrue(self.dialog.isVisible())
        self.assertEqual(editor.keySequence().toString(), "Esc")
        self.assertFalse(self.dialog.apply_button.isEnabled())
        QTest.keyClick(editor, Qt.Key.Key_Tab)
        self.assertEqual(editor.keySequence().toString(), "Tab")
        self.assertTrue(self.dialog.apply_button.isEnabled())

    def test_captured_shifted_punctuation_matches_runtime_event(self):
        editor = self.dialog.editors["sample_color"]
        editor.setFocus()
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Plus,
            Qt.KeyboardModifier.ShiftModifier,
            "+",
        )
        QCoreApplication.sendEvent(editor, event)
        self.assertEqual(editor.keySequence().toString(), "Shift++")
        self.dialog.apply_button.click()
        self.assertEqual(self.shortcuts.action(event), "sample_color")

    def test_multistroke_programmatic_value_is_invalid(self):
        self.set_binding("sample_color", "Ctrl+K, Ctrl+L")
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn(
            "one key combination", self.dialog.conflict_labels["sample_color"].text()
        )
        self.assertEqual(self.shortcuts.bindings["sample_color"], "T")

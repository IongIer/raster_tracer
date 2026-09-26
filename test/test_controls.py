"""Tracing shortcuts respect modifiers and stay out of other input widgets."""

from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
from qgis.PyQt.QtGui import QKeyEvent
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtWidgets import QDialog, QLineEdit, QTextBrowser, QVBoxLayout

from ..pointtool import TracingModes
from .tracing_fixture import TraceFixture


class ControlsTest(TraceFixture):
    def test_modified_keys_do_not_change_modes_or_cancel_work(self):
        submitted = self.control_tasks()
        self.add_anchors()
        self.assertEqual(len(submitted), 1)
        self.assertTrue(self.tool.tracking_is_active)
        for modifier in (
            Qt.KeyboardModifier.ShiftModifier,
            Qt.KeyboardModifier.ControlModifier,
            Qt.KeyboardModifier.AltModifier,
            Qt.KeyboardModifier.MetaModifier,
        ):
            for key in (
                Qt.Key.Key_A,
                Qt.Key.Key_D,
                Qt.Key.Key_B,
                Qt.Key.Key_Escape,
            ):
                with self.subTest(modifier=modifier, key=key):
                    self.tool.keyPressEvent(
                        QKeyEvent(QEvent.Type.KeyPress, key, modifier)
                    )
                    self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
                    self.assertTrue(self.tool.tracking_is_active)
                    self.assertFalse(submitted[0].cancelled)
        self.key(Qt.Key.Key_Escape)
        self.assertTrue(submitted[0].cancelled)

    def test_layers_panel_ignores_modified_keys(self):
        self.accept(8, 2)
        tree = self.iface.layerTreeView()
        QCoreApplication.sendEvent(
            tree,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_A,
                Qt.KeyboardModifier.ControlModifier,
            ),
        )
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
        QCoreApplication.sendEvent(
            tree,
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier
            ),
        )
        self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)

    def test_reference_reuses_nonmodal_window_without_triggering_tracing(self):
        self.accept(8, 2)
        dock = self.plugin.dockwidget
        dock.controlsButton.click()
        reference = dock.controls_dialog
        self.assertTrue(reference.isVisible())
        self.assertFalse(reference.isModal())
        browser = reference.findChild(QTextBrowser, "controlsReference")
        self.assertIn("Shift+right-click", browser.toPlainText())
        QTest.keyClick(browser, Qt.Key.Key_A)
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
        reference.close()
        dock.controlsButton.click()
        self.assertIs(dock.controls_dialog, reference)
        self.assertTrue(reference.isVisible())
        reference.close()

    def test_typing_into_dialog_does_not_run_tracing_shortcuts(self):
        self.accept(8, 2)
        dialog = QDialog(self.window)
        layout = QVBoxLayout(dialog)
        editor = QLineEdit(dialog)
        layout.addWidget(editor)
        try:
            dialog.show()
            editor.setFocus()
            QTest.keyClicks(editor, "adtbs")
            self.assertEqual(editor.text(), "adtbs")
            self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
            self.assertEqual(len(self.tool.session.anchors), 1)
        finally:
            dialog.close()
            dialog.deleteLater()

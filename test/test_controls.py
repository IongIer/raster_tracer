"""Tracing shortcuts respect modifiers and stay out of other input widgets."""

from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
from qgis.PyQt.QtGui import QKeyEvent
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtWidgets import QAction, QDialog, QLineEdit, QTextBrowser, QVBoxLayout

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
                Qt.Key.Key_E,
                Qt.Key.Key_B,
                Qt.Key.Key_Escape,
            ):
                with self.subTest(modifier=modifier, key=key):
                    self.tool.keyPressEvent(
                        QKeyEvent(QEvent.Type.KeyPress, key, modifier)
                    )
                    self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
                    self.assertFalse(self.tool.enhanced_tracing)
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
            QTest.keyClicks(editor, "adetbn")
            self.assertEqual(editor.text(), "adetbn")
            self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
            self.assertFalse(self.tool.enhanced_tracing)
            self.assertEqual(len(self.tool.session.anchors), 1)
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_custom_bindings_apply_to_canvas_and_layers_panel(self):
        bindings = dict(self.plugin.shortcuts.bindings)
        bindings["toggle_straight"] = "Ctrl+J"
        bindings["toggle_dense_straight"] = ""
        self.plugin.shortcuts.apply(bindings, "Alt")
        self.accept(8, 2)
        self.key(Qt.Key.Key_A)
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
        self.tool.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier
            )
        )
        self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)
        QCoreApplication.sendEvent(
            self.iface.layerTreeView(),
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier
            ),
        )
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
        self.assertIn(
            "Alt+right-click", self.plugin.dockwidget.checkBoxOpenAttributes.toolTip()
        )
        self.assertNotIn(
            "Shift+right-click", self.plugin.dockwidget.checkBoxOpenAttributes.toolTip()
        )

    def test_bindings_and_finish_modifier_survive_plugin_recreation(self):
        from .. import classFactory

        bindings = dict(self.plugin.shortcuts.bindings)
        bindings["cancel_pending"] = "F9"
        bindings["sample_color"] = ""
        self.plugin.shortcuts.apply(bindings, "Meta")
        self.plugin.dockwidget.close()
        self.plugin.run()
        self.assertEqual(self.plugin.shortcuts.bindings, bindings)
        self.assertEqual(self.plugin.shortcuts.finish_modifier, "Meta")
        self.plugin.unload()
        self.plugin = classFactory(self.iface)
        self.plugin.initGui()
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.assertEqual(self.tool.shortcuts.bindings, bindings)
        self.assertEqual(self.tool.shortcuts.finish_modifier, "Meta")

    def test_canvas_capture_handles_navigation_keys_only_while_tracing(self):
        bindings = dict(self.plugin.shortcuts.bindings, toggle_straight="Tab")
        self.plugin.shortcuts.apply(bindings, "Shift")
        QTest.keyClick(self.canvas, Qt.Key.Key_Tab)
        self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)
        self.canvas.setMapTool(self.iface.pan_tool)
        QTest.keyClick(self.canvas, Qt.Key.Key_Tab)
        self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)
        self.plugin.activate_map_tool()
        QTest.keyClick(self.canvas, Qt.Key.Key_Tab)
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)

    def test_qgis_shortcut_keeps_priority_and_remapping_restores_tracing(self):
        bindings = dict(self.plugin.shortcuts.bindings, toggle_straight="Ctrl+J")
        self.plugin.shortcuts.apply(bindings, "Shift")
        action = QAction("Existing QGIS action", self.canvas)
        action.setShortcut("Ctrl+J")
        action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.canvas.addAction(action)
        triggered = []
        action.triggered.connect(lambda: triggered.append(True))
        try:
            self.window.show()
            self.window.activateWindow()
            self.canvas.setFocus()
            self.app.processEvents()
            QTest.keyClick(
                self.canvas, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier
            )
            self.assertEqual(triggered, [True])
            self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
            self.plugin.shortcuts.apply(
                dict(bindings, toggle_straight="Ctrl+K"), "Shift"
            )
            QTest.keyClick(
                self.canvas, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier
            )
            self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)
            self.assertEqual(action.shortcut().toString(), "Ctrl+J")
        finally:
            self.canvas.removeAction(action)
            action.deleteLater()
            self.window.hide()

    def test_unhandled_keys_in_inline_editors_never_reach_tracing(self):
        self.plugin.shortcuts.apply(
            dict(self.plugin.shortcuts.bindings, toggle_straight="Ctrl+J"), "Shift"
        )
        self.accept(8, 2)
        try:
            self.window.show()
            self.window.activateWindow()
            for parent in (self.iface.layerTreeView(), self.canvas):
                with self.subTest(parent=parent):
                    editor = QLineEdit(parent)
                    try:
                        parent.show()
                        editor.show()
                        editor.setFocus()
                        self.app.processEvents()
                        self.assertTrue(editor.hasFocus())
                        QTest.keyClick(
                            editor, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier
                        )
                        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
                    finally:
                        editor.close()
                        editor.deleteLater()
        finally:
            self.window.hide()

    def test_assigned_navigation_key_repeat_does_not_pan_the_canvas(self):
        self.plugin.shortcuts.apply(
            dict(self.plugin.shortcuts.bindings, toggle_straight="Left"), "Shift"
        )
        before = self.canvas.extent().toString()
        QTest.keyClick(self.canvas, Qt.Key.Key_Left)
        self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)
        QCoreApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Left,
                Qt.KeyboardModifier.NoModifier,
                "",
                True,
            ),
        )
        self.assertEqual(self.tool.tracing_mode, TracingModes.LINE)
        self.assertEqual(self.canvas.extent().toString(), before)

    def test_configured_keys_do_not_intercept_form_input_or_repeat_toggles(self):
        bindings = dict(self.plugin.shortcuts.bindings)
        bindings["toggle_straight"] = "J"
        self.plugin.shortcuts.apply(bindings, "None")
        self.accept(8, 2)
        self.tool.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_J,
                Qt.KeyboardModifier.NoModifier,
                "j",
                True,
            )
        )
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
        dialog = QDialog(self.window)
        layout = QVBoxLayout(dialog)
        editor = QLineEdit(dialog)
        layout.addWidget(editor)
        try:
            dialog.show()
            editor.setFocus()
            QTest.keyClicks(editor, "jadtbn")
            self.assertEqual(editor.text(), "jadtbn")
            self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)
        finally:
            dialog.close()
            dialog.deleteLater()

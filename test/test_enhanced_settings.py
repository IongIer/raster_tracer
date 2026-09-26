"""The enhanced tracer selection stays visible, persistent and tool-local."""

from qgis.PyQt.QtCore import QEvent, QSettings, Qt
from qgis.PyQt.QtGui import QKeyEvent

from .. import classFactory
from ..pointtool import TracingModes
from .tracing_fixture import TraceFixture


class EnhancedSettingsTest(TraceFixture):
    def test_selection_defaults_off_and_survives_reopening_and_recreation(self):
        self.assertFalse(self.plugin.dockwidget.checkBoxEnhancedTracing.isChecked())
        self.assertFalse(self.tool.enhanced_tracing)
        self.plugin.dockwidget.checkBoxEnhancedTracing.setChecked(True)
        self.assertTrue(self.tool.enhanced_tracing)
        self.assertTrue(QSettings().value("RasterScribe/trace/enhanced", type=bool))
        self.plugin.dockwidget.close()
        self.plugin.run()
        self.assertTrue(self.plugin.dockwidget.checkBoxEnhancedTracing.isChecked())
        self.assertTrue(self.plugin.tool_identify.enhanced_tracing)
        self.plugin.unload()
        self.plugin = classFactory(self.iface)
        self.plugin.initGui()
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.assertTrue(self.plugin.dockwidget.checkBoxEnhancedTracing.isChecked())
        self.assertTrue(self.tool.enhanced_tracing)
        self.plugin.dockwidget.checkBoxEnhancedTracing.setChecked(False)
        self.assertFalse(self.tool.enhanced_tracing)
        self.assertFalse(QSettings().value("RasterScribe/trace/enhanced", type=bool))

    def test_shortcut_syncs_checkbox_and_preserves_straight_and_dense_modes(self):
        for key, mode in (
            (Qt.Key.Key_A, TracingModes.LINE),
            (Qt.Key.Key_D, TracingModes.DENSE_LINE),
            (Qt.Key.Key_D, TracingModes.PATH),
        ):
            self.key(key)
            self.assertEqual(self.tool.tracing_mode, mode)
            self.key(Qt.Key.Key_E)
            self.assertTrue(self.tool.enhanced_tracing)
            self.assertTrue(self.plugin.dockwidget.checkBoxEnhancedTracing.isChecked())
            self.assertEqual(self.tool.tracing_mode, mode)
            self.key(Qt.Key.Key_E)
            self.assertFalse(self.tool.enhanced_tracing)
            self.assertFalse(self.plugin.dockwidget.checkBoxEnhancedTracing.isChecked())
            self.assertEqual(self.tool.tracing_mode, mode)

    def test_custom_shortcut_replaces_e_and_requires_exact_modifiers(self):
        self.plugin.shortcuts.apply(
            dict(self.plugin.shortcuts.bindings, toggle_enhanced_tracing="Ctrl+J"),
            "Shift",
        )
        self.key(Qt.Key.Key_E)
        self.key(Qt.Key.Key_J)
        self.assertFalse(self.tool.enhanced_tracing)
        self.tool.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier
            )
        )
        self.assertTrue(self.tool.enhanced_tracing)
        self.assertTrue(self.plugin.dockwidget.checkBoxEnhancedTracing.isChecked())

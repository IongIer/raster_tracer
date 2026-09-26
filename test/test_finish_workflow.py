"""Finish gestures, saved preferences and scratch layers with real QGIS edits."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsField,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QEvent,
    QMetaType,
    QPointF,
    QSettings,
    Qt,
    QTimer,
)
from qgis.PyQt.QtGui import QMouseEvent
from qgis.PyQt.QtWidgets import QApplication

from .. import classFactory
from .. import raster_scribe as plugin_module
from ..attribute_form import FinishedFeatureDialog
from ..pointtool import TracingModes
from .tracing_fixture import TraceFixture
from .utilities import write_raster


class FinishWorkflowTest(TraceFixture):
    def trace_line(self):
        self.tool.tracing_mode = TracingModes.LINE
        self.add_anchors()
        self.assertIsNotNone(self.tool.session.geometry)

    def finish_click(self, modifiers=Qt.KeyboardModifier.NoModifier):
        # Exercise QGIS canvas dispatch too, including its Shift mouse handling.
        for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            event = QMouseEvent(
                kind,
                QPointF(50, 50),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton
                if kind == QEvent.Type.MouseButtonPress
                else Qt.MouseButton.NoButton,
                modifiers,
            )
            QCoreApplication.sendEvent(self.canvas.viewport(), event)

    def test_start_tracing_tracks_target_selection_and_editing(self):
        button = self.plugin.dockwidget.startTracingButton
        self.assertTrue(button.isEnabled())
        self.canvas.setMapTool(self.iface.pan_tool)
        self.assertTrue(self.vector.commitChanges())
        self.assertFalse(button.isEnabled())
        self.assertIn("Enable editing", button.toolTip())
        self.plugin.start_tracing()
        self.assertIs(self.canvas.mapTool(), self.iface.pan_tool)
        self.assertTrue(self.vector.startEditing())
        self.assertTrue(button.isEnabled())
        button.click()
        self.assertIs(self.canvas.mapTool(), self.tool)
        self.assertFalse(self.tool.suspended)
        for geometry in ("LineString", "Point", "Polygon", "MultiCurve"):
            with self.subTest(geometry=geometry):
                layer = QgsVectorLayer(geometry + "?crs=EPSG:3857", geometry, "memory")
                self.project.addMapLayer(layer)
                self.assertTrue(layer.startEditing())
                self.iface.setActiveLayer(layer)
                self.assertEqual(button.isEnabled(), geometry == "MultiCurve")
        self.iface.setActiveLayer(self.raster)
        self.assertFalse(button.isEnabled())
        self.iface.setActiveLayer(self.vector)
        self.assertTrue(button.isEnabled())
        self.project.removeMapLayer(self.vector.id())
        self.assertFalse(button.isEnabled())

    def test_start_tracing_preserves_draft_mode_and_saved_options(self):
        self.trace_line()
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        self.tool.tracing_mode = TracingModes.DENSE_LINE
        self.plugin.dockwidget.checkBoxSmooth.setChecked(False)
        session_id = self.tool.session.session_id
        geometry = bytes(self.tool.session.geometry.asWkb())
        self.plugin.dockwidget.startTracingButton.click()
        self.assertEqual(self.tool.session.session_id, session_id)
        self.assertEqual(bytes(self.tool.session.geometry.asWkb()), geometry)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.tool.tracing_mode, TracingModes.DENSE_LINE)
        self.assertFalse(self.tool.smooth_line)
        self.assertTrue(self.tool.open_attributes_on_finish)

    def test_start_tracing_rejects_provider_without_feature_creation(self):
        provider = self.vector.dataProvider()
        capabilities = (
            provider.capabilities() & ~Qgis.VectorProviderCapability.AddFeatures
        )
        self.canvas.setMapTool(self.iface.pan_tool)
        with patch.object(provider, "capabilities", return_value=capabilities):
            self.plugin.start_tracing()
            self.assertFalse(self.plugin.dockwidget.startTracingButton.isEnabled())
            self.assertIs(self.canvas.mapTool(), self.iface.pan_tool)

    def test_finish_gestures_open_exact_feature_and_do_not_change_preference(self):
        for enabled in (False, True):
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ShiftModifier,
                Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier,
            ):
                with self.subTest(enabled=enabled, modifiers=modifiers):
                    self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(enabled)
                    self.trace_line()
                    before = {feature.id() for feature in self.vector.getFeatures()}
                    expected = bytes(self.tool.session.geometry.asWkb())
                    with patch.object(
                        plugin_module, "show_finished_feature_form"
                    ) as show:
                        self.finish_click(modifiers)
                    added = [
                        f for f in self.vector.getFeatures() if f.id() not in before
                    ]
                    self.assertEqual(len(added), 1)
                    self.assertEqual(bytes(added[0].geometry().asWkb()), expected)
                    invert = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                    if enabled != invert:
                        show.assert_called_once_with(
                            self.iface, self.vector, added[0].id()
                        )
                    else:
                        show.assert_not_called()
                    self.assertFalse(self.tool._finishing)
                    self.assertFalse(self.vector.isEditCommandActive())
                    self.assertFalse(self.tool.has_active_trace())
                    self.assertEqual(
                        QSettings().value(
                            "RasterScribe/attributes/open_after_finish", type=bool
                        ),
                        enabled,
                    )

    def test_preference_survives_close_and_new_plugin_instance(self):
        self.assertFalse(self.plugin.dockwidget.checkBoxOpenAttributes.isChecked())
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        self.plugin.dockwidget.close()
        self.plugin.run()
        self.assertTrue(self.plugin.dockwidget.checkBoxOpenAttributes.isChecked())
        self.assertTrue(self.plugin.tool_identify.open_attributes_on_finish)
        self.plugin.unload()
        self.plugin = classFactory(self.iface)
        self.plugin.initGui()
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.assertTrue(self.plugin.dockwidget.checkBoxOpenAttributes.isChecked())
        self.assertTrue(self.tool.open_attributes_on_finish)
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(False)
        self.plugin.dockwidget.close()
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.assertFalse(self.tool.open_attributes_on_finish)

    def test_custom_finish_modifier_opens_form_once_and_disabled_never_inverts(self):
        for modifier, clicks in (
            (
                "Alt",
                (
                    (Qt.KeyboardModifier.AltModifier, True),
                    (Qt.KeyboardModifier.ShiftModifier, False),
                ),
            ),
            (
                "None",
                (
                    (Qt.KeyboardModifier.ShiftModifier, False),
                    (Qt.KeyboardModifier.AltModifier, False),
                ),
            ),
        ):
            self.plugin.shortcuts.apply(self.plugin.shortcuts.bindings, modifier)
            for pressed, expected in clicks:
                with self.subTest(modifier=modifier, pressed=pressed):
                    self.trace_line()
                    with patch.object(
                        plugin_module, "show_finished_feature_form"
                    ) as show:
                        self.finish_click(pressed)
                    self.assertEqual(show.call_count, int(expected))
                    self.assertFalse(self.tool.open_attributes_on_finish)
        self.assertNotIn(
            "+right-click", self.plugin.dockwidget.checkBoxOpenAttributes.toolTip()
        )

    def test_finish_opens_real_form_and_accepts_first_attribute(self):
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        self.trace_line()
        outcomes = []

        def fill_form():
            dialog = QApplication.activeModalWidget()
            try:
                self.assertIsInstance(dialog, FinishedFeatureDialog)
                self.assertFalse(self.tool.has_active_trace())
                self.assertFalse(self.tool._finishing)
                self.assertFalse(self.vector.isEditCommandActive())
                self.assertTrue(
                    dialog.add_field(
                        QgsField("height", QMetaType.Type.Double, "double")
                    )
                )
                dialog.form.changeAttribute("height", 127.5)
                dialog.accept()
                outcomes.append(dialog.result())
            except Exception as error:
                outcomes.append(error)
            finally:
                if dialog is not None and dialog.isVisible():
                    dialog.reject()

        QTimer.singleShot(0, fill_form)
        self.finish_click()
        self.assertEqual(outcomes, [FinishedFeatureDialog.DialogCode.Accepted])
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertEqual(next(self.vector.getFeatures())["height"], 127.5)
        self.assertEqual(self.vector.undoStack().count(), 3)
        self.assertTrue(self.vector.isEditable())
        self.assertEqual(self.vector.dataProvider().featureCount(), 0)

    def test_implicit_finishes_do_not_open_attributes(self):
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        with patch.object(plugin_module, "show_finished_feature_form") as show:
            self.trace_line()
            self.tool.finish_session()
            self.trace_line()
            self.canvas.setMapTool(self.iface.pan_tool)
            self.plugin.activate_map_tool()
            self.trace_line()
            self.vector.commitChanges(False)
            self.trace_line()
            self.tool.raster_layer_has_changed(self.raster)
            self.trace_line()
            self.plugin.dockwidget.close()
            show.assert_not_called()
        self.assertEqual(self.vector.featureCount(), 5)

    def test_empty_repeated_and_failed_finishes_never_open_wrong_feature(self):
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        with patch.object(plugin_module, "show_finished_feature_form") as show:
            self.finish_click()
            show.assert_not_called()
            self.trace_line()
            expected = bytes(self.tool.session.geometry.asWkb())
            with patch.object(self.vector, "addFeature", return_value=False):
                self.finish_click()
            show.assert_not_called()
            self.assertEqual(bytes(self.tool.session.geometry.asWkb()), expected)
            self.finish_click()
            self.finish_click()
            show.assert_called_once()
            self.assertEqual(self.vector.featureCount(), 1)

    def test_form_failure_keeps_finished_line_undoable(self):
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        self.trace_line()
        with patch.object(
            plugin_module,
            "show_finished_feature_form",
            side_effect=RuntimeError("form failed"),
        ):
            self.finish_click()
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertFalse(self.tool.has_active_trace())
        self.vector.undoStack().undo()
        self.assertEqual(self.vector.featureCount(), 0)

    def test_late_segment_result_does_not_reopen_form_or_modify_finished_line(self):
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        submitted = self.control_tasks()
        self.tool.tracing_mode = TracingModes.LINE
        self.accept(8, 2)
        self.accept(8, 7)
        expected = bytes(self.tool.session.geometry.asWkb())
        self.tool.tracing_mode = TracingModes.PATH
        self.accept(8, 13)
        pending = submitted[-1]
        with patch.object(plugin_module, "show_finished_feature_form") as show:
            self.finish_click()
            pending.finish()
            show.assert_called_once()
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertEqual(
            bytes(next(self.vector.getFeatures()).geometry().asWkb()), expected
        )

    def test_scratch_layer_uses_selected_raster_crs_and_is_ready_to_trace(self):
        crs = QgsCoordinateReferenceSystem.fromProj(
            "+proj=tmerc +lat_0=0 +lon_0=17.123 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        self.assertTrue(crs.isValid())
        self.raster.setCrs(crs)
        layer = self.plugin.create_scratch_layer()
        self.assertIsNotNone(layer)
        self.assertEqual(layer.crs(), crs)
        self.assertEqual(layer.wkbType(), Qgis.WkbType.MultiLineString)
        self.assertEqual(layer.providerType(), "memory")
        self.assertEqual(layer.fields().count(), 0)
        self.assertTrue(layer.isEditable())
        self.assertIs(self.iface.activeLayer(), layer)
        self.assertIs(self.tool.get_current_vector_layer(), layer)
        self.assertIs(
            self.plugin.dockwidget.mMapLayerComboBox.currentLayer(), self.raster
        )
        self.assertIs(self.canvas.mapTool(), self.tool)
        self.assertEqual(self.project.crs().authid(), "EPSG:3857")

    def test_scratch_button_tracks_missing_raster_and_undefined_crs(self):
        button = self.plugin.dockwidget.createScratchLayerButton
        self.assertTrue(button.isEnabled())
        self.raster.setCrs(QgsCoordinateReferenceSystem())
        self.assertFalse(button.isEnabled())
        self.assertIsNone(self.plugin.create_scratch_layer())
        self.raster.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        self.assertTrue(button.isEnabled())
        self.project.removeMapLayer(self.raster.id())
        self.assertFalse(button.isEnabled())
        self.assertIsNone(self.plugin.create_scratch_layer())

    def test_scratch_button_rejects_rasters_unsupported_by_tracing(self):
        path = Path(self.directory.name) / "single-band.tif"
        write_raster(
            path,
            (np.zeros((2, 2), dtype=np.uint8),),
            gdal.GDT_Byte,
            projection=self.raster.crs().toWkt(),
        )
        raster = QgsRasterLayer(str(path), "Single band")
        self.assertTrue(raster.isValid())
        self.project.addMapLayer(raster)
        combo = self.plugin.dockwidget.mMapLayerComboBox
        combo.setLayer(raster)
        self.assertFalse(self.plugin.dockwidget.createScratchLayerButton.isEnabled())
        self.assertFalse(self.plugin.dockwidget.startTracingButton.isEnabled())
        self.assertIsNone(self.plugin.create_scratch_layer())
        combo.setLayer(self.raster)
        self.assertTrue(self.plugin.dockwidget.createScratchLayerButton.isEnabled())
        self.assertTrue(self.plugin.dockwidget.startTracingButton.isEnabled())
        self.project.removeMapLayer(raster.id())
        self.project.removeMapLayer(self.raster.id())
        self.assertFalse(self.plugin.dockwidget.startTracingButton.isEnabled())

    def test_scratch_creation_preserves_old_draft_on_failure_and_finishes_on_success(
        self,
    ):
        self.plugin.dockwidget.checkBoxOpenAttributes.setChecked(True)
        self.trace_line()
        before = set(self.project.mapLayers())
        with patch.object(self.vector, "addFeature", return_value=False):
            self.assertIsNone(self.plugin.create_scratch_layer())
        self.assertEqual(set(self.project.mapLayers()), before)
        self.assertIs(self.iface.activeLayer(), self.vector)
        self.assertTrue(self.tool.has_active_trace())
        with patch.object(plugin_module, "show_finished_feature_form") as show:
            first = self.plugin.create_scratch_layer()
            second = self.plugin.create_scratch_layer()
            show.assert_not_called()
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertNotEqual(first.name(), second.name())
        self.tool.tracing_mode = TracingModes.LINE
        self.add_anchors()
        self.tool.finish_session()
        self.assertEqual(second.featureCount(), 1)

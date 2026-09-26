"""Integration coverage for the same plugin under Qt5/QGIS 3 and Qt6/QGIS 4."""

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt import sip
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QEvent,
    QPoint,
    QPointF,
    QSettings,
    Qt,
)
from qgis.PyQt.QtGui import QColor, QKeyEvent, QMouseEvent

from .. import classFactory
from ..pointtool import TracingModes
from ..pointtool_tasks import execute_request
from ..utils import RasterSampler
from .tracing_fixture import TraceFixture
from .utilities import wait_until


class CompatibilityTest(TraceFixture):
    def test_preferences_are_separate_from_raster_tracer(self):
        settings = QSettings()
        settings.setValue("RasterTracer/preview/enabled", False)
        settings.setValue("RasterTracer/color/value", "#ff123456")
        self.plugin.dockwidget.close()
        settings.remove("RasterScribe")
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.tools.append(self.tool)
        self.assertTrue(self.plugin.dockwidget.checkBoxPreview.isChecked())
        self.assertFalse(self.plugin.dockwidget.checkBoxColor.isChecked())
        self.plugin.set_trace_color_from_tool(QColor("#abcdef"))
        self.assertEqual(settings.value("RasterTracer/color/value"), "#ff123456")
        self.assertFalse(settings.value("RasterTracer/preview/enabled", type=bool))
        self.assertEqual(settings.value("RasterScribe/color/value"), "#ffabcdef")

    def test_start_close_and_reopen(self):
        self.assertTrue(self.plugin.pluginIsActive)
        self.assertIs(self.canvas.mapTool(), self.tool)
        self.assertFalse(self.plugin.actions[0].icon().pixmap(24, 24).isNull())
        self.assertIs(self.tool.get_current_vector_layer(), self.vector)
        self.plugin.preview_color_changed(QColor("#80402010"))
        self.assertEqual(QSettings().value("RasterScribe/preview/color"), "#80402010")
        dock = self.plugin.dockwidget
        dock.close()
        self.assertFalse(self.plugin.pluginIsActive)
        self.assertIs(self.canvas.mapTool(), self.iface.pan_tool)
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.tools.append(self.tool)
        self.assertIs(self.canvas.mapTool(), self.tool)
        self.assertIs(
            self.plugin.dockwidget.mMapLayerComboBox.currentLayer(), self.raster
        )
        self.assertIs(self.tool.rlayer, self.raster)
        self.assertEqual(
            self.plugin.dockwidget.previewColorButton.color(), QColor("#80402010")
        )

    def test_read_byte_raster_for_color_sampling(self):
        sampler = RasterSampler(self.raster, self.project)
        bounds, bands, valid = sampler.read_small(8, 2, radius=1)
        self.assertEqual(bounds, (7, 10, 1, 4))
        self.assertEqual(valid.shape, (3, 3))
        self.assertTrue(valid.all())
        for band in bands:
            self.assertEqual(band.dtype, np.dtype("uint8"))
            self.assertEqual(float(band[1, 1]), 0)
            self.assertEqual(float(band[0, 1]), 255)

    def test_raster_selection_tracks_project_changes(self):
        combo = self.plugin.dockwidget.mMapLayerComboBox
        self.assertEqual(combo.count(), 1)
        self.tool.tracing_mode = TracingModes.LINE
        self.add_anchors()

        self.project.addMapLayer(QgsVectorLayer("Point", "A new vector", "memory"))
        self.assertEqual(combo.count(), 1)
        self.assertIs(combo.currentLayer(), self.raster)

        another_raster = QgsRasterLayer(self.raster.source(), "A new raster")
        self.assertTrue(another_raster.isValid())
        self.project.addMapLayer(another_raster)
        self.assertEqual(combo.count(), 2)
        self.assertIs(combo.currentLayer(), self.raster)
        self.assertIs(self.tool.rlayer, self.raster)
        # Inserting another layer must not finish the active draft.
        self.assertEqual(len(self.tool.anchors), 2)
        self.assertEqual(self.vector.featureCount(), 0)

        combo.setLayer(another_raster)
        self.assertIs(self.tool.rlayer, another_raster)
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertFalse(self.tool.has_active_trace())
        self.project.removeMapLayer(another_raster.id())
        self.assertIs(combo.currentLayer(), self.raster)
        self.assertIs(self.tool.rlayer, self.raster)
        self.project.removeMapLayer(self.raster.id())
        self.assertEqual(combo.count(), 0)
        self.assertIsNone(combo.currentLayer())
        self.assertIsNone(self.tool.rlayer)

    def test_trace_known_line(self):
        self.accept(8, 2)
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        request = self.tool.make_request(endpoint)
        result = execute_request(request.worker_input())
        self.assertEqual(result.status, "success")
        path = [(i + result.origin[0], j + result.origin[1]) for i, j in result.path]
        self.assertEqual(path, self.expected_path)
        self.assertEqual(result.cost, 0)

    def test_background_trace_and_save(self):
        self.add_anchors()
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertIsNotNone(self.tool.session.geometry)
        self.assertTrue(self.vector.commitChanges())
        self.assertEqual(self.vector.featureCount(), 1)
        geometry = next(self.vector.getFeatures()).geometry()
        self.assertFalse(geometry.isEmpty())
        self.assertAlmostEqual(geometry.length(), 11)

    def test_preview_and_cancel_restart(self):
        self.accept(8, 2)
        preview = self.tool.preview_controller
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        request = self.tool.make_request(endpoint, QPoint(10, 20))
        preview.queue(request)
        preview.ensure_inflight_started()
        self.tool.clear_preview()
        request = self.tool.make_request(endpoint, QPoint(10, 20))
        preview.queue(request)
        preview.ensure_inflight_started()
        wait_until(lambda: preview._cached_result is not None)
        result = preview.matching_cached(request)
        self.assertEqual(
            [(i + result.origin[0], j + result.origin[1]) for i, j in result.path],
            self.expected_path,
        )
        self.tool.clear_preview()
        self.assertIsNone(preview._cached_result)
        self.assertFalse(preview._rubber_band.isVisible())

    def test_cancel_trace_then_restart(self):
        self.add_anchors()
        event = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
        self.tool.keyPressEvent(event)
        self.assertFalse(self.tool.tracking_is_active)
        self.assertEqual(len(self.tool.anchors), 1)
        self.accept(8, 13)
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertTrue(self.tool.finish_session())
        self.assertEqual(self.vector.featureCount(), 1)

    def test_mouse_clicks_create_and_finish_line(self):
        self.click(8, 2, Qt.MouseButton.LeftButton)
        self.click(8, 13, Qt.MouseButton.LeftButton)
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 0)
        geometry = self.tool.session.geometry
        # Synthetic clicks are rounded to integer screen pixels, unlike
        # the exact map-coordinate assertions in the background trace test.
        self.assertAlmostEqual(
            geometry.length(), 11, delta=2 * self.canvas.mapUnitsPerPixel()
        )
        self.click(8, 13, Qt.MouseButton.RightButton)
        self.assertFalse(self.tool.has_active_trace())
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertEqual(
            bytes(next(self.vector.getFeatures()).geometry().asWkb()),
            bytes(geometry.asWkb()),
        )

    def test_pan_zoom_and_self_snap_keep_the_draft(self):
        self.tool.tracing_mode = TracingModes.LINE
        for point in ((1002, 1992), (1008, 1992), (1013, 1996)):
            self.tool.accept_click(point)
        expected = bytes(self.tool.session.geometry.asWkb())
        anchors = self.tool.anchors
        extent = self.canvas.extent()
        for kind, position, button, buttons in (
            (
                QEvent.Type.MouseButtonPress,
                (50, 50),
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.MiddleButton,
            ),
            (
                QEvent.Type.MouseMove,
                (70, 60),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.MiddleButton,
            ),
            (
                QEvent.Type.MouseButtonRelease,
                (70, 60),
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.NoButton,
            ),
        ):
            event = QMouseEvent(
                kind,
                QPointF(*position),
                button,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            )
            QCoreApplication.sendEvent(self.canvas.viewport(), event)
        self.assertNotEqual(self.canvas.extent(), extent)
        self.canvas.zoomByFactor(0.5)
        self.canvas.refresh()
        self.app.processEvents()
        self.assertEqual(self.tool.anchors, anchors)
        self.assertEqual(bytes(self.tool.session.geometry.asWkb()), expected)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertTrue(self.tool.draft_band.isVisible())
        self.assertEqual(self.tool.snap_to_itself(1008, 1992.25, 0.5), (1008, 1992))
        self.assertTrue(self.tool.finish_session())
        self.assertEqual(
            bytes(next(self.vector.getFeatures()).geometry().asWkb()), expected
        )

    def test_color_snap_and_settings(self):
        self.plugin.ensure_trace_color_enabled()
        self.plugin.set_trace_color_from_tool(QColor("black"))
        self.tool.snap_tolerance_changed(3)
        self.assertEqual(self.tool.snap(7, 8), (8, 8))
        self.assertEqual(QSettings().value("RasterScribe/color/value"), "#ff000000")

    def test_keyboard_filter(self):
        self.accept(8, 2)
        for key, mode in (
            (Qt.Key.Key_A, TracingModes.LINE),
            (Qt.Key.Key_D, TracingModes.DENSE_LINE),
        ):
            event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
            QCoreApplication.sendEvent(self.iface.layerTreeView(), event)
            self.assertEqual(self.tool.tracing_mode, mode)

    def test_different_project_and_vector_crs(self):
        self.project.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        self.canvas.setDestinationCrs(self.project.crs())
        self.tool.raster_layer_has_changed(self.raster)
        self.add_anchors()
        first = self.tool.to_coords(8, 2)
        self.assertEqual(self.tool.to_indexes(first.x(), first.y()), (8, 2))
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertTrue(self.tool.finish_session())
        geometry = next(self.vector.getFeatures()).geometry()
        transform = QgsCoordinateTransform(
            self.project.crs(), self.vector.crs(), self.project
        )
        expected = transform.transform(first)
        vertex = next(geometry.vertices())
        self.assertAlmostEqual(vertex.x(), expected.x(), places=6)
        self.assertAlmostEqual(vertex.y(), expected.y(), places=6)
        self.assertAlmostEqual(geometry.length(), 11, places=6)

    def test_repeated_unload_deletes_main_window_actions(self):
        for cycle in range(3):
            with self.subTest(cycle=cycle):
                if cycle:
                    self.plugin = classFactory(self.iface)
                    self.plugin.initGui()
                    self.plugin.run()
                    self.tools.append(self.plugin.tool_identify)
                actions = list(self.plugin.actions)
                self.assertEqual(len(actions), 1)
                self.assertIs(actions[0].parent(), self.window)
                self.plugin.unload()
                self.plugin.unload()  # Repeated unload remains harmless.
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.assertTrue(all(sip.isdeleted(action) for action in actions))
                self.assertEqual(self.plugin.actions, [])

"""Integration coverage for the same plugin under Qt5/QGIS 3 and Qt6/QGIS 4."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from qgis.PyQt.QtCore import (
    QCoreApplication, QEvent, QPoint, QPointF, QSettings, Qt,
)
from qgis.PyQt.QtGui import QColor, QKeyEvent, QMouseEvent
from qgis.core import (
    QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsProject, QgsRasterLayer, QgsVectorLayer,
)
from qgis.gui import QgsMapMouseEvent

from .. import classFactory
from ..pointtool import Anchor, TracingModes
from ..utils import RasterSampler
from .utilities import create_rgb_raster, get_qgis_app, wait_until


class CompatibilityTest(unittest.TestCase):
    """Use real providers, widgets, task threads and geometry operations."""

    def setUp(self):
        self.app, self.canvas, self.iface, self.window = get_qgis_app()
        self.project = QgsProject.instance()
        self.project.clear()
        QSettings().clear()
        self.project.setCrs(QgsCoordinateReferenceSystem('EPSG:3857'))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        raster_path = Path(self.directory.name) / 'line.tif'
        create_rgb_raster(raster_path)
        self.raster = QgsRasterLayer(str(raster_path), 'Trace raster')
        self.vector = QgsVectorLayer(
            'MultiLineString?crs=EPSG:3857', 'Traced lines', 'memory')
        self.assertTrue(self.raster.isValid())
        self.assertTrue(self.vector.isValid())
        self.project.addMapLayer(self.raster)
        self.project.addMapLayer(self.vector)
        self.canvas.setDestinationCrs(self.project.crs())
        self.canvas.setLayers([self.vector, self.raster])
        self.canvas.setExtent(self.raster.extent())
        self.canvas.setMapTool(self.iface.pan_tool)
        self.iface.setActiveLayer(self.vector)
        self.assertTrue(self.vector.startEditing())
        self.plugin = classFactory(self.iface)
        self.tools = []
        self.addCleanup(self.cleanup_plugin)
        self.plugin.initGui()
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.tools.append(self.tool)
        self.plugin.dockwidget.mMapLayerComboBox.setLayer(self.raster)
        self.tool.raster_layer_has_changed(self.raster)
        self.tool.smooth_line = False
        self.expected_path = [(8, column) for column in range(2, 14)]

    def cleanup_plugin(self):
        for tool in self.tools:
            tool.task_controller.cancel()
            tool.clear_preview()
        manager = QgsApplication.taskManager()
        wait_until(lambda: manager.countActiveTasks() == 0)
        self.app.processEvents()
        dock = self.plugin.dockwidget
        if dock is not None:
            dock.close()
            self.window.removeDockWidget(dock)
            dock.deleteLater()
        toolbar = self.plugin.toolbar
        self.plugin.unload()
        self.window.removeToolBar(toolbar)
        toolbar.deleteLater()
        for tool in self.tools:
            for item in (tool.rubber_band, tool.marker_snap,
                         tool.preview_controller._rubber_band, *tool.markers):
                self.canvas.scene().removeItem(item)
            tool.raster_context.raster_sampler = None
            tool.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.canvas.setLayers([])
        self.project.clear()
        self.app.processEvents()

    def add_anchors(self):
        for row, column in (self.expected_path[0], self.expected_path[-1]):
            point = self.tool.to_coords(row, column)
            self.tool.add_anchor_points(point.x(), point.y(), row, column)

    def test_start_close_and_reopen(self):
        self.assertTrue(self.plugin.pluginIsActive)
        self.assertIs(self.canvas.mapTool(), self.tool)
        self.assertFalse(self.plugin.actions[0].icon().pixmap(24, 24).isNull())
        self.assertIs(self.tool.get_current_vector_layer(), self.vector)
        self.plugin.preview_color_changed(QColor('#80402010'))
        self.assertEqual(QSettings().value('RasterTracer/preview/color'),
                         '#80402010')
        dock = self.plugin.dockwidget
        dock.close()
        self.assertFalse(self.plugin.pluginIsActive)
        self.assertIs(self.canvas.mapTool(), self.iface.pan_tool)
        self.window.removeDockWidget(dock)
        dock.deleteLater()
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.tools.append(self.tool)
        self.assertIs(self.canvas.mapTool(), self.tool)

    def test_read_byte_raster_as_float(self):
        sampler = RasterSampler(self.raster, self.project)
        bands, origin, shape = sampler.read_window(7, 10, 1, 15)
        self.assertEqual(origin, (7, 1))
        self.assertEqual(shape, (3, 14))
        for band in bands:
            self.assertEqual(band.dtype, np.dtype('float32'))
            self.assertEqual(float(band[1, 1]), 0)
            self.assertEqual(float(band[0, 1]), 255)

    def test_trace_known_line(self):
        path, cost = self.tool.trace_over_image((8, 2), (8, 13))
        self.assertEqual(path, self.expected_path)
        self.assertEqual(cost, 0)

    def test_background_trace_and_save(self):
        self.add_anchors()
        self.tool.trace_over_image(
            (8, 2), (8, 13), do_it_as_task=True, vlayer=self.vector)
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertTrue(self.vector.commitChanges())
        geometry = next(self.vector.getFeatures()).geometry()
        self.assertFalse(geometry.isEmpty())
        self.assertAlmostEqual(geometry.length(), 11)

    def test_preview_and_cancel_restart(self):
        preview = self.tool.preview_controller
        preview.queue((8, 2), (8, 13), QPoint(10, 20))
        preview.ensure_inflight_started()
        self.tool.clear_preview()
        preview.queue((8, 2), (8, 13), QPoint(10, 20))
        preview.ensure_inflight_started()
        wait_until(lambda: preview._cached_path is not None)
        self.assertEqual(
            preview.take_path_if_valid((8, 2), (8, 13), QPoint(10, 20)),
            self.expected_path)
        self.tool.clear_preview()
        self.assertIsNone(preview._cached_path)
        self.assertFalse(preview._rubber_band.isVisible())

    def test_cancel_trace_then_restart(self):
        self.add_anchors()
        self.tool.trace_over_image(
            (8, 2), (8, 13), do_it_as_task=True, vlayer=self.vector)
        self.tool._cancel_inflight_segment()
        manager = QgsApplication.taskManager()
        wait_until(lambda: manager.countActiveTasks() == 0)
        self.app.processEvents()
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(len(self.tool.anchors), 1)
        point = self.tool.to_coords(8, 13)
        self.tool.add_anchor_points(point.x(), point.y(), 8, 13)
        self.tool.trace_over_image(
            (8, 2), (8, 13), do_it_as_task=True, vlayer=self.vector)
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 1)

    def test_mouse_clicks_create_and_finish_line(self):
        def click(row, column, button):
            point = self.tool.to_coords(row, column)
            screen = self.tool.toCanvasCoordinates(point)
            event = QMouseEvent(
                QEvent.Type.MouseButtonRelease, QPointF(screen),
                button, button, Qt.KeyboardModifier.NoModifier)
            self.tool.canvasReleaseEvent(QgsMapMouseEvent(self.canvas, event))

        click(8, 2, Qt.MouseButton.LeftButton)
        click(8, 13, Qt.MouseButton.LeftButton)
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 1)
        geometry = next(self.vector.getFeatures()).geometry()
        # Synthetic clicks are rounded to integer screen pixels, unlike
        # the exact map-coordinate assertions in the background trace test.
        self.assertAlmostEqual(
            geometry.length(), 11, delta=2 * self.canvas.mapUnitsPerPixel())
        click(8, 13, Qt.MouseButton.RightButton)
        self.assertFalse(self.tool.has_active_trace())

    def test_color_snap_and_settings(self):
        self.plugin.ensure_trace_color_enabled()
        self.plugin.set_trace_color_from_tool(QColor('black'))
        self.tool._ensure_window_for_indices([(7, 8)], reason='test-snap')
        self.tool.snap_tolerance_changed(3)
        self.assertEqual(self.tool.snap(7, 8), (8, 8))
        self.assertEqual(QSettings().value('RasterTracer/color/value'),
                         '#ff000000')

    def test_keyboard_filter(self):
        self.tool.anchors = [Anchor(1002.5, 1991.5, 8, 2)]
        for key, mode in ((Qt.Key.Key_A, TracingModes.LINE),
                          (Qt.Key.Key_D, TracingModes.DENSE_LINE)):
            event = QKeyEvent(QEvent.Type.KeyPress, key,
                              Qt.KeyboardModifier.NoModifier)
            QCoreApplication.sendEvent(self.iface.layerTreeView(), event)
            self.assertEqual(self.tool.tracing_mode, mode)

    def test_different_project_and_vector_crs(self):
        self.project.setCrs(QgsCoordinateReferenceSystem('EPSG:4326'))
        self.canvas.setDestinationCrs(self.project.crs())
        self.tool.raster_layer_has_changed(self.raster)
        self.add_anchors()
        first = self.tool.to_coords(8, 2)
        self.assertEqual(self.tool.to_indexes(first.x(), first.y()), (8, 2))
        path, _ = self.tool.trace_over_image((8, 2), (8, 13))
        self.tool.draw_path(path, self.vector)
        geometry = next(self.vector.getFeatures()).geometry()
        transform = QgsCoordinateTransform(
            self.project.crs(), self.vector.crs(), self.project)
        expected = transform.transform(first)
        vertex = next(geometry.vertices())
        self.assertAlmostEqual(vertex.x(), expected.x(), places=6)
        self.assertAlmostEqual(vertex.y(), expected.y(), places=6)
        self.assertAlmostEqual(geometry.length(), 11, places=6)

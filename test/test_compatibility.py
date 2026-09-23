"""Integration coverage for the same plugin under Qt5/QGIS 3 and Qt6/QGIS 4."""

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
)
from qgis.gui import QgsMapMouseEvent
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QEvent,
    QPoint,
    QPointF,
    QSettings,
    Qt,
)
from qgis.PyQt.QtGui import QColor, QKeyEvent, QMouseEvent

from ..pointtool import TracingModes
from ..utils import RasterSampler
from .tracing_fixture import TraceFixture
from .utilities import wait_until


class CompatibilityTest(TraceFixture):
    def test_start_close_and_reopen(self):
        self.assertTrue(self.plugin.pluginIsActive)
        self.assertIs(self.canvas.mapTool(), self.tool)
        self.assertFalse(self.plugin.actions[0].icon().pixmap(24, 24).isNull())
        self.assertIs(self.tool.get_current_vector_layer(), self.vector)
        self.plugin.preview_color_changed(QColor("#80402010"))
        self.assertEqual(QSettings().value("RasterTracer/preview/color"), "#80402010")
        dock = self.plugin.dockwidget
        dock.close()
        self.assertFalse(self.plugin.pluginIsActive)
        self.assertIs(self.canvas.mapTool(), self.iface.pan_tool)
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
            self.assertEqual(band.dtype, np.dtype("float32"))
            self.assertEqual(float(band[1, 1]), 0)
            self.assertEqual(float(band[0, 1]), 255)

    def test_trace_known_line(self):
        path, cost = self.tool.trace_over_image((8, 2), (8, 13))
        self.assertEqual(path, self.expected_path)
        self.assertEqual(cost, 0)

    def test_background_trace_and_save(self):
        self.add_anchors()
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertTrue(self.vector.commitChanges())
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
        self.assertEqual(self.vector.featureCount(), 1)

    def test_mouse_clicks_create_and_finish_line(self):
        def click(row, column, button):
            point = self.tool.to_coords(row, column)
            screen = self.tool.toCanvasCoordinates(point)
            event = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(screen),
                button,
                button,
                Qt.KeyboardModifier.NoModifier,
            )
            self.tool.canvasReleaseEvent(QgsMapMouseEvent(self.canvas, event))

        click(8, 2, Qt.MouseButton.LeftButton)
        click(8, 13, Qt.MouseButton.LeftButton)
        wait_until(lambda: not self.tool.tracking_is_active)
        self.assertEqual(self.vector.featureCount(), 1)
        geometry = next(self.vector.getFeatures()).geometry()
        # Synthetic clicks are rounded to integer screen pixels, unlike
        # the exact map-coordinate assertions in the background trace test.
        self.assertAlmostEqual(
            geometry.length(), 11, delta=2 * self.canvas.mapUnitsPerPixel()
        )
        click(8, 13, Qt.MouseButton.RightButton)
        self.assertFalse(self.tool.has_active_trace())

    def test_color_snap_and_settings(self):
        self.plugin.ensure_trace_color_enabled()
        self.plugin.set_trace_color_from_tool(QColor("black"))
        self.tool.snap_tolerance_changed(3)
        self.assertEqual(self.tool.snap(7, 8), (8, 8))
        self.assertEqual(QSettings().value("RasterTracer/color/value"), "#ff000000")

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
        geometry = next(self.vector.getFeatures()).geometry()
        transform = QgsCoordinateTransform(
            self.project.crs(), self.vector.crs(), self.project
        )
        expected = transform.transform(first)
        vertex = next(geometry.vertices())
        self.assertAlmostEqual(vertex.x(), expected.x(), places=6)
        self.assertAlmostEqual(vertex.y(), expected.y(), places=6)
        self.assertAlmostEqual(geometry.length(), 11, places=6)

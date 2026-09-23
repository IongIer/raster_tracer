"""Integration coverage for the same plugin under Qt5/QGIS 3 and Qt6/QGIS 4."""

import tempfile
import unittest
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.gui import QgsMapMouseEvent
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QEvent,
    QPointF,
    QSettings,
    Qt,
)
from qgis.PyQt.QtGui import QMouseEvent

from .. import classFactory
from .utilities import create_rgb_raster, get_qgis_app, wait_until


class TraceFixture(unittest.TestCase):
    """Use real providers, widgets, task threads and geometry operations."""

    def setUp(self):
        self.app, self.canvas, self.iface, self.window = get_qgis_app()
        self.project = QgsProject.instance()
        self.project.clear()
        QSettings().clear()
        self.project.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        raster_path = Path(self.directory.name) / "line.tif"
        create_rgb_raster(raster_path)
        self.raster = QgsRasterLayer(str(raster_path), "Trace raster")
        self.vector = QgsVectorLayer(
            "MultiLineString?crs=EPSG:3857", "Traced lines", "memory"
        )
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
        self.plugin.unload()
        manager = QgsApplication.taskManager()
        wait_until(lambda: manager.countActiveTasks() == 0)
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.canvas.setLayers([])
        self.project.clear()
        self.app.processEvents()

    def click(self, row, column, button=Qt.MouseButton.LeftButton):
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

    def accept(self, row, column):
        self.tool.accept_click(self.tool.to_coords(row, column))

    def add_anchors(self):
        for row, column in (self.expected_path[0], self.expected_path[-1]):
            self.accept(row, column)

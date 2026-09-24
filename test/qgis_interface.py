"""Small QgsInterface substitute backed by real QGIS and Qt widgets."""

from qgis.core import QgsLayerTreeModel, QgsProject
from qgis.gui import QgsLayerTreeView, QgsMessageBar
from qgis.PyQt.QtCore import QObject
from qgis.PyQt.QtWidgets import QMenu, QToolBar


class QgisInterface(QObject):
    """Implement the interface methods used by Raster Scribe."""

    def __init__(self, canvas, window):
        super().__init__(window)
        self.canvas = canvas
        self.window = window
        self.tree = QgsLayerTreeView(window)
        self.model = QgsLayerTreeModel(QgsProject.instance().layerTreeRoot(), self.tree)
        self.tree.setModel(self.model)
        self.messages = QgsMessageBar(window)
        self.menu = QMenu(window)
        self.menu.addAction("Undo", self._undo)

    def _undo(self):
        layer = self.activeLayer()
        if layer is not None:
            layer.undoStack().undo()

    def activeLayer(self):
        return self.tree.currentLayer()

    def setActiveLayer(self, layer):
        self.tree.setCurrentLayer(layer)

    def layerTreeView(self):
        return self.tree

    def messageBar(self):
        return self.messages

    def editMenu(self):
        return self.menu

    def addToolBar(self, name):
        return self.window.addToolBar(name)

    def addPluginToMenu(self, name, action):
        self.window.menuBar().addAction(action)

    def removePluginMenu(self, name, action):
        self.window.menuBar().removeAction(action)

    def addPluginToRasterMenu(self, name, action):
        self.window.menuBar().addAction(action)

    def removePluginRasterMenu(self, name, action):
        self.window.menuBar().removeAction(action)

    def removeToolBarIcon(self, action):
        for toolbar in self.window.findChildren(QToolBar):
            toolbar.removeAction(action)

    def mapCanvas(self):
        return self.canvas

    def mainWindow(self):
        return self.window

    def addDockWidget(self, area, dock_widget):
        self.window.addDockWidget(area, dock_widget)

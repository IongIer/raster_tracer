"""Shared real QGIS application and temporary-fixture helpers."""

import gc
import time
from pathlib import Path

from .. import __file__ as plugin_file

PLUGIN_DIR = Path(plugin_file).parent
QGIS_APP = None
CANVAS = None
PARENT = None
IFACE = None


def get_qgis_app():
    """Start QGIS once; missing QGIS is a test failure, never a silent skip."""
    from qgis.gui import QgsMapCanvas, QgsMapToolPan
    from qgis.PyQt.QtWidgets import QMainWindow
    from qgis.testing import start_app

    from .qgis_interface import QgisInterface

    global QGIS_APP, CANVAS, PARENT, IFACE
    if QGIS_APP is None:
        QGIS_APP = start_app()
        PARENT = QMainWindow()
        CANVAS = QgsMapCanvas(PARENT)
        CANVAS.resize(400, 400)
        IFACE = QgisInterface(CANVAS, PARENT)
        IFACE.pan_tool = QgsMapToolPan(CANVAS)
        CANVAS.setMapTool(IFACE.pan_tool)
    return QGIS_APP, CANVAS, IFACE, PARENT


def stop_qgis_widgets():
    """Destroy canvas widgets before QGIS unloads its providers at exit."""
    from qgis.core import QgsProject
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import QCoreApplication, QEvent

    global CANVAS, PARENT, IFACE
    if CANVAS is not None:
        CANVAS.unsetMapTool(CANVAS.mapTool())
        CANVAS.stopRendering()
        CANVAS.setLayers([])
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        # Release controller/reference cycles while Qt and providers exist.
        gc.collect()
        sip.delete(PARENT)
        CANVAS = PARENT = IFACE = None
        QgsProject.instance().clear()


def wait_until(predicate, timeout=10):
    """Process Qt signals while waiting for background QGIS tasks."""
    from qgis.PyQt.QtTest import QTest

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for QGIS task")
        QTest.qWait(10)


def create_rgb_raster(path):
    """Create a tiny black line on white; caller owns directory cleanup."""
    import numpy as np
    from osgeo import gdal, osr

    pixels = np.full((16, 16), 255, dtype=np.uint8)
    pixels[8, 2:14] = 0
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3857)
    write_raster(
        path,
        (pixels,) * 3,
        gdal.GDT_Byte,
        geotransform=(1000, 1, 0, 2000, 0, -1),
        projection=crs.ExportToWkt(),
    )


def write_raster(
    path,
    bands,
    data_type,
    *,
    nodata=None,
    mask=None,
    geotransform=None,
    projection=None,
):
    """Write a GeoTIFF and release it before readers open it."""
    from osgeo import gdal

    height, width = bands[0].shape
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), width, height, len(bands), data_type
    )
    try:
        dataset.SetGeoTransform(
            geotransform if geotransform is not None else (0, 1, 0, height, 0, -1)
        )
        if projection is not None:
            dataset.SetProjection(projection)
        for number, values in enumerate(bands, 1):
            band = dataset.GetRasterBand(number)
            band.WriteArray(values)
            if nodata is not None:
                band.SetNoDataValue(nodata)
        if mask is not None:
            dataset.CreateMaskBand(gdal.GMF_PER_DATASET)
            dataset.GetRasterBand(1).GetMaskBand().WriteArray(mask)
        dataset.FlushCache()
    finally:
        band = None
        dataset = None

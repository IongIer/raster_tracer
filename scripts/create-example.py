"""Recreate the tutorial project with QGIS's Python; overwrites example files."""

import os
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from osgeo import gdal, osr
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsLineSymbol,
    QgsProject,
    QgsRasterLayer,
    QgsVectorFileWriter,
    QgsVectorLayer,
)


def main():
    root = Path(__file__).resolve().parent.parent / "documentation" / "example"
    root.mkdir(parents=True, exist_ok=True)
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    project.setCrs(crs)

    # A dark, gently curved line, with enough width for a first tracing exercise.
    rows, columns = np.indices((96, 160))
    center = 48 + 22 * np.sin((columns - 12) * np.pi / 68)
    line = (columns >= 12) & (columns <= 147) & (abs(rows - center) <= 2)
    values = np.where(line, 15, 255).astype(np.uint8)
    path = root / "line.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), 160, 96, 3, gdal.GDT_Byte, options=["COMPRESS=DEFLATE"]
    )
    dataset.SetGeoTransform((1000, 1, 0, 2000, 0, -1))
    spatial = osr.SpatialReference()
    spatial.ImportFromEPSG(3857)
    dataset.SetProjection(spatial.ExportToWkt())
    for number in range(1, 4):
        dataset.GetRasterBand(number).WriteArray(values)
    dataset = None
    raster = QgsRasterLayer(str(path), "Example raster")

    target = root / "traced-lines.gpkg"
    target.unlink(missing_ok=True)
    empty = QgsVectorLayer("MultiLineString?crs=EPSG:3857", "Traced lines", "memory")
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = "traced_lines"
    outcome = QgsVectorFileWriter.writeAsVectorFormatV3(
        empty, str(target), project.transformContext(), options
    )
    if outcome[0] != QgsVectorFileWriter.NoError:
        raise RuntimeError(outcome)
    vector = QgsVectorLayer(
        str(target) + "|layername=traced_lines", "Traced lines", "ogr"
    )
    if not raster.isValid() or not vector.isValid():
        raise RuntimeError("Could not create tutorial layers")
    vector.renderer().setSymbol(
        QgsLineSymbol.createSimple({"line_color": "0,100,220", "line_width": "0.5"})
    )
    project.addMapLayer(raster)
    project.addMapLayer(vector)
    from qgis.core import QgsReferencedRectangle

    project.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(raster.extent(), crs)
    )
    if not project.write(str(root / "example.qgs")):
        raise RuntimeError("Could not save tutorial project")
    project.clear()
    # Release providers before exitQgis destroys the provider registry.
    del vector, raster, empty
    app.exitQgis()
    # Remove cached statistics and any empty attachment archive.
    (root / "line.tif.aux.xml").unlink(missing_ok=True)
    attachments = root / "example_attachments.zip"
    if attachments.exists():
        with zipfile.ZipFile(attachments) as archive:
            empty_attachments = not archive.namelist()
        if empty_attachments:
            attachments.unlink()
    print(root)


if __name__ == "__main__":
    main()

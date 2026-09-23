"""GUI-owned raster metadata, coordinate transforms and small interactive reads."""

from math import floor

import numpy as np
from osgeo import gdal
from qgis.core import QgsCoordinateTransform

from .exceptions import InvalidRasterError, OutsideMapError
from .pointtool_session import RasterSourceSpec, as_xy, new_id


def get_indxs_from_raster_coords(geo_ref, xy):
    x, y = as_xy(xy)
    left, top, dx, dy = geo_ref
    return floor((top - y) / dy), floor((x - left) / dx)


def get_coords_from_raster_indxs(geo_ref, ij):
    i, j = ij
    left, top, dx, dy = geo_ref
    return left + (j + 0.5) * dx, top - (i + 0.5) * dy


class RasterSampler:
    def __init__(self, layer, project_instance, working_crs=None):
        if not layer.isValid():
            raise InvalidRasterError("Invalid raster layer")
        self.raster_path = layer.source()
        self.dataset = gdal.Open(self.raster_path)
        if self.dataset is None or self.dataset.RasterCount < 3:
            raise InvalidRasterError("An RGB raster is required")
        mapping = self.dataset.GetGeoTransform(can_return_null=True)
        if mapping is None:
            # Unreferenced images use the provider's default pixel coordinates.
            extent = layer.dataProvider().extent()
            self.geo_ref = (
                extent.xMinimum(),
                extent.yMaximum(),
                layer.rasterUnitsPerPixelX(),
                layer.rasterUnitsPerPixelY(),
            )
        else:
            if mapping[2] != 0 or mapping[4] != 0 or mapping[1] <= 0 or mapping[5] >= 0:
                raise InvalidRasterError(
                    "Rotated or skewed raster mapping is unsupported"
                )
            self.geo_ref = (mapping[0], mapping[3], mapping[1], -mapping[5])
        self.height, self.width = self.dataset.RasterYSize, self.dataset.RasterXSize
        self.source = RasterSourceSpec(
            self.raster_path, self.height, self.width, new_id()
        )
        working_crs = working_crs or project_instance.crs()
        self.trfm_from_src = QgsCoordinateTransform(
            layer.crs(), working_crs, project_instance.transformContext()
        )
        self.trfm_to_src = QgsCoordinateTransform(
            working_crs, layer.crs(), project_instance.transformContext()
        )
        self._small_cache = None

    def to_indexes(self, x, y):
        indexes = get_indxs_from_raster_coords(
            self.geo_ref, self.trfm_to_src.transform(x, y)
        )
        self.validate(*indexes)
        return indexes

    def validate(self, i, j):
        if not (0 <= i < self.height and 0 <= j < self.width):
            raise OutsideMapError("Endpoint outside raster")

    def to_coords(self, i, j):
        return self.trfm_from_src.transform(
            *get_coords_from_raster_indxs(self.geo_ref, (i, j))
        )

    def read_window(self, i_min, i_max, j_min, j_max, dtype=np.float32):
        # Synchronous helper for small GUI reads and integration probes only.
        from .pointtool_raster import read_rgb

        bounds = (
            max(0, i_min),
            min(self.height, i_max),
            max(0, j_min),
            min(self.width, j_max),
        )
        bands, _ = read_rgb(self.dataset, self.source, bounds, lambda: False)
        shape = (bounds[1] - bounds[0], bounds[3] - bounds[2])
        return (
            tuple(band.astype(dtype) for band in bands),
            (bounds[0], bounds[2]),
            shape,
        )

    def read_small(self, i, j, radius=0):
        from .pointtool_raster import read_rgb

        self.validate(i, j)
        if not 0 <= radius <= 99:
            raise ValueError("Color snap radius must be between 0 and 99")
        bounds = (
            max(0, i - radius),
            min(self.height, i + radius + 1),
            max(0, j - radius),
            min(self.width, j + radius + 1),
        )
        if self._small_cache is None or self._small_cache[0] != bounds:
            bands, valid = read_rgb(self.dataset, self.source, bounds, lambda: False)
            self._small_cache = (bounds, bands, valid)
        return self._small_cache

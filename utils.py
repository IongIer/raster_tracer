import os
import time

from osgeo import gdal
from qgis.core import QgsCoordinateTransform, QgsMessageLog, Qgis
import numpy as np


PROFILE_ENABLED = os.environ.get("RASTER_TRACER_PROFILE", "0") == "1"


class PossiblyIndexedImageError(Exception):
    pass


def get_indxs_from_raster_coords(geo_ref, xy):
    x, y = xy
    top_left_x, top_left_y, we_resolution, ns_resolution = geo_ref
    i = int((y - top_left_y) / ns_resolution) * -1
    j = int((x - top_left_x) / we_resolution)
    return i, j


def get_coords_from_raster_indxs(geo_ref, ij):
    i, j = ij
    top_left_x, top_left_y, we_resolution, ns_resolution = geo_ref
    y = (top_left_y - (i + 0.5) * ns_resolution)
    x = top_left_x - (j + 0.5) * we_resolution * -1
    return x, y

class RasterSampler:
    def __init__(self, layer, project_instance):
        self.layer = layer
        self.project_instance = project_instance
        self.provider = layer.dataProvider()
        extent = self.provider.extent()
        project_crs = project_instance.crs()
        self.trfm_from_src = QgsCoordinateTransform(
            self.provider.crs(),
            project_crs,
            project_instance,
        )
        self.trfm_to_src = QgsCoordinateTransform(
            project_crs,
            self.provider.crs(),
            project_instance,
        )

        dx = layer.rasterUnitsPerPixelX()
        dy = layer.rasterUnitsPerPixelY()
        top_left_x = extent.xMinimum()
        top_left_y = extent.yMaximum()

        self.geo_ref = (top_left_x, top_left_y, dx, dy)

        self.raster_path = layer.source()
        self.dataset = gdal.Open(self.raster_path)
        if self.dataset is None:
            raise PossiblyIndexedImageError

        if self.dataset.RasterCount < 3:
            raise PossiblyIndexedImageError

        self.height = self.dataset.RasterYSize
        self.width = self.dataset.RasterXSize

    def to_indexes(self, x, y):
        return get_indxs_from_raster_coords(
            self.geo_ref,
            self.trfm_to_src.transform(x, y),
        )

    def to_coords(self, i, j):
        return self.trfm_from_src.transform(
            *get_coords_from_raster_indxs(self.geo_ref, (i, j)),
        )

    def to_coords_provider(self, i, j):
        return get_coords_from_raster_indxs(self.geo_ref, (i, j))

    def to_coords_provider2(self, x, y):
        return self.trfm_to_src.transform(x, y)

    def read_window(self, i_min, i_max, j_min, j_max, dtype=np.float32):
        start_time = time.perf_counter() if PROFILE_ENABLED else None

        i_min_clamped = max(0, min(i_min, self.height))
        j_min_clamped = max(0, min(j_min, self.width))
        i_max_clamped = max(0, min(i_max, self.height))
        j_max_clamped = max(0, min(j_max, self.width))

        height = i_max_clamped - i_min_clamped
        width = j_max_clamped - j_min_clamped

        if height <= 0 or width <= 0:
            return None, (i_min_clamped, j_min_clamped), (0, 0)

        try:
            band1 = self.dataset.GetRasterBand(1).ReadAsArray(
                j_min_clamped,
                i_min_clamped,
                width,
                height,
            )
            band2 = self.dataset.GetRasterBand(2).ReadAsArray(
                j_min_clamped,
                i_min_clamped,
                width,
                height,
            )
            band3 = self.dataset.GetRasterBand(3).ReadAsArray(
                j_min_clamped,
                i_min_clamped,
                width,
                height,
            )
        except AttributeError:
            raise PossiblyIndexedImageError

        bands = (
            np.array(band1, dtype=dtype, copy=False),
            np.array(band2, dtype=dtype, copy=False),
            np.array(band3, dtype=dtype, copy=False),
        )

        if PROFILE_ENABLED:
            duration = time.perf_counter() - start_time
            total_bytes = sum(b.nbytes for b in bands)
            QgsMessageLog.logMessage(
                (
                    "[profiling] read_window "
                    f"origin=({i_min_clamped},{j_min_clamped}) "
                    f"shape=({height},{width}) duration={duration:.2f}s "
                    f"bytes={(total_bytes / (1024 ** 2)):.1f}"
                ),
                "RasterTracer",
                Qgis.Info,
            )

        return bands, (i_min_clamped, j_min_clamped), (height, width)


def get_whole_raster(layer, project_instance):
    start_time = time.perf_counter() if PROFILE_ENABLED else None
    sampler = RasterSampler(layer, project_instance)
    if PROFILE_ENABLED:
        duration = time.perf_counter() - start_time
        layer_name = layer.name() if hasattr(layer, "name") else ""
        if not layer_name:
            layer_name = layer.source()
        QgsMessageLog.logMessage(
            (
                "[profiling] get_whole_raster sampler "
                f"'{layer_name}': {duration:.2f}s size="
                f"({sampler.height},{sampler.width})"
            ),
            "RasterTracer",
            Qgis.Info,
        )
    return sampler

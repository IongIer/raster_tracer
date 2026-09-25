"""Numerical raster validity, budgets, immutable cache and endpoint regressions."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor

from ..exceptions import (
    InvalidRasterError,
    OutsideMapError,
    ResourceLimitError,
    TraceCancelled,
)
from ..pointtool_raster import (
    MAX_ARRAY_BYTES,
    color_cost,
    prepare_snapshot,
    read_rgb,
    validate_budget,
    window_bounds,
)
from ..pointtool_session import RasterSourceSpec, WorkerRequest
from ..pointtool_tasks import execute_request
from ..utils import RasterSampler, get_indxs_from_raster_coords
from .tracing_fixture import TraceFixture
from .utilities import get_qgis_app


class RasterPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qgis_app()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "validity.tif"

    def source(self, values, nodata=None):
        height, width = values.shape
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(self.path), width, height, 3, gdal.GDT_Float64
        )
        dataset.SetGeoTransform((0, 1, 0, height, 0, -1))
        for index in (1, 2, 3):
            band = dataset.GetRasterBand(index)
            band.WriteArray(values)
            if nodata is not None:
                band.SetNoDataValue(nodata)
        dataset = None
        return RasterSourceSpec(str(self.path), height, width, 1)

    def test_unreferenced_rgb_uses_provider_pixel_coordinates(self):
        values = np.arange(6, dtype=np.uint8).reshape(2, 3)
        dataset = gdal.GetDriverByName("MEM").Create("", 3, 2, 3, gdal.GDT_Byte)
        for band in (1, 2, 3):
            dataset.GetRasterBand(band).WriteArray(values)
        for driver, extension in (("GTiff", ".tif"), ("PNG", ".png")):
            with self.subTest(driver=driver):
                path = self.path.with_suffix(extension)
                output = gdal.GetDriverByName(driver).CreateCopy(str(path), dataset)
                output.FlushCache()
                output = None
                layer = QgsRasterLayer(str(path), "unreferenced")
                self.assertTrue(layer.isValid())
                project = QgsProject.instance()
                layer.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
                sampler = RasterSampler(layer, project, layer.crs())
                self.assertIsNone(sampler.dataset.GetGeoTransform(can_return_null=True))
                extent = layer.dataProvider().extent()
                dx, dy = extent.width() / 3, extent.height() / 2
                for row in range(2):
                    for col in range(3):
                        x = extent.xMinimum() + (col + 0.5) * dx
                        y = extent.yMaximum() - (row + 0.5) * dy
                        self.assertEqual(sampler.to_indexes(x, y), (row, col))
                        self.assertEqual(sampler.to_coords(row, col), QgsPointXY(x, y))
                        _, bands, valid = sampler.read_small(row, col)
                        self.assertTrue(valid[0, 0])
                        self.assertEqual(
                            [b[0, 0] for b in bands], [values[row, col]] * 3
                        )
                with self.assertRaises(OutsideMapError):
                    sampler.to_indexes(extent.xMinimum() - dx / 2, extent.yMaximum())

    def test_floor_edges_and_minimum_window_at_all_corners(self):
        geo = (0, 1000, 1, 1)
        self.assertEqual(get_indxs_from_raster_coords(geo, (-0.01, 999.5)), (0, -1))
        self.assertEqual(get_indxs_from_raster_coords(geo, (0.5, 1000.01)), (-1, 0))
        self.assertEqual(get_indxs_from_raster_coords(geo, (1000.01, 0.5)), (999, 1000))
        self.assertEqual(get_indxs_from_raster_coords(geo, (0.5, -0.01)), (1000, 0))
        for point in ((0, 0), (0, 999), (999, 0), (999, 999)):
            top, bottom, left, right = window_bounds([point], 1000, 1000, padding=0)
            self.assertEqual((bottom - top, right - left), (512, 512))
        for point in ((-1, 0), (0, -1), (1000, 0), (0, 1000)):
            with self.assertRaises(OutsideMapError):
                window_bounds([point], 1000, 1000)

    def test_nodata_nonfinite_masks_and_cost_range(self):
        values = np.array([[0, 1, 2, 3], [np.nan, np.inf, -999, 10], [1, 1, 1, 1]])
        # GDAL's all-valid mask does not rule out NaN or infinity in float bands.
        source = self.source(values)
        dataset = gdal.Open(source.path)
        _, valid = read_rgb(dataset, source, (0, 3, 0, 4), lambda: False)
        self.assertEqual(valid[1].tolist(), [False, False, True, True])
        dataset = None
        source = self.source(values, -999)
        dataset = gdal.Open(source.path)
        bands, valid = read_rgb(dataset, source, (0, 3, 0, 4), lambda: False)
        self.assertEqual(valid[1].tolist(), [False, False, False, True])
        costs = color_cost(bands, valid, (0, 0, 0))
        self.assertEqual(costs.dtype, np.dtype("int64"))
        self.assertTrue((costs >= 0).all())
        self.assertEqual(costs[0].tolist(), [0, 3, 12, 27])
        for array in (*bands, valid, costs):
            self.assertFalse(array.flags.writeable)
        for color in ((np.nan, 0, 0), (np.inf, 0, 0)):
            with self.assertRaises(InvalidRasterError):
                color_cost(bands, valid, color)
        huge = np.array([[2.0**31]])
        with self.assertRaises(InvalidRasterError):
            color_cost((huge, huge, np.zeros((1, 1))), np.ones((1, 1), bool), (0, 0, 0))
        work = WorkerRequest(1, source, (0, 3, 0, 4), (0, 0), (1, 0), None, False)
        self.assertEqual(execute_request(work).status, "invalid_input")

    def test_failed_and_malformed_reads(self):
        source = RasterSourceSpec("unused", 2, 2, 1)
        for array in (None, np.zeros((1, 2)), np.zeros((2, 2), dtype=complex)):
            band = Mock()
            band.ReadAsArray.return_value = array
            dataset = Mock()
            dataset.GetRasterBand.return_value = band
            with self.assertRaises(InvalidRasterError):
                read_rgb(dataset, source, (0, 2, 0, 2), lambda: False)
        for mask in (None, np.zeros((1, 2))):
            band.ReadAsArray.return_value = np.zeros((2, 2), dtype=np.uint8)
            band.GetMaskFlags.return_value = gdal.GMF_PER_DATASET
            band.GetMaskBand.return_value.ReadAsArray.return_value = mask
            with self.assertRaises(InvalidRasterError):
                read_rgb(dataset, source, (0, 2, 0, 2), lambda: False)
        work = WorkerRequest(1, source, (0, 2, 0, 2), (0, 0), (1, 1), None, False)
        self.assertEqual(execute_request(work).status, "invalid_input")

    def test_cancellation_during_final_byte_band_read(self):
        source = RasterSourceSpec("unused", 2, 2, 1)
        dataset = Mock()
        band = dataset.GetRasterBand.return_value
        band.GetMaskFlags.return_value = gdal.GMF_ALL_VALID
        band.ReadAsArray.return_value = np.zeros((2, 2), dtype=np.uint8)
        # Even without mask reads or a finite-value scan, cancellation during
        # the final native read must not return a completed RGB window.
        with self.assertRaises(TraceCancelled):
            read_rgb(
                dataset,
                source,
                (0, 2, 0, 2),
                lambda: band.ReadAsArray.call_count == 3,
            )

    def test_cache_reuses_allocation_color_source_and_exact_search_view(self):
        values = np.ones((6, 6)) * 10
        values[0] = 0
        source = self.source(values)
        work = WorkerRequest(1, source, (0, 6, 0, 6), (1, 1), (1, 4), (0, 0, 0), False)
        snapshot, cost, valid, _ = prepare_snapshot(work, None, lambda: False)
        with patch(
            "raster_scribe.pointtool_raster.read_rgb",
            side_effect=AssertionError("unexpected read"),
        ):
            reused, same, _, timings = prepare_snapshot(work, snapshot, lambda: False)
            self.assertIs(reused, snapshot)
            self.assertTrue(np.shares_memory(same, cost))
            self.assertEqual(timings["copy_bytes"], 0)
            narrow = replace(work, bounds=(1, 5, 1, 5))
            warm = execute_request(narrow, snapshot)
            self.assertEqual(warm.origin, (1, 1))
            self.assertTrue(np.shares_memory(warm.snapshot.cost, snapshot.cost))
        cold = execute_request(narrow)
        self.assertEqual(warm.cost, cold.cost)
        self.assertEqual(warm.path, cold.path)
        self.assertEqual(warm.cost, 900)
        changed, other_cost, _, _ = prepare_snapshot(
            replace(work, color=(1, 1, 1)), snapshot, lambda: False
        )
        self.assertFalse(np.shares_memory(other_cost, cost))
        self.assertTrue(np.shares_memory(changed.bands[0], snapshot.bands[0]))
        self.assertEqual(cost[1, 1], 300)
        self.assertFalse(cost.flags.writeable)
        # The auto-color cache also retains just the most recent resolved RGB.
        automatic, _, _, _ = prepare_snapshot(
            replace(work, color=None), changed, lambda: False
        )
        self.assertEqual(automatic.cost_key[1], (10, 10, 10))
        with patch("raster_scribe.pointtool_raster.read_rgb", wraps=read_rgb) as reader:
            prepare_snapshot(
                replace(work, source=replace(source, revision=2)),
                snapshot,
                lambda: False,
            )
            self.assertEqual(reader.call_count, 1)

    def test_resource_limits_before_reads_and_scratch_accounting(self):
        with self.assertRaises(ResourceLimitError):
            validate_budget((0, 4096, 0, 4096))
        with self.assertRaises(ResourceLimitError):
            validate_budget((0, 100, 0, 100), live_bytes=MAX_ARRAY_BYTES)
        source = RasterSourceSpec("/not/opened", 10000, 10000, 1)
        work = WorkerRequest(
            1, source, (0, 4000, 0, 4000), (0, 0), (3999, 3999), None, False
        )
        with patch(
            "raster_scribe.pointtool_raster.gdal.OpenEx",
            side_effect=AssertionError("must reject first"),
        ):
            self.assertEqual(execute_request(work).status, "resource_limit")


class EndpointTest(TraceFixture):
    def feature(self, points, layer=None):
        layer = layer or self.vector
        feature = QgsFeature(layer.fields())
        feature.setGeometry(
            QgsGeometry.fromMultiPolylineXY([[QgsPointXY(*p) for p in points]])
        )
        self.assertTrue(layer.addFeature(feature))
        return feature.id()

    def test_nearest_later_feature_inclusive_threshold_and_transform_failure(self):
        self.feature([(1003, 1991), (1003, 1990)])
        self.feature([(1001, 1991), (1001, 1990)])
        self.assertEqual(self.tool.snap_to_itself(1000, 1991, 3), (1001, 1991))
        self.assertEqual(self.tool.snap_to_itself(1000, 1991, 1), (1001, 1991))
        with patch(
            "raster_scribe.pointtool.QgsCoordinateTransform",
            side_effect=ValueError("bad transform"),
        ):
            self.assertEqual(self.tool.snap_to_itself(1000, 1991, 3), (1000, 1991))

    def test_geographic_query_and_all_vertices_in_working_crs(self):
        layer = QgsVectorLayer("MultiLineString?crs=EPSG:4326", "geographic", "memory")
        self.project.addMapLayer(layer)
        layer.startEditing()
        self.iface.setActiveLayer(layer)
        transform = QgsCoordinateTransform(
            layer.crs(), QgsCoordinateReferenceSystem("EPSG:3857"), self.project
        )
        center = transform.transform(10, 60)
        # At latitude 60 a northward degree is about twice an eastward degree.
        # The closest vertex in degrees is therefore the farther in map units.
        self.feature([(10, 60.0006), (10.001, 60)], layer)
        x, y = self.tool.snap_to_itself(center.x(), center.y(), 150)
        expected = transform.transform(10.001, 60)
        self.assertAlmostEqual(x, expected.x(), places=6)
        self.assertAlmostEqual(y, expected.y(), places=6)
        self.canvas.setDestinationCrs(layer.crs())
        self.assertEqual(self.tool.snap_to_itself(10, 60, 0.0007), (10, 60.0006))

    def test_color_neighborhood_edges_positive_radius_and_both_snaps(self):
        self.tool.trace_color_changed(QColor("black"))
        self.tool.snap_tolerance_changed(3)
        self.assertEqual(self.tool.snap(5, 8), (8, 8))  # Positive radius included.
        for i, j in ((0, 0), (0, 15), (15, 0), (15, 15)):
            row, col = self.tool.snap(i, j)
            self.assertTrue(0 <= row < 16 and 0 <= col < 16)
        self.feature([(1008.7, 1991.3), (1008.7, 1990.3)])
        self.tool.snap2_tolerance_changed(1)
        endpoint = self.tool.resolve_endpoint((1008.1, 1992.9))
        self.assertEqual(endpoint.xy, (1008.7, 1991.3))
        self.assertEqual(endpoint.pixel, (8, 8))
        self.tool.accept_click((1008.1, 1992.9))
        self.assertEqual(self.tool.anchors[0], endpoint)
        with self.assertRaises(ValueError):
            self.tool.snap_tolerance_changed(100)

    def test_t_small_reads_and_reject_unsupported_mapping(self):
        with patch("raster_scribe.pointtool_raster.read_rgb", wraps=read_rgb) as reader:
            self.tool.raster_context.sample_color_at_indices(8, 8)
            self.tool.raster_context.sample_color_at_indices(8, 8)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(reader.call_args.args[2], (8, 9, 8, 9))
        for name, mapping in (
            ("rotated", (0, 1, 0.1, 10, 0, -1)),
            ("skewed", (0, 1, 0, 10, 0.1, -1)),
            ("reversed_x", (0, -1, 0, 10, 0, -1)),
            ("south_up", (0, 1, 0, 10, 0, 1)),
        ):
            with self.subTest(mapping=name):
                path = Path(self.directory.name) / f"{name}.tif"
                dataset = gdal.GetDriverByName("GTiff").Create(str(path), 2, 2, 3)
                dataset.SetGeoTransform(mapping)
                dataset = None
                layer = QgsRasterLayer(str(path), name)
                with self.assertRaises(InvalidRasterError):
                    RasterSampler(layer, self.project)

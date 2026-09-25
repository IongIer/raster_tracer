"""Compact RGB storage preserves masks, color arithmetic and trace results."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from osgeo import gdal

from .. import pointtool_raster as raster
from ..pointtool_session import RasterSourceSpec, WorkerRequest
from ..pointtool_tasks import execute_request
from .utilities import get_qgis_app


class CompactCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qgis_app()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def source(self, name, bands, data_type=gdal.GDT_Byte, nodata=None, mask=None):
        height, width = bands[0].shape
        path = Path(self.directory.name) / f"{name}.tif"
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(path), width, height, len(bands), data_type
        )
        dataset.SetGeoTransform((0, 1, 0, height, 0, -1))
        for number, values in enumerate(bands, 1):
            band = dataset.GetRasterBand(number)
            band.WriteArray(values)
            if nodata is not None:
                band.SetNoDataValue(nodata)
        if mask is not None:
            dataset.CreateMaskBand(gdal.GMF_PER_DATASET)
            dataset.GetRasterBand(1).GetMaskBand().WriteArray(mask)
        dataset = None
        return RasterSourceSpec(str(path), height, width, 1)

    def read(self, source):
        dataset = gdal.Open(source.path)
        return raster.read_rgb(
            dataset, source, (0, source.height, 0, source.width), lambda: False
        )

    def test_byte_nodata_and_explicit_masks_remain_impassable(self):
        values = np.arange(12, dtype=np.uint8).reshape(3, 4)
        channels = (values.copy(), values.copy(), values.copy())
        channels[1][1, 2] = 255
        mask = np.full(values.shape, 255, dtype=np.uint8)
        mask[2, 1] = 0
        for name, options, invalid in (
            ("nodata", {"nodata": 255}, (1, 2)),
            ("mask", {"mask": mask}, (2, 1)),
        ):
            with self.subTest(source=name):
                bands, valid = self.read(self.source(name, channels, **options))
                expected = np.ones(values.shape, dtype=bool)
                expected[invalid] = False
                np.testing.assert_array_equal(valid, expected)
                for actual, original in zip(bands, channels):
                    self.assertEqual(actual.dtype, np.dtype("uint8"))
                    np.testing.assert_array_equal(actual, original)
                    self.assertFalse(actual.flags.writeable)
                self.assertFalse(valid.flags.writeable)
                self.assertEqual(raster.color_cost(bands, valid, (0, 0, 0))[invalid], 0)

    def test_byte_color_costs_do_not_wrap_or_round_targets(self):
        bands = (
            np.array([[0, 255, 1], [254, 16, 128]], dtype=np.uint8),
            np.array([[255, 0, 254], [1, 128, 16]], dtype=np.uint8),
            np.array([[1, 128, 255], [0, 254, 16]], dtype=np.uint8),
        )
        valid = np.ones((2, 3), dtype=bool)
        valid[1, 2] = False
        for color in (
            (255, 0, 128),
            (-40, 512, 1.75),
            (0.5, 127.25, 255.75),
            (np.uint8(255), np.float32(0.5), np.int64(300)),
        ):
            with self.subTest(color=color):
                # Python float arithmetic is independent of NumPy's uint8
                # promotion rules, which differ between supported QGIS builds.
                expected = np.array(
                    [
                        [
                            int(
                                sum(
                                    (float(band[row, col]) - float(target)) ** 2
                                    for band, target in zip(bands, color)
                                )
                            )
                            if valid[row, col]
                            else 0
                            for col in range(3)
                        ]
                        for row in range(2)
                    ],
                    dtype=np.int64,
                )
                actual = raster.color_cost(bands, valid, color)
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(actual.flags.writeable)

    def test_cropped_snapshots_count_their_compact_backing_allocations(self):
        values = np.arange(40 * 48, dtype=np.uint16).reshape(40, 48).astype(np.uint8)
        source = self.source("backing", (values, values, values))
        work = WorkerRequest(
            1, source, (9, 21, 11, 27), (10, 12), (19, 25), (10, 20, 30), False
        )
        with patch.object(raster, "CACHE_ALIGNMENT", 16):
            snapshot, _, _, _ = raster.prepare_snapshot(work, None, lambda: False)
        self.assertEqual(snapshot.bounds, (0, 32, 0, 32))
        # Three byte channels, a byte validity mask and an int64 cost per pixel.
        self.assertEqual(snapshot.nbytes, 32 * 32 * 12)
        view = np.s_[9:21, 11:27]
        cropped = replace(
            snapshot,
            bounds=work.bounds,
            bands=tuple(band[view] for band in snapshot.bands),
            valid=snapshot.valid[view],
            cost=snapshot.cost[view],
        )
        self.assertEqual(cropped.nbytes, snapshot.nbytes)
        for array in (*cropped.bands, cropped.valid, cropped.cost):
            self.assertFalse(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
        changed, graph, _, _ = raster.prepare_snapshot(
            replace(work, color=(255.5, -1, 300)), cropped, lambda: False
        )
        self.assertEqual(changed.nbytes, 32 * 32 * 4 + 12 * 16 * 8)
        reference = raster.color_cost(
            tuple(band.astype(np.float64) for band in cropped.bands),
            cropped.valid,
            (255.5, -1, 300),
        )
        np.testing.assert_array_equal(graph, reference)

    def test_fixed_automatic_and_resampled_colors_match_float_traces(self):
        rng = np.random.default_rng(1975)
        values = tuple(rng.integers(0, 256, (32, 40), dtype=np.uint8) for _ in range(3))
        byte_source = self.source("byte", values)
        float_source = self.source("float", values, gdal.GDT_Float64)
        work = WorkerRequest(
            1, byte_source, (5, 22, 7, 29), (6, 8), (20, 27), (110, 60, 30), False
        )
        sequence = (
            work,
            work,  # Fixed-color hit.
            replace(work, color=(180.25, 80.5, 35.75)),  # T resampling.
            replace(work, color=None),
            replace(work, color=None),  # Automatic-color hit.
            replace(work, color=None, goal=(19, 26)),  # New automatic color.
        )
        cached = None
        with patch.object(raster, "CACHE_ALIGNMENT", 16):
            for index, request in enumerate(sequence):
                with self.subTest(request=index):
                    actual = execute_request(request, cached)
                    reference = execute_request(replace(request, source=float_source))
                    self.assertEqual(actual.status, "success", actual.diagnostics)
                    self.assertEqual(
                        (actual.status, actual.cost, actual.path, actual.origin),
                        (
                            reference.status,
                            reference.cost,
                            reference.path,
                            reference.origin,
                        ),
                    )
                    np.testing.assert_array_equal(
                        actual.snapshot.cost, reference.snapshot.cost
                    )
                    if cached is not None:
                        self.assertTrue(actual.timings["rgb_cache_hit"])
                        for band, previous in zip(actual.snapshot.bands, cached.bands):
                            self.assertIs(band, previous)
                    self.assertEqual(actual.timings["cost_cache_hit"], index in (1, 4))
                    cached = actual.snapshot

    def test_nonbyte_and_mixed_bands_preserve_values(self):
        byte = np.array([[0, 255], [1, 128]], dtype=np.uint8)
        integer = np.array([[0, 65535], [256, 32768]], dtype=np.uint16)
        fractional = np.array([[-0.5, 255.25], [1.125, 500.75]], dtype=np.float32)
        for name, values, kind in (
            ("uint16", integer, gdal.GDT_UInt16),
            ("float32", fractional, gdal.GDT_Float32),
        ):
            with self.subTest(source=name):
                bands, valid = self.read(self.source(name, (values,) * 3, kind))
                self.assertTrue(valid.all())
                for band in bands:
                    self.assertEqual(band.dtype, np.dtype("float64"))
                    np.testing.assert_array_equal(band, values)
                    self.assertFalse(band.flags.writeable)

        sources = [
            self.source(name, (values,), kind).path
            for name, values, kind in (
                ("mixed-byte", byte, gdal.GDT_Byte),
                ("mixed-uint16", integer, gdal.GDT_UInt16),
                ("mixed-float32", fractional, gdal.GDT_Float32),
            )
        ]
        vrt_path = Path(self.directory.name) / "mixed.vrt"
        vrt = gdal.BuildVRT(str(vrt_path), sources, separate=True)
        self.assertIsNotNone(vrt)
        vrt = None
        bands, valid = self.read(RasterSourceSpec(str(vrt_path), 2, 2, 1))
        self.assertEqual(
            [band.dtype for band in bands], [np.uint8, np.float64, np.float64]
        )
        expected = (byte.astype(np.float64), integer.astype(np.float64), fractional)
        for actual, original in zip(bands, expected):
            np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(
            raster.color_cost(bands, valid, (255.5, 300, -2.25)),
            raster.color_cost(expected, valid, (255.5, 300, -2.25)),
        )

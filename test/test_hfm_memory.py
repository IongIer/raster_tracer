"""HFM budgeting must include snapshots retained by the task or its caller."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from osgeo import gdal

from .. import pointtool_raster as raster
from .. import pointtool_tasks as tasks
from ..pointtool_session import RasterSourceSpec, WorkerRequest
from .utilities import get_qgis_app, write_raster


class HfmRetainedMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qgis_app()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "colors.tif"
        values = np.tile(np.arange(40, dtype=np.uint8), (32, 1))
        write_raster(path, (values, values, values), gdal.GDT_Byte)
        source = RasterSourceSpec(str(path), 32, 40, 1)
        self.work = WorkerRequest(
            1,
            source,
            (0, 32, 0, 40),
            (16, 4),
            (16, 30),
            None,
            False,
            enhanced_tracing=True,
        )
        self.snapshot, _, _, _ = raster.prepare_snapshot(self.work, None, lambda: False)

    def execute_recording_budget(self, work, snapshot):
        observed = []

        def record(graph, valid, start, goal, coarse, cancel, *, live_bytes):
            observed.append(live_bytes)
            return coarse, {}

        with patch.object(tasks, "refine_path", side_effect=record):
            result = tasks.execute_request(work, snapshot)
        self.assertEqual(result.status, "success", result.diagnostics)
        self.assertEqual(len(observed), 1)
        return result.snapshot, observed[0]

    def test_automatic_color_miss_retains_both_costs_and_counts_rgb_once(self):
        previous = self.snapshot
        current, budget = self.execute_recording_budget(
            replace(self.work, goal=(16, 31)), previous
        )
        self.assertIsNot(current.cost, previous.cost)
        self.assertIs(current.bands[0], previous.bands[0])
        self.assertIs(current.valid, previous.valid)
        self.assertEqual(budget, current.nbytes + previous.cost.nbytes)
        self.assertGreater(budget, current.nbytes)

    def test_automatic_color_hit_counts_shared_snapshot_only_once(self):
        current, budget = self.execute_recording_budget(self.work, self.snapshot)
        self.assertIs(current.cost, self.snapshot.cost)
        self.assertEqual(budget, current.nbytes)

    def test_noncovering_caller_snapshot_retains_its_full_backing_arrays(self):
        previous = replace(
            self.snapshot,
            bounds=(0, 8, 0, 8),
            bands=tuple(band[:8, :8] for band in self.snapshot.bands),
            valid=self.snapshot.valid[:8, :8],
            cost=self.snapshot.cost[:8, :8],
        )
        self.assertEqual(previous.nbytes, self.snapshot.nbytes)
        current, budget = self.execute_recording_budget(self.work, previous)
        self.assertIsNot(current.bands[0], previous.bands[0])
        self.assertEqual(budget, current.nbytes + previous.nbytes)

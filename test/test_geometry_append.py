"""Native geometry updates preserve coordinates, copy isolation and QGIS undo."""

import math
import random
import unittest

from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsLineString,
    QgsMultiLineString,
    QgsPointXY,
)

from ..pointtool import TracingModes, append_path_points
from .tracing_fixture import TraceFixture


def original_append(existing, points):
    """Oracle: the previous complete-polyline conversion."""
    lines = existing.asMultiPolyline()
    if not lines or not lines[-1]:
        raise ValueError("Target line is empty")
    lines[-1].extend(points[1:] if lines[-1][-1] == points[0] else points)
    return QgsGeometry.fromMultiPolylineXY(lines)


class GeometryAppendTest(unittest.TestCase):
    def assert_append_matches(self, existing, points):
        original = bytes(existing.asWkb())
        bounds = existing.boundingBox().asWktPolygon()
        retained = QgsGeometry(existing)
        feature = QgsFeature()
        feature.setGeometry(existing)
        expected = original_append(existing, points)
        actual = append_path_points(existing, points)
        self.assertEqual(bytes(actual.asWkb()), bytes(expected.asWkb()))
        self.assertEqual(actual.boundingBox(), expected.boundingBox())
        self.assertEqual(existing.boundingBox().asWktPolygon(), bounds)
        for previous in (existing, retained, feature.geometry()):
            self.assertEqual(bytes(previous.asWkb()), original)

    def test_shared_disconnected_approximate_and_duplicate_endpoints(self):
        existing = QgsGeometry.fromWkt("MULTILINESTRING((0 0,1 1))")
        for first in ((1, 1), (9, 10), (math.nextafter(1.0, 2.0), 1)):
            with self.subTest(first=first):
                points = [
                    QgsPointXY(*first),
                    QgsPointXY(20, 30),
                    QgsPointXY(20, 30),
                    QgsPointXY(30, 40),
                ]
                self.assert_append_matches(existing, points)
        self.assert_append_matches(existing, [QgsPointXY(1, 1)])

    def test_multipart_and_closed_lines_preserve_order_and_cached_bounds(self):
        for wkt in (
            "MULTILINESTRING((0 0,1 1),(5 6,7 8))",
            "MULTILINESTRING((0 0,0 0))",
            "MULTILINESTRING((0 0,1 1,0 0))",
        ):
            with self.subTest(wkt=wkt):
                self.assert_append_matches(
                    QgsGeometry.fromWkt(wkt),
                    [QgsPointXY(7, 8), QgsPointXY(-20, -30), QgsPointXY(30, 40)],
                )

    def test_curves_and_dimensions_keep_existing_conversion(self):
        for wkt in (
            "MULTILINESTRING Z((0 0 1,1 1 2))",
            "MULTILINESTRING M((0 0 1,1 1 2))",
            "MULTILINESTRING ZM((0 0 1 2,1 1 3 4))",
            "MULTICURVE(CIRCULARSTRING(0 0,1 1,2 0))",
            "MULTICURVE(COMPOUNDCURVE((0 0,1 0),CIRCULARSTRING(1 0,2 1,3 0)))",
            "MULTICURVE((0 0,1 1),CIRCULARSTRING(1 1,2 2,3 1))",
        ):
            with self.subTest(wkt=wkt):
                self.assert_append_matches(
                    QgsGeometry.fromWkt(wkt), [QgsPointXY(1, 1), QgsPointXY(10, 20)]
                )

    def test_random_multipart_output_matches_previous_conversion(self):
        rng = random.Random(941)
        for case in range(30):
            parts = [
                [
                    QgsPointXY(rng.random(), rng.random())
                    for _ in range(rng.randrange(2, 80))
                ]
                for _ in range(rng.randrange(1, 5))
            ]
            existing = QgsGeometry.fromMultiPolylineXY(parts)
            seam = parts[-1][-1] if case % 2 else QgsPointXY(rng.random(), rng.random())
            points = [seam] + [
                QgsPointXY(rng.random(), rng.random())
                for _ in range(rng.randrange(1, 30))
            ]
            with self.subTest(case=case):
                self.assert_append_matches(existing, points)

    def test_empty_and_empty_final_part_still_rejected(self):
        multi = QgsMultiLineString()
        multi.addGeometry(QgsLineString([QgsPointXY(0, 0), QgsPointXY(1, 1)]))
        multi.addGeometry(QgsLineString())
        points = [QgsPointXY(2, 2), QgsPointXY(3, 3)]
        for geometry in (
            QgsGeometry.fromWkt("MULTILINESTRING EMPTY"),
            QgsGeometry(multi),
        ):
            with self.subTest(wkt=geometry.asWkt()):
                original = bytes(geometry.asWkb())
                with self.assertRaisesRegex(ValueError, "Target line is empty"):
                    append_path_points(geometry, points)
                self.assertEqual(bytes(geometry.asWkb()), original)


class GeometryAppendIntegrationTest(TraceFixture):
    def test_committed_prefix_copy_b_undo_and_qgis_undo_redo(self):
        self.tool.tracing_mode = TracingModes.LINE
        self.tool.accept_click((999, 1990))
        self.tool.accept_click((1010, 1990))
        first = QgsGeometry(self.tool.session.geometry)
        first_wkb = bytes(first.asWkb())
        first.boundingBox()
        self.tool.accept_click((1021, 1990))
        second = QgsGeometry(self.tool.session.geometry)
        second_wkb = bytes(second.asWkb())
        self.assertEqual(bytes(first.asWkb()), first_wkb)
        self.assertEqual(
            [(p.x(), p.y()) for p in second.vertices()],
            [(999, 1990), (1010, 1990), (1021, 1990)],
        )
        self.tool.remove_last_anchor_point()
        self.assertEqual(bytes(self.tool.session.geometry.asWkb()), first_wkb)
        self.assertEqual(bytes(second.asWkb()), second_wkb)
        self.tool.accept_click((1021, 1990))
        self.assertEqual(bytes(self.tool.session.geometry.asWkb()), second_wkb)
        self.assertEqual(self.vector.featureCount(), 0)
        self.tool.finish_session()
        fid = next(self.vector.getFeatures()).id()
        self.vector.undoStack().undo()
        self.assertEqual(self.vector.featureCount(), 0)
        self.vector.undoStack().redo()
        self.assertEqual(
            bytes(self.vector.getFeature(fid).geometry().asWkb()), second_wkb
        )
        self.assertEqual(bytes(first.asWkb()), first_wkb)
        self.assertEqual(bytes(second.asWkb()), second_wkb)
        self.assertTrue(self.vector.rollBack())
        self.assertEqual(self.vector.featureCount(), 0)

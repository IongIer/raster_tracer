"""QGIS vertex snapping semantics and edits/shortcuts around previews."""

import math
import random
from unittest.mock import patch

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from ..pointtool import TracingModes
from .tracing_fixture import TraceFixture


def line_geometry(points):
    return QgsGeometry.fromMultiPolylineXY([[QgsPointXY(*p) for p in points]])


class SnappingOptimizationTest(TraceFixture):
    def add_feature(self, points, layer=None):
        layer = layer or self.vector
        feature = QgsFeature(layer.fields())
        feature.setGeometry(line_geometry(points))
        self.assertTrue(layer.addFeature(feature))
        return feature.id()

    def reference_snap(self, x, y, tolerance):
        layer = self.tool.get_current_vector_layer()
        working = self.canvas.mapSettings().destinationCrs()
        to_layer = QgsCoordinateTransform(working, layer.crs(), self.project)
        to_map = QgsCoordinateTransform(layer.crs(), working, self.project)
        rectangle = to_layer.transformBoundingBox(
            QgsRectangle(x - tolerance, y - tolerance, x + tolerance, y + tolerance)
        )
        request = QgsFeatureRequest().setFilterRect(rectangle).setSubsetOfAttributes([])
        best = None
        for feature in layer.getFeatures(request):
            for index, vertex in enumerate(feature.geometry().vertices()):
                point = to_map.transform(QgsPointXY(vertex))
                sx, sy = point.x(), point.y()
                dx, dy = sx - x, sy - y
                distance = dx * dx + dy * dy
                key = distance, feature.id(), -index
                if distance <= tolerance**2 and (best is None or key < best[0]):
                    best = key, (sx, sy)
        return best[1] if best else (x, y)

    def test_native_last_vertex_ties_in_lines_multipart_curves_and_closed_lines(self):
        cases = (
            ("MULTILINESTRING((-1 0,1 0))", (1, 0), 1),
            ("MULTILINESTRING((-1 0,1 0),(0 -1,0 1))", (0, 1), 3),
            ("MULTICURVE(CIRCULARSTRING(1 0,0 1,-1 0))", (-1, 0), 2),
            ("MULTILINESTRING((1 0,0 1,-1 0,0 -1,1 0))", (1, 0), 4),
        )
        for wkt, expected, expected_index in cases:
            with self.subTest(wkt=wkt):
                kind = (
                    "MultiCurve" if wkt.startswith("MULTICURVE") else "MultiLineString"
                )
                layer = QgsVectorLayer(f"{kind}?crs=EPSG:3857", "ties", "memory")
                self.project.addMapLayer(layer)
                self.assertTrue(layer.startEditing())
                self.iface.setActiveLayer(layer)
                geometry = QgsGeometry.fromWkt(wkt)
                feature = QgsFeature(layer.fields())
                feature.setGeometry(geometry)
                self.assertTrue(layer.addFeature(feature))
                self.assertEqual(
                    geometry.closestVertex(QgsPointXY(0, 0))[1], expected_index
                )
                self.assertEqual(self.tool.snap_to_itself(0, 0, 1), expected)

    def test_native_random_multipart_matches_scalar_map_distance(self):
        rng = random.Random(1749)
        geometry = QgsGeometry.fromMultiPolylineXY(
            [
                [
                    QgsPointXY(rng.uniform(-2, 2), rng.uniform(-2, 2))
                    for _ in range(size)
                ]
                for size in (91, 73)
            ]
        )
        feature = QgsFeature(self.vector.fields())
        feature.setGeometry(geometry)
        self.assertTrue(self.vector.addFeature(feature))
        for _ in range(60):
            args = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 2)
            with self.subTest(args=args):
                self.assertEqual(
                    self.tool.snap_to_itself(*args), self.reference_snap(*args)
                )

    def test_inclusive_tolerance_invalid_inputs_and_empty_candidates(self):
        query = (1000, 1991)
        self.assertEqual(self.tool.snap_to_itself(*query, 1), query)
        empty = QgsFeature(self.vector.fields())
        self.assertTrue(self.vector.addFeature(empty))
        self.assertEqual(self.tool.snap_to_itself(*query, 1), query)
        self.add_feature([(1001, 1991), (1002, 1991)])
        self.assertEqual(
            self.tool.snap_to_itself(*query, math.nextafter(1.0, 0.0)), query
        )
        self.assertEqual(self.tool.snap_to_itself(*query, 1), (1001, 1991))
        self.assertEqual(
            self.tool.snap_to_itself(*query, math.nextafter(1.0, 2.0)), (1001, 1991)
        )
        self.assertEqual(self.tool.snap_to_itself(1001, 1991, 0), (1001, 1991))
        for tolerance in (None, -1, math.inf, math.nan):
            with self.subTest(tolerance=tolerance):
                self.assertEqual(self.tool.snap_to_itself(*query, tolerance), query)
        # A segment crosses the query window, but its vertices are outside it.
        self.add_feature([(999, 1991), (1001, 1991)])
        self.assertEqual(self.tool.snap_to_itself(*query, 0.5), query)
        self.iface.setActiveLayer(self.raster)
        self.assertEqual(self.tool.snap_to_itself(*query, 1), query)

    def test_transformed_crs_compares_all_vertices_in_map_units(self):
        layer = QgsVectorLayer("MultiLineString?crs=EPSG:4326", "geographic", "memory")
        self.project.addMapLayer(layer)
        self.assertTrue(layer.startEditing())
        self.iface.setActiveLayer(layer)
        fid = self.add_feature([(10, 60.0006), (10.001, 60)], layer)
        geometry = layer.getFeature(fid).geometry()
        original = bytes(geometry.asWkb())
        self.assertEqual(geometry.closestVertex(QgsPointXY(10, 60))[1], 0)
        transform = QgsCoordinateTransform(
            layer.crs(), self.project.crs(), self.project
        )
        center = transform.transform(10, 60)
        north = transform.transform(10, 60.0006)
        east = transform.transform(10.001, 60)
        # At latitude 60 the northward offset is smaller in degrees but larger
        # on the canvas. Selecting a layer-space nearest vertex would be wrong.
        self.assertGreater(center.sqrDist(north), center.sqrDist(east))
        self.assertEqual(
            self.tool.snap_to_itself(center.x(), center.y(), 150),
            (east.x(), east.y()),
        )
        self.assertEqual(bytes(layer.getFeature(fid).geometry().asWkb()), original)
        # Symmetric east/west offsets at longitude zero have equal map distances:
        # the transformed path also chooses the last vertex within the feature.
        self.assertTrue(
            layer.changeGeometry(fid, line_geometry([(-0.001, 60), (0.001, 60)]))
        )
        center = transform.transform(0, 60)
        east = transform.transform(0.001, 60)
        self.assertEqual(
            self.tool.snap_to_itself(center.x(), center.y(), 150),
            (east.x(), east.y()),
        )

    def test_transformed_draft_snapping_preserves_geometry_and_dimensions(self):
        self.vector.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        geometry = QgsGeometry.fromWkt(
            "MULTILINESTRING ZM ((10 60.0006 7 2,10.001 60 8 3),"
            "(10.002 60 9 4,10.003 60 10 5))"
        )
        original = bytes(geometry.asWkb())
        self.tool.session.target_id = self.vector.id()
        self.tool.session.geometry = geometry
        self.addCleanup(self.tool.session.reset)
        transform = QgsCoordinateTransform(
            self.vector.crs(), self.project.crs(), self.project
        )
        center = transform.transform(10, 60)
        east = transform.transform(10.001, 60)
        for _ in range(2):
            self.assertEqual(
                self.tool.snap_to_itself(center.x(), center.y(), 150),
                (east.x(), east.y()),
            )
            self.assertEqual(bytes(self.tool.session.geometry.asWkb()), original)
        self.assertEqual(self.vector.featureCount(), 0)

    def test_failed_geometry_transform_does_not_snap_in_layer_coordinates(self):
        self.vector.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        self.add_feature([(0.001, 0), (0.002, 0)])
        with patch.object(
            QgsGeometry,
            "transform",
            return_value=Qgis.GeometryOperationResult.InvalidBaseGeometry,
        ):
            self.assertEqual(self.tool.snap_to_itself(0, 0, 500), (0, 0))

    def test_feature_ties_geometry_edit_undo_delete_and_rollback(self):
        first = self.add_feature([(1001, 1991), (999, 1991)])
        fid = self.add_feature([(1000, 1992), (1000, 1990)])

        def check():
            self.assertEqual(
                self.tool.snap_to_itself(1000, 1991, 1),
                self.reference_snap(1000, 1991, 1),
            )

        check()
        expected = (999, 1991) if first < fid else (1000, 1990)
        self.assertEqual(self.tool.snap_to_itself(1000, 1991, 1), expected)
        self.vector.beginEditCommand("Move vertices")
        self.assertTrue(
            self.vector.changeGeometry(fid, line_geometry([(1000, 1991)] * 80))
        )
        self.vector.endEditCommand()
        check()
        self.assertEqual(self.tool.snap_to_itself(1000, 1991, 1), (1000, 1991))
        self.vector.undoStack().undo()
        check()
        self.vector.undoStack().redo()
        check()
        self.assertTrue(self.vector.deleteFeature(fid))
        check()
        self.vector.undoStack().undo()
        check()
        self.assertTrue(self.vector.rollBack())
        check()
        self.assertEqual(self.tool.snap_to_itself(1000, 1991, 1), (1000, 1991))

    def test_layer_crs_changes_and_curves_use_current_geometry(self):
        self.add_feature([(1001, 1991)])
        self.assertEqual(self.tool.snap_to_itself(1000, 1991, 1), (1001, 1991))
        layer = QgsVectorLayer("MultiCurve?crs=EPSG:4326", "curves", "memory")
        self.project.addMapLayer(layer)
        layer.startEditing()
        feature = QgsFeature(layer.fields())
        feature.setGeometry(
            QgsGeometry.fromWkt("MULTICURVE(CIRCULARSTRING(10 60,10.001 60,10.002 60))")
        )
        self.assertTrue(layer.addFeature(feature))
        self.iface.setActiveLayer(layer)
        transform = QgsCoordinateTransform(
            layer.crs(), self.project.crs(), self.project
        )
        center = transform.transform(10.0001, 60)
        args = center.x(), center.y(), 50
        self.assertEqual(self.tool.snap_to_itself(*args), self.reference_snap(*args))
        self.canvas.setDestinationCrs(layer.crs())
        self.assertEqual(self.tool.snap_to_itself(10, 60, 0), (10, 60))
        layer.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        self.assertEqual(self.tool.snap_to_itself(10, 60, 0), (10, 60))
        self.tool.suspend()
        self.tool.activate()
        self.assertEqual(
            self.tool.snap_to_itself(10, 60, 0.01), self.reference_snap(10, 60, 0.01)
        )

    def test_dense_shortcut_bypasses_search_and_generated_vertices_snap_after_undo(
        self,
    ):
        scheduler = self.plugin.task_controller
        with patch.object(scheduler, "submit", side_effect=AssertionError("no search")):
            self.key(Qt.Key.Key_D)
            self.assertEqual(self.tool.tracing_mode, TracingModes.DENSE_LINE)
            self.tool.accept_click((999, 1990))
            self.tool.accept_click((1321, 1990))
        vertex = self.tool.session.geometry.asMultiPolyline()[0][1]
        target = (vertex.x(), vertex.y())
        cursor = (vertex.x(), vertex.y() + 0.25)
        self.assertEqual(self.tool.snap_to_itself(*cursor, 0.5), target)
        self.tool.remove_last_anchor_point()
        self.assertEqual(self.tool.snap_to_itself(*cursor, 0.5), cursor)
        self.tool.accept_click((1321, 1990))
        self.assertEqual(self.tool.snap_to_itself(*cursor, 0.5), target)
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.tool.tracing_mode, TracingModes.PATH)

    def test_t_resamples_color_reuses_rgb_and_rejects_old_color_result(self):
        scheduler = self.plugin.task_controller
        submitted = self.control_tasks()
        self.tool.trace_color_changed(QColor("black"))
        self.accept(8, 2)
        self.tool._hover(self.tool.toCanvasCoordinates(self.tool.to_coords(8, 13)))
        self.tool.preview_controller.ensure_inflight_started()
        submitted[-1].finish()
        snapshot = scheduler._snapshot
        self.assertIsNotNone(snapshot)
        self.tool.last_mouse_event_pos = self.tool.toCanvasCoordinates(
            self.tool.to_coords(4, 8)
        )
        self.key(Qt.Key.Key_T)
        self.assertEqual(self.tool.trace_color_value, (255.0, 255.0, 255.0))
        self.tool.preview_controller.ensure_inflight_started()
        stale = submitted[-1]
        self.assertIsNotNone(stale.snapshot)
        for previous, current in zip(snapshot.bands, stale.snapshot.bands):
            self.assertIs(previous, current)
        self.assertIsNone(stale.snapshot.cost)
        self.tool.last_mouse_event_pos = self.tool.toCanvasCoordinates(
            self.tool.to_coords(8, 8)
        )
        self.key(Qt.Key.Key_T)
        self.assertEqual(self.tool.trace_color_value, (0.0, 0.0, 0.0))
        self.tool.preview_controller.ensure_inflight_started()
        stale.finish()
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(len(self.tool.anchors), 1)
        self.assertEqual(submitted[-1].work.color, (0.0, 0.0, 0.0))
        submitted[-1].finish()

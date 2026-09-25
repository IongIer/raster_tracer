"""Completed draft writes respect native defaults and provider transactions."""

from pathlib import Path
from unittest.mock import patch

from qgis.core import (
    Qgis,
    QgsDefaultValue,
    QgsFeature,
    QgsGeometry,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from ..pointtool import TracingModes
from .tracing_fixture import TraceFixture


class DraftProvidersTest(TraceFixture):
    def replace_vector(self, layer, transaction_mode=Qgis.TransactionMode.Disabled):
        self.assertTrue(self.tool.finish_session())
        self.assertTrue(self.vector.rollBack())
        self.project.removeMapLayer(self.vector)
        self.assertTrue(self.project.setTransactionMode(transaction_mode))
        self.vector = layer
        self.assertTrue(layer.isValid())
        self.project.addMapLayer(layer)
        self.canvas.setLayers([layer, self.raster])
        self.iface.setActiveLayer(layer)
        self.assertTrue(layer.startEditing())
        self.tool.tracing_mode = TracingModes.LINE

    def geopackage(self, name, transaction_mode):
        path = Path(self.directory.name) / f"{name}.gpkg"
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = "lines"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            self.vector, str(path), self.project.transformContext(), options
        )
        self.assertEqual(result[0], QgsVectorFileWriter.NoError, result)
        self.replace_vector(
            QgsVectorLayer(f"{path}|layername=lines", name, "ogr"), transaction_mode
        )

    def draft(self, points=((999, 1990), (1010, 1990), (1021, 1990))):
        for point in points:
            self.tool.accept_click(point)
        self.assertEqual(len(self.tool.anchors), len(points))
        self.assertIsNotNone(self.tool.session.geometry)
        self.assertFalse(self.vector.isEditCommandActive())
        return bytes(self.tool.session.geometry.asWkb())

    def test_native_feature_defaults_use_completed_geometry(self):
        self.replace_vector(
            QgsVectorLayer(
                "MultiLineString?crs=EPSG:3857&field=length:double"
                "&field=label:string&field=ident:integer",
                "Default fields",
                "memory",
            )
        )
        self.vector.setDefaultValueDefinition(
            0, QgsDefaultValue("length($geometry)", True)
        )
        self.vector.setDefaultValueDefinition(1, QgsDefaultValue("'contour'"))
        self.vector.setDefaultValueDefinition(
            2, QgsDefaultValue('coalesce(maximum("ident"), 0) + 1')
        )
        for number in (1, 2):
            expected = self.draft()
            self.assertEqual(self.vector.featureCount(), number - 1)
            self.assertTrue(self.tool.finish_session())
            feature = next(
                feature
                for feature in self.vector.getFeatures()
                if feature["ident"] == number
            )
            self.assertEqual(bytes(feature.geometry().asWkb()), expected)
            self.assertEqual(feature["length"], 22)
            self.assertEqual(feature["label"], "contour")
        stack = self.vector.undoStack()
        self.assertEqual(stack.count(), 2)
        self.assertEqual([stack.command(i).childCount() for i in range(2)], [1, 1])
        stack.undo()
        self.assertEqual(self.vector.featureCount(), 1)
        stack.redo()
        self.assertEqual(self.vector.featureCount(), 2)

    def test_geopackage_transaction_keeps_external_edit_and_draft_independent(self):
        self.geopackage("automatic", Qgis.TransactionMode.AutomaticGroups)
        self.assertIsNotNone(self.vector.dataProvider().transaction())
        other = QgsFeature(self.vector.fields())
        other.setGeometry(QgsGeometry.fromWkt("MULTILINESTRING((1 1,2 2))"))
        self.vector.beginEditCommand("Other feature")
        self.assertTrue(self.vector.addFeature(other))
        self.vector.endEditCommand()
        original = bytes(other.geometry().asWkb())
        expected = self.draft()
        anchors = self.tool.anchors
        changed = QgsGeometry.fromWkt("MULTILINESTRING((3 3,4 4))")
        # A bare external write exposes transaction savepoint mistakes which
        # a simple memory-provider undo test cannot detect.
        self.assertTrue(self.vector.changeGeometry(other.id(), changed))
        stack = self.vector.undoStack()
        self.assertEqual(self.tool.anchors, anchors)
        stack.undo()
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertEqual(
            bytes(self.vector.getFeature(other.id()).geometry().asWkb()), original
        )
        self.assertEqual(bytes(self.tool.session.geometry.asWkb()), expected)
        stack.redo()
        self.assertTrue(self.tool.finish_session())
        traced = next(
            feature
            for feature in self.vector.getFeatures()
            if feature.id() != other.id()
        )
        self.assertEqual(bytes(traced.geometry().asWkb()), expected)
        self.assertEqual(stack.count(), 3)
        stack.undo()
        # OGR's cached featureCount can lag a transaction savepoint rollback.
        # Inspect actual features on both sides of the edit buffer instead.
        self.assertEqual(len(list(self.vector.getFeatures())), 1)
        self.assertEqual(len(list(self.vector.dataProvider().getFeatures())), 1)
        self.assertEqual(
            bytes(self.vector.getFeature(other.id()).geometry().asWkb()),
            bytes(changed.asWkb()),
        )
        stack.undo()
        self.assertEqual(
            bytes(self.vector.getFeature(other.id()).geometry().asWkb()), original
        )
        stack.redo()
        stack.redo()
        self.assertEqual(len(list(self.vector.getFeatures())), 2)
        provider = list(self.vector.dataProvider().getFeatures())
        self.assertEqual(len(provider), 2)
        self.assertIn(
            expected, [bytes(feature.geometry().asWkb()) for feature in provider]
        )

    def test_buffered_group_save_includes_draft_only_layer(self):
        self.geopackage("buffered", Qgis.TransactionMode.BufferedGroups)
        self.assertIsNone(self.vector.dataProvider().transaction())
        for stop_editing in (False, True):
            with self.subTest(stop_editing=stop_editing):
                before = self.vector.featureCount()
                expected = self.draft()
                self.assertEqual(self.vector.featureCount(), before)
                self.assertTrue(self.vector.commitChanges(stop_editing))
                self.assertEqual(self.tool.anchors, ())
                self.assertEqual(self.vector.featureCount(), before + 1)
                self.assertEqual(self.vector.isEditable(), not stop_editing)
                self.assertEqual(self.vector.undoStack().count(), 0)
                provider = list(self.vector.dataProvider().getFeatures())
                self.assertEqual(len(provider), before + 1)
                self.assertIn(
                    expected,
                    [bytes(feature.geometry().asWkb()) for feature in provider],
                )

    def test_buffered_group_rollback_discards_unpublished_draft(self):
        self.geopackage("rollback", Qgis.TransactionMode.BufferedGroups)
        self.draft()
        self.assertTrue(self.vector.rollBack())
        self.assertEqual(self.tool.anchors, ())
        self.assertIsNone(self.tool.session.geometry)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(list(self.vector.dataProvider().getFeatures()), [])

    def test_failed_publication_during_save_retains_recoverable_draft(self):
        self.tool.tracing_mode = TracingModes.LINE
        expected = self.draft()
        anchors = self.tool.anchors
        with patch.object(self.vector, "addFeature", return_value=False):
            # A signal handler cannot veto QGIS's save-and-stop operation.
            # Retain accepted work even if saving the other edits succeeds.
            self.assertTrue(self.vector.commitChanges())
        self.assertFalse(self.vector.isEditable())
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.tool.anchors, anchors)
        self.assertEqual(bytes(self.tool.session.geometry.asWkb()), expected)
        self.assertTrue(self.vector.startEditing())
        self.assertTrue(self.tool.finish_session())
        self.assertEqual(self.tool.anchors, ())
        self.assertEqual(self.vector.featureCount(), 1)
        self.assertEqual(
            bytes(next(self.vector.getFeatures()).geometry().asWkb()), expected
        )

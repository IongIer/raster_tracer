"""Draft backtracking, one-action publication and independent QGIS undo."""

from unittest.mock import patch

from qgis import utils as qgis_utils
from qgis.core import QgsCoordinateReferenceSystem, QgsFeature, QgsGeometry
from qgis.PyQt.QtCore import Qt

from ..pointtool import TracingModes
from .tracing_fixture import TraceFixture


class LineUndoTest(TraceFixture):
    def setUp(self):
        super().setUp()
        self.tool.tracing_mode = TracingModes.LINE

    def trace(self, points=((999, 1990), (1010, 1990), (1021, 1990))):
        count = self.vector.featureCount()
        for point in points:
            self.tool.accept_click(point)
        self.assertEqual(len(self.tool.anchors), len(points))
        self.assertEqual(self.vector.featureCount(), count)
        return self.draft_geometry()

    def draft_geometry(self):
        return bytes(self.tool.session.geometry.asWkb())

    def geometry(self, fid):
        return bytes(self.vector.getFeature(fid).geometry().asWkb())

    def assert_finished_line_undo(self, expected):
        self.assertEqual(self.tool.anchors, ())
        self.assertIsNone(self.tool.session.geometry)
        self.assertFalse(self.vector.isEditCommandActive())
        self.assertEqual(self.vector.featureCount(), 1)
        feature = next(self.vector.getFeatures())
        fid = feature.id()
        self.assertEqual(self.geometry(fid), expected)
        stack = self.vector.undoStack()
        self.assertTrue(stack.canUndo())
        stack.undo()
        self.assertEqual(self.vector.featureCount(), 0)
        stack.redo()
        self.assertEqual(self.geometry(fid), expected)
        return fid

    def add_external_feature(self):
        feature = QgsFeature(self.vector.fields())
        feature.setGeometry(QgsGeometry.fromWkt("MULTILINESTRING((1 1,2 2))"))
        self.vector.beginEditCommand("Other feature")
        self.assertTrue(self.vector.addFeature(feature))
        self.vector.endEditCommand()
        return feature

    def test_finish_publishes_one_feature_with_whole_line_undo_redo(self):
        expected = self.trace()
        self.assertFalse(self.vector.isEditCommandActive())
        self.assertEqual(self.vector.undoStack().count(), 0)
        self.assertTrue(self.tool.finish_session())
        self.assertEqual(self.vector.undoStack().count(), 1)
        fid = self.assert_finished_line_undo(expected)
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.geometry(fid), expected)

    def test_hundreds_of_segments_do_not_retain_geometry_undo_commands(self):
        self.tool.accept_click((1000, 1990))
        for index in range(1, 201):
            self.tool.accept_click((1000 + index, 1990 + index % 2))
        self.assertEqual(len(self.tool.anchors), 201)
        self.assertEqual(self.tool.session.geometry.constGet().nCoordinates(), 201)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.vector.undoStack().count(), 0)
        expected = self.draft_geometry()
        self.tool.finish_session()
        stack = self.vector.undoStack()
        self.assertEqual(stack.count(), 1)
        # A macro containing every prior geometry would have identical visible
        # undo behavior but retain the quadratic memory growth being prevented.
        self.assertLessEqual(stack.command(0).childCount(), 2)
        self.assert_finished_line_undo(expected)

    def test_draft_is_drawn_until_publication_and_updates_after_b(self):
        self.trace()
        self.assertTrue(self.tool.draft_band.isVisible())
        rendered = self.tool.draft_band.asGeometry()
        self.assertEqual(
            [(p.x(), p.y()) for p in rendered.vertices()],
            [(999, 1990), (1010, 1990), (1021, 1990)],
        )
        self.key(Qt.Key.Key_B)
        self.assertTrue(self.tool.draft_band.isVisible())
        rendered = self.tool.draft_band.asGeometry()
        self.assertEqual(
            [(p.x(), p.y()) for p in rendered.vertices()],
            [(999, 1990), (1010, 1990)],
        )
        expected = self.draft_geometry()
        self.tool.finish_session()
        self.assertFalse(self.tool.draft_band.isVisible())
        self.assert_finished_line_undo(expected)

    def test_b_truncates_dense_and_closed_segments_then_can_retrace(self):
        first = self.trace(((999, 1990), (1010, 1990)))
        self.tool.tracing_mode = TracingModes.DENSE_LINE
        self.tool.accept_click((1021, 1990))
        dense = self.draft_geometry()
        self.tool.tracing_mode = TracingModes.LINE
        self.tool.accept_click((999, 1990))
        self.assertNotEqual(self.draft_geometry(), dense)
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.draft_geometry(), dense)
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.draft_geometry(), first)
        self.assertEqual(len(self.tool.anchors), 2)
        self.tool.tracing_mode = TracingModes.DENSE_LINE
        self.tool.accept_click((1021, 1990))
        self.assertEqual(self.draft_geometry(), dense)
        self.tool.finish_session()
        self.assert_finished_line_undo(dense)

    def test_b_first_segment_keeps_start_anchor_and_can_retrace(self):
        self.trace(((999, 1990), (1010, 1990)))
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.vector.undoStack().count(), 0)
        self.assertFalse(self.vector.isEditCommandActive())
        self.assertEqual(len(self.tool.anchors), 1)
        self.assertIsNone(self.tool.session.geometry)
        self.tool.accept_click((1021, 1990))
        expected = self.draft_geometry()
        self.tool.finish_session()
        self.assert_finished_line_undo(expected)

    def test_b_start_anchor_leaves_no_empty_undo_action(self):
        self.tool.accept_click((999, 1990))
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.tool.anchors, ())
        self.assertIsNone(self.tool.session.geometry)
        self.assertFalse(self.vector.isEditCommandActive())
        self.assertEqual(self.vector.undoStack().count(), 0)

    def test_unrelated_edits_before_and_after_have_independent_undo(self):
        other = self.add_external_feature()
        expected = self.trace()
        self.tool.finish_session()
        fid = next(f.id() for f in self.vector.getFeatures() if f.id() != other.id())
        self.vector.beginEditCommand("Move other feature")
        self.assertTrue(
            self.vector.changeGeometry(
                other.id(), QgsGeometry.fromWkt("MULTILINESTRING((3 3,4 4))")
            )
        )
        self.vector.endEditCommand()
        stack = self.vector.undoStack()
        self.assertEqual(stack.count(), 3)
        stack.undo()
        self.assertEqual(self.geometry(fid), expected)
        self.assertEqual(self.geometry(other.id()), bytes(other.geometry().asWkb()))
        stack.undo()
        self.assertFalse(self.vector.getFeature(fid).isValid())
        self.assertTrue(self.vector.getFeature(other.id()).isValid())
        stack.undo()
        self.assertEqual(self.vector.featureCount(), 0)
        stack.redo()
        stack.redo()
        self.assertEqual(self.geometry(fid), expected)

    def test_external_geometry_edits_and_undo_do_not_discard_draft(self):
        other = self.add_external_feature()
        expected = self.trace()
        self.vector.beginEditCommand("Move other feature")
        self.assertTrue(
            self.vector.changeGeometry(
                other.id(), QgsGeometry.fromWkt("MULTILINESTRING((3 3,4 4))")
            )
        )
        self.vector.endEditCommand()
        self.assertEqual(self.draft_geometry(), expected)
        stack = self.vector.undoStack()
        stack.undo()
        self.assertEqual(self.draft_geometry(), expected)
        self.assertEqual(self.geometry(other.id()), bytes(other.geometry().asWkb()))
        self.key(Qt.Key.Key_B)
        self.assertEqual(len(self.tool.anchors), 2)
        self.assertEqual(self.geometry(other.id()), bytes(other.geometry().asWkb()))
        stack.undo()
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(len(self.tool.anchors), 2)
        expected = self.draft_geometry()
        self.tool.finish_session()
        self.assert_finished_line_undo(expected)

    def test_external_add_cancels_pending_search_without_discarding_draft(self):
        submitted = self.control_tasks()
        expected = self.trace(((1002, 1992), (1008, 1992)))
        self.tool.tracing_mode = TracingModes.PATH
        self.accept(8, 13)
        pending = submitted[-1]
        other = self.add_external_feature()
        pending.finish()
        self.assertFalse(self.tool.tracking_is_active)
        self.assertEqual(self.draft_geometry(), expected)
        self.assertEqual(len(self.tool.anchors), 2)
        self.key(Qt.Key.Key_B)
        self.assertIsNone(self.tool.session.geometry)
        self.assertEqual(self.geometry(other.id()), bytes(other.geometry().asWkb()))

    def test_failed_later_segment_retains_previous_draft(self):
        expected = self.trace(((999, 1990), (1010, 1990)))
        anchors = self.tool.anchors
        with patch.object(
            self.tool, "build_path_geometry", side_effect=ValueError("transform failed")
        ):
            self.tool.accept_click((1021, 1990))
        self.assertEqual(self.tool.anchors, anchors)
        self.assertEqual(self.draft_geometry(), expected)
        self.assertFalse(self.tool.tracking_is_active)
        self.assertEqual(self.vector.undoStack().count(), 0)
        self.tool.accept_click((1021, 1990))
        self.assertEqual(len(self.tool.anchors), 3)
        expected = self.draft_geometry()
        self.tool.finish_session()
        self.assert_finished_line_undo(expected)

    def test_failed_publication_preserves_draft_for_retry(self):
        expected = self.trace()
        anchors = self.tool.anchors
        with patch.object(self.vector, "addFeature", return_value=False):
            self.assertFalse(self.tool.finish_session())
        self.assertEqual(self.tool.anchors, anchors)
        self.assertEqual(self.draft_geometry(), expected)
        self.assertFalse(self.vector.isEditCommandActive())
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.vector.undoStack().count(), 0)
        self.assertTrue(self.tool.finish_session())
        self.assert_finished_line_undo(expected)

    def test_publication_does_not_close_or_join_another_edit_command(self):
        expected = self.trace()
        self.vector.beginEditCommand("Another tool")
        self.assertFalse(self.tool.finish_session())
        self.assertTrue(self.vector.isEditCommandActive())
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.draft_geometry(), expected)
        self.vector.endEditCommand()
        self.assertTrue(self.tool.finish_session())
        self.assert_finished_line_undo(expected)

    def test_pending_b_and_escape_keep_accepted_segments(self):
        submitted = self.control_tasks()
        expected = self.trace(((1002, 1992), (1008, 1992)))
        self.tool.tracing_mode = TracingModes.PATH
        for key in (Qt.Key.Key_B, Qt.Key.Key_Escape):
            self.accept(8, 13)
            pending = submitted[-1]
            self.key(key)
            pending.finish()  # Deliberately deliver success after cancellation.
            self.assertFalse(self.tool.tracking_is_active)
            self.assertEqual(len(self.tool.anchors), 2)
            self.assertEqual(self.draft_geometry(), expected)
            self.assertEqual(self.vector.undoStack().count(), 0)
        self.tool.finish_session()
        self.assert_finished_line_undo(expected)

    def test_tool_switch_publishes_draft_and_rejects_pending_result(self):
        submitted = self.control_tasks()
        expected = self.trace(((1002, 1992), (1008, 1992)))
        self.tool.tracing_mode = TracingModes.PATH
        self.accept(8, 13)
        pending = submitted[-1]
        self.canvas.setMapTool(self.iface.pan_tool)
        pending.finish()
        self.assert_finished_line_undo(expected)

    def test_context_change_publishes_draft(self):
        expected = self.trace()
        self.canvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        self.assert_finished_line_undo(expected)

    def test_target_removal_publishes_before_layer_leaves_project(self):
        expected = self.trace()
        layer = self.project.takeMapLayer(self.vector)
        self.assertIs(layer, self.vector)
        self.assert_finished_line_undo(expected)
        self.project.addMapLayer(layer)

    def test_close_publishes_draft(self):
        expected = self.trace()
        self.plugin.dockwidget.close()
        self.assert_finished_line_undo(expected)

    def test_failed_publication_vetoes_dock_close_and_can_be_retried(self):
        expected = self.trace()
        dock = self.plugin.dockwidget
        with patch.object(self.vector, "addFeature", return_value=False):
            self.assertFalse(dock.close())
        self.assertIs(self.plugin.dockwidget, dock)
        self.assertTrue(self.plugin.pluginIsActive)
        self.assertFalse(self.tool.disposed)
        self.assertEqual(self.draft_geometry(), expected)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertEqual(self.vector.undoStack().count(), 0)
        self.assertTrue(dock.close())
        self.assertTrue(self.tool.disposed)
        self.assert_finished_line_undo(expected)

    def test_unload_publishes_draft(self):
        expected = self.trace()
        self.plugin.unload()
        self.assert_finished_line_undo(expected)

    def test_host_unload_keeps_failed_draft_registered_then_retries(self):
        expected = self.trace()
        dock = self.plugin.dockwidget
        name = "raster_scribe_unload_test"
        # Exercise QGIS's real registry handling without unloading the package
        # modules which the rest of this test process still needs.
        with (
            patch.object(qgis_utils, "plugins", {name: self.plugin}),
            patch.object(qgis_utils, "active_plugins", [name]),
            patch.object(qgis_utils, "_plugin_modules", {}),
            patch.object(qgis_utils, "showException") as report,
        ):
            with patch.object(self.vector, "addFeature", return_value=False):
                self.assertFalse(qgis_utils.unloadPlugin(name))
            report.assert_called_once()
            self.assertIs(qgis_utils.plugins[name], self.plugin)
            self.assertIn(name, qgis_utils.active_plugins)
            self.assertIs(self.plugin.dockwidget, dock)
            self.assertTrue(self.plugin.pluginIsActive)
            self.assertFalse(self.plugin._unloaded)
            self.assertFalse(self.tool.disposed)
            self.assertEqual(self.draft_geometry(), expected)
            self.assertEqual(self.vector.featureCount(), 0)
            self.assertTrue(qgis_utils.unloadPlugin(name))
            self.assertNotIn(name, qgis_utils.plugins)
            self.assertNotIn(name, qgis_utils.active_plugins)
            report.assert_called_once()
        self.assertTrue(self.plugin._unloaded)
        self.assertTrue(self.tool.disposed)
        self.assertIsNone(self.plugin.dockwidget)
        self.assert_finished_line_undo(expected)

    def test_save_publishes_draft_with_and_without_stopping_editing(self):
        for stop_editing in (False, True):
            with self.subTest(stop_editing=stop_editing):
                expected = self.trace()
                self.assertTrue(self.vector.commitChanges(stop_editing))
                self.assertEqual(self.tool.anchors, ())
                self.assertFalse(self.vector.isEditCommandActive())
                self.assertEqual(self.vector.isEditable(), not stop_editing)
                self.assertEqual(self.vector.undoStack().count(), 0)
                self.assertEqual(self.vector.featureCount(), 1)
                saved = next(self.vector.getFeatures())
                self.assertEqual(bytes(saved.geometry().asWkb()), expected)
                if stop_editing:
                    self.assertTrue(self.vector.startEditing())
                for point in ((1000, 1980), (1010, 1980)):
                    self.tool.accept_click(point)
                self.assertEqual(self.vector.featureCount(), 1)
                self.tool.finish_session()
                self.assertEqual(self.vector.featureCount(), 2)
                self.vector.undoStack().undo()
                self.assertEqual(self.vector.featureCount(), 1)
                self.assertEqual(self.geometry(saved.id()), expected)
                self.assertTrue(self.vector.deleteFeature(saved.id()))
                self.assertTrue(self.vector.commitChanges(False))

    def test_rollback_discards_draft_and_pending_work(self):
        submitted = self.control_tasks()
        self.trace(((1002, 1992), (1008, 1992)))
        self.tool.tracing_mode = TracingModes.PATH
        self.accept(8, 13)
        pending = submitted[-1]
        self.assertTrue(self.vector.rollBack())
        pending.finish()
        self.assertEqual(self.tool.anchors, ())
        self.assertIsNone(self.tool.session.geometry)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertFalse(self.vector.isEditCommandActive())
        self.assertEqual(self.vector.undoStack().count(), 0)

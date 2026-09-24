"""Public input and deterministic late-terminal regressions with real QGIS edits."""

import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from osgeo import gdal
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QEvent, QPoint, QSettings, Qt
from qgis.PyQt.QtGui import QColor, QKeyEvent

from ..pointtool import TracingModes
from ..pointtool_session import TraceResult
from ..pointtool_tasks import FindPathTask, execute_request
from .tracing_fixture import TraceFixture
from .utilities import create_rgb_raster, wait_until


class ControlledTask:
    """Deliver terminal outcomes explicitly, including success after cancellation."""

    def __init__(self, work, snapshot, callback):
        self.work, self.snapshot, self.callback = work, snapshot, callback
        self.outcome = None
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def finish(self, status=None):
        result = (
            execute_request(self.work, self.snapshot)
            if status is None
            else TraceResult(self.work.request_id, status)
        )
        self.callback(self, result)


class TracingHardeningTest(TraceFixture):
    def setUp(self):
        super().setUp()
        self.submitted = []
        self.scheduler = self.plugin.task_controller
        self.scheduler._factory = ControlledTask
        self.scheduler._submit = self.submitted.append

    def key(self, key):
        self.tool.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        )

    def preview(self, row=8, col=13, start=True):
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(row, col))
        screen = self.tool.toCanvasCoordinates(QgsPointXY(*endpoint.xy))
        request = self.tool.make_request(endpoint, screen)
        self.tool.preview_controller.queue(request)
        if start:
            self.tool.preview_controller.ensure_inflight_started()
        return endpoint, screen

    def commit_segment(self, start=(8, 2), goal=(8, 8)):
        if not self.tool.anchors:
            self.accept(*start)
        self.accept(*goal)
        self.submitted[-1].finish()
        self.assertFalse(self.tool.tracking_is_active)

    def test_escape_first_later_and_immediate_restart(self):
        for later in (False, True):
            with self.subTest(later=later):
                self.tool.finish_session()
                if later:
                    self.commit_segment()
                else:
                    self.accept(8, 2)
                previous = (
                    self.vector.featureCount(),
                    self.tool.current_feature_id,
                    self.tool.anchors,
                )
                self.accept(8, 12)
                old = self.submitted[-1]
                self.key(Qt.Key.Key_Escape)
                self.assertFalse(self.tool.tracking_is_active)
                self.assertEqual(self.tool.anchors, previous[2])
                self.accept(8, 13)
                self.assertTrue(self.tool.tracking_is_active)
                self.assertIs(self.scheduler._task, old)
                old.finish()  # Force stale success even though cancel was requested.
                self.assertEqual(
                    (self.vector.featureCount(), self.tool.current_feature_id),
                    previous[:2],
                )
                self.assertTrue(self.tool.tracking_is_active)
                self.submitted[-1].finish()
                self.assertEqual(len(self.tool.anchors), len(previous[2]) + 1)

    def test_preview_adoption_freezes_mouse_and_debounce(self):
        for queued in (True, False):
            self.tool.finish_session()
            self.accept(8, 2)
            endpoint, screen = self.preview(start=not queued)
            self.tool.accept_click(endpoint.xy, screen)
            self.assertTrue(self.tool.session.pending.adopted_preview)
            task = self.submitted[-1]
            self.tool._hover(QPoint(screen.x() + 40, screen.y()))
            self.tool.preview_controller.ensure_inflight_started()
            self.assertIs(self.scheduler._task, task)
            before = self.vector.featureCount()
            task.finish()
            self.assertEqual(self.vector.featureCount(), before + 1)
            self.assertEqual(self.tool.anchors[-1], endpoint)

    def test_return_to_running_or_cached_preview_discards_pending_hover(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                self.tool.finish_session()
                self.accept(8, 2)
                self.preview()
                task = self.submitted[-1]
                if cached:
                    task.finish()
                preview = self.tool.preview_controller
                self.preview(col=10, start=False)
                self.assertTrue(preview._timer.isActive())
                self.preview(start=False)
                self.assertIsNone(preview._pending_request)
                self.assertFalse(preview._timer.isActive())
                submitted = len(self.submitted)
                preview._execute_preview_request()
                self.assertFalse(task.cancelled)
                if not cached:
                    task.finish()
                self.assertEqual(len(self.submitted), submitted)
                self.assertFalse(self.scheduler.active)
                self.assertTrue(preview._rubber_band.isVisible())
                self.assertEqual(preview._cached_result.path, tuple(self.expected_path))

    def test_repeated_pending_hover_keeps_debounce_until_submission(self):
        self.accept(8, 2)
        self.preview(start=False)
        preview = self.tool.preview_controller
        pending = preview._pending_request
        self.preview(start=False)
        self.assertTrue(preview._timer.isActive())
        self.assertIs(preview._pending_request, pending)
        self.assertFalse(self.submitted)
        preview.ensure_inflight_started()
        self.assertEqual(len(self.submitted), 1)
        self.submitted[-1].finish()
        self.assertTrue(preview._rubber_band.isVisible())

    def test_escape_queued_and_running_adopted_preview(self):
        for queued in (True, False):
            self.tool.finish_session()
            self.accept(8, 2)
            endpoint, screen = self.preview(start=not queued)
            self.tool.accept_click(endpoint.xy, screen)
            old = self.submitted[-1]
            self.key(Qt.Key.Key_Escape)
            self.accept(8, 12)
            old.finish()
            self.assertEqual(self.vector.featureCount(), 0 if queued else 1)
            self.submitted[-1].finish()
            self.assertFalse(self.tool.tracking_is_active)

    def test_terminal_failures_release_endpoint_and_next_click(self):
        self.accept(8, 2)
        for status in (
            "cancelled",
            "error",
            "no_path",
            "invalid_input",
            "resource_limit",
        ):
            with self.subTest(status=status):
                self.accept(8, 13)
                self.submitted[-1].finish(status)
                self.assertFalse(self.tool.tracking_is_active)
                self.assertEqual(len(self.tool.anchors), 1)
                self.assertFalse(self.tool._pending_markers)
                self.assertFalse(self.scheduler.active)
        self.accept(8, 13)
        self.submitted[-1].finish()
        self.assertEqual(self.vector.featureCount(), 1)

    def test_task_adapter_reports_false_exception_and_cancel_once(self):
        self.accept(8, 2)
        request = self.tool.make_request(
            self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        )
        for cancelled in (False, True):
            outcomes = []
            task = FindPathTask(
                request.worker_input(),
                None,
                lambda task, result: outcomes.append(result),
            )
            if cancelled:
                task.cancel()
            task.finished(False)
            task.finished(False)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0].status, "cancelled" if cancelled else "error")
        outcomes = []
        task = FindPathTask(
            request.worker_input(), None, lambda task, result: outcomes.append(result)
        )
        with patch(
            "raster_scribe.pointtool_tasks.execute_request",
            side_effect=RuntimeError("preparation error"),
        ):
            self.assertFalse(task.run())
        task.finished(False)
        self.assertEqual(outcomes[0].status, "error")

    def test_semantic_changes_revoke_queued_running_cached_preview(self):
        changes = [
            lambda: self.tool.trace_color_changed(QColor("black")),
            lambda: setattr(self.tool, "smooth_line", True),
            lambda: self.tool.snap_tolerance_changed(2),
            lambda: self.tool.snap2_tolerance_changed(2),
            lambda: setattr(self.tool, "tracing_mode", TracingModes.LINE),
        ]
        for change in changes:
            for state in ("queued", "running", "cached"):
                self.tool.finish_session()
                self.tool.trace_color_changed(False)
                self.tool.smooth_line = False
                self.tool.snap_tolerance_changed(None)
                self.tool.snap2_tolerance_changed(None)
                self.tool.tracing_mode = TracingModes.PATH
                self.accept(8, 2)
                endpoint, screen = self.preview(start=state != "queued")
                old = self.submitted[-1] if state != "queued" else None
                if state == "cached":
                    old.finish()
                before = self.vector.featureCount()
                change()
                self.tool.accept_click(endpoint.xy, screen)
                if state == "running":
                    old.finish()
                if self.scheduler.active:
                    self.submitted[-1].finish()
                self.assertEqual(self.vector.featureCount(), before + 1)
                self.assertFalse(self.tool.tracking_is_active)

    def test_preview_disable_and_presentation_changes(self):
        self.accept(8, 2)
        endpoint, screen = self.preview()
        self.tool.accept_click(endpoint.xy, screen)
        self.tool.set_preview_enabled(False)
        self.assertFalse(self.tool.tracking_is_active)
        self.submitted[-1].finish()
        self.assertEqual(self.vector.featureCount(), 0)
        self.accept(8, 13)
        request_id = self.tool.session.pending.request.request_id
        revision = self.tool.session.revision
        self.tool.set_preview_enabled(True)
        self.tool.set_preview_enabled(False)
        self.tool.set_preview_color(QColor("green"))
        self.tool.set_preview_width(3)
        self.tool.smooth_line = self.tool.smooth_line
        self.tool.snap_tolerance_changed(self.tool.snap_tolerance)
        self.tool.trace_color_changed(False)
        self.assertEqual(self.tool.session.revision, revision)
        self.assertEqual(self.tool.session.pending.request.request_id, request_id)
        self.submitted[-1].finish()
        self.assertEqual(self.vector.featureCount(), 1)

    def test_context_boundaries_reject_late_success(self):
        other_path = Path(self.directory.name) / "other.tif"
        create_rgb_raster(other_path)
        dataset = gdal.Open(str(other_path), gdal.GA_Update)
        dataset.SetGeoTransform((5000, 2, 0, 6000, 0, -2))
        dataset = None
        other_raster = QgsRasterLayer(str(other_path), "other")
        self.project.addMapLayer(other_raster)
        other_vector = QgsVectorLayer(
            "MultiLineString?crs=EPSG:3857", "other", "memory"
        )
        self.project.addMapLayer(other_vector)
        other_vector.startEditing()
        changed_context = QgsCoordinateTransformContext()
        self.assertTrue(
            changed_context.addCoordinateOperation(
                QgsCoordinateReferenceSystem("EPSG:4326"),
                QgsCoordinateReferenceSystem("EPSG:3857"),
                "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
                "+step +proj=webmerc +ellps=WGS84",
            )
        )
        actions = [
            lambda: self.tool.raster_layer_has_changed(other_raster),
            lambda: self.tool.raster_layer_has_changed(None),
            lambda: self.tool.raster_layer_has_changed(
                QgsRasterLayer("/no/file", "invalid")
            ),
            lambda: self.canvas.setDestinationCrs(
                QgsCoordinateReferenceSystem("EPSG:4326")
            ),
            lambda: self.project.setCrs(QgsCoordinateReferenceSystem("EPSG:4326")),
            lambda: self.project.setTransformContext(changed_context),
            lambda: self.raster.setCrs(QgsCoordinateReferenceSystem("EPSG:4326")),
            lambda: self.vector.setCrs(QgsCoordinateReferenceSystem("EPSG:4326")),
            lambda: self.iface.setActiveLayer(other_vector),
            lambda: self.vector.rollBack(),
            lambda: self.canvas.setMapTool(self.iface.pan_tool),
        ]
        for index, action in enumerate(actions):
            with self.subTest(boundary=index):
                self.project.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
                self.canvas.setDestinationCrs(self.project.crs())
                self.vector.setCrs(self.project.crs())
                self.raster.setCrs(self.project.crs())
                self.iface.setActiveLayer(self.vector)
                if not self.vector.isEditable():
                    self.vector.startEditing()
                self.plugin.run()
                self.tool.raster_layer_has_changed(self.raster)
                self.accept(8, 2)
                self.accept(8, 13)
                old = self.submitted[-1]
                action()
                old.finish()
                self.assertEqual(self.vector.featureCount(), 0)
                self.assertEqual(other_vector.featureCount(), 0)
                self.assertFalse(self.tool.has_active_trace())

    def test_removed_raster_and_target(self):
        for remove_raster in (False, True):
            self.accept(8, 2)
            self.accept(8, 13)
            task = self.submitted[-1]
            if remove_raster:
                self.project.removeMapLayer(self.raster)
            else:
                old_id = self.vector.id()
                self.project.takeMapLayer(self.vector)
                self.assertIsNone(self.project.mapLayer(old_id))
            task.finish()
            self.assertFalse(self.tool.has_active_trace())
            if not remove_raster:
                self.project.addMapLayer(self.vector)
                self.iface.setActiveLayer(self.vector)

    def test_close_reopen_and_unload_keep_one_running_worker(self):
        self.accept(8, 2)
        self.accept(8, 13)
        old_tool, old_task = self.tool, self.submitted[-1]
        self.plugin.dockwidget.close()
        self.assertTrue(old_tool.disposed)
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.tool.raster_layer_has_changed(self.raster)
        self.accept(8, 2)
        self.accept(8, 12)
        self.assertIs(self.scheduler._task, old_task)
        old_task.finish()
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertIsNot(self.scheduler._task, old_task)
        current = self.submitted[-1]
        self.plugin.unload()
        current.finish()
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertFalse(self.scheduler.active)
        self.assertIsNone(self.plugin.layer_tree_filter)

    def test_b_initial_pending_committed_and_external_edit(self):
        self.accept(8, 2)
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.tool.anchors, ())
        self.assertEqual(self.vector.undoStack().index(), 0)
        self.commit_segment()
        fid = self.tool.current_feature_id
        self.accept(8, 13)
        old = self.submitted[-1]
        self.key(Qt.Key.Key_B)
        old.finish()
        self.assertEqual(self.tool.current_feature_id, fid)
        self.assertEqual(len(self.tool.anchors), 2)
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertIsNone(self.tool.current_feature_id)
        self.assertEqual(len(self.tool.anchors), 1)
        self.commit_segment(goal=(8, 13))
        self.assertEqual(self.vector.featureCount(), 1)
        self.vector.beginEditCommand("unrelated")
        feature = QgsFeature(self.vector.fields())
        feature.setGeometry(
            QgsGeometry.fromMultiPolylineXY(
                [[QgsPointXY(1001, 1991), QgsPointXY(1002, 1991)]]
            )
        )
        self.vector.addFeature(feature)
        self.vector.endEditCommand()
        index = self.vector.undoStack().index()
        self.assertFalse(self.tool.has_active_trace())
        self.key(Qt.Key.Key_B)
        self.assertEqual(self.vector.undoStack().index(), index)
        self.assertEqual(self.vector.featureCount(), 2)

    def test_failed_add_change_transform_and_missing_feature(self):
        self.accept(8, 2)
        for method in ("addFeature", "changeGeometry"):
            if method == "changeGeometry":
                self.commit_segment()
            previous = self.tool.anchors
            geometry = (
                self.vector.getFeature(self.tool.current_feature_id).geometry().asWkt()
                if self.tool.current_feature_id is not None
                else None
            )
            self.accept(8, 13)
            with patch.object(self.vector, method, return_value=False):
                self.submitted[-1].finish()
            self.assertEqual(self.tool.anchors, previous)
            if geometry is not None:
                self.assertEqual(
                    self.vector.getFeature(self.tool.current_feature_id)
                    .geometry()
                    .asWkt(),
                    geometry,
                )
        self.accept(8, 13)
        with patch.object(
            self.tool, "build_path_points", side_effect=ValueError("transform failed")
        ):
            self.submitted[-1].finish()
        self.assertEqual(self.tool.anchors, previous)
        self.accept(8, 13)
        task = self.submitted[-1]
        self.vector.deleteFeature(self.tool.current_feature_id)
        task.finish()
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertFalse(self.tool.has_active_trace())

    def test_preview_defers_vector_conversion_until_commit(self):
        self.vector.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        capture = self.tool._capture_context
        transforms = []

        def capture_with_spy(layer):
            context = capture(layer)
            transform = Mock(wraps=context.map_to_vector)
            transforms.append(transform)
            return replace(context, map_to_vector=transform)

        with patch.object(self.tool, "_capture_context", side_effect=capture_with_spy):
            self.tool.accept_click((1002.1, 1991.9))
        endpoint, screen = self.preview()
        self.submitted[-1].finish()
        transform = transforms[0]
        transform.transform.assert_not_called()
        rendered = self.tool.preview_controller._rubber_band.asGeometry()
        self.assertFalse(rendered.isEmpty())

        # Commit needs layer points, not another discarded map geometry.
        with patch.object(
            QgsGeometry, "fromPolylineXY", wraps=QgsGeometry.fromPolylineXY
        ) as map_geometry:
            self.tool.accept_click(endpoint.xy, screen)
        map_geometry.assert_not_called()
        self.assertTrue(transform.transform.called)
        committed = self.vector.getFeature(self.tool.current_feature_id).geometry()
        self.assertEqual(
            len(list(committed.vertices())), len(list(rendered.vertices()))
        )
        for actual, preview in zip(committed.vertices(), rendered.vertices()):
            expected = transform.transform(QgsPointXY(preview))
            self.assertAlmostEqual(actual.x(), expected.x(), places=10)
            self.assertAlmostEqual(actual.y(), expected.y(), places=10)

        # A failed deferred transform still rejects the edit transaction.
        previous = self.tool.anchors
        original_geometry = committed.asWkt()
        endpoint, screen = self.preview(8, 12)
        transform.transform.reset_mock()
        transform.transform.side_effect = ValueError("transform failed")
        self.submitted[-1].finish()
        transform.transform.assert_not_called()
        self.assertFalse(
            self.tool.preview_controller._rubber_band.asGeometry().isEmpty()
        )
        self.tool.accept_click(endpoint.xy, screen)
        self.assertFalse(self.tool.tracking_is_active)
        self.assertEqual(self.tool.anchors, previous)
        self.assertEqual(
            self.vector.getFeature(self.tool.current_feature_id).geometry().asWkt(),
            original_geometry,
        )

    def test_preview_commit_exact_endpoints_same_pixel_and_noop(self):
        for smoothing in (False, True):
            self.tool.finish_session()
            self.tool.smooth_line = smoothing
            self.tool.accept_click((1002.1, 1991.9))
            endpoint = self.tool.resolve_endpoint((1013.2, 1991.7))
            request = self.tool.make_request(endpoint, QPoint(10, 20))
            self.tool.preview_controller.queue(request)
            self.tool.preview_controller.ensure_inflight_started()
            self.submitted[-1].finish()
            rendered = self.tool.preview_controller._rubber_band.asGeometry()
            self.tool.accept_click(endpoint.xy, QPoint(10, 20))
            committed = (
                self.vector.getFeature(self.tool.current_feature_id)
                .geometry()
                .asMultiPolyline()[0]
            )
            self.assertEqual([QgsPointXY(v) for v in rendered.vertices()], committed)
            self.assertEqual((committed[-1].x(), committed[-1].y()), endpoint.xy)
            anchor_count = len(self.tool.anchors)
            self.tool.accept_click(endpoint.xy)
            self.assertEqual(len(self.tool.anchors), anchor_count)
        self.tool.finish_session()
        self.tool.accept_click((1002.1, 1991.9))
        self.tool.accept_click((1002.2, 1991.8))
        self.submitted[-1].finish()
        line = (
            self.vector.getFeature(self.tool.current_feature_id)
            .geometry()
            .asMultiPolyline()[0]
        )
        self.assertEqual(len(line), 2)
        self.assertEqual((line[-1].x(), line[-1].y()), (1002.2, 1991.8))

    def test_straight_and_dense_modes_outside_raster_and_junction(self):
        self.tool.tracing_mode = TracingModes.LINE
        self.tool.accept_click((999, 1990))
        self.tool.accept_click((1010, 1990))
        self.assertEqual(len(self.tool.anchors), 2)
        self.tool.tracing_mode = TracingModes.DENSE_LINE
        self.tool.accept_click((1021, 1990))
        line = (
            self.vector.getFeature(self.tool.current_feature_id)
            .geometry()
            .asMultiPolyline()[0]
        )
        self.assertEqual(
            [(p.x(), p.y()) for p in line],
            [(999, 1990), (1010, 1990), (1015, 1990), (1020, 1990), (1021, 1990)],
        )
        self.assertFalse(self.submitted)

    def test_layer_undo_and_feature_id_collisions_do_not_cross_targets(self):
        self.commit_segment()
        other = QgsVectorLayer("MultiLineString?crs=EPSG:3857", "other", "memory")
        self.project.addMapLayer(other)
        other.startEditing()
        feature = QgsFeature(other.fields())
        feature.setGeometry(
            self.vector.getFeature(self.tool.current_feature_id).geometry()
        )
        other.addFeature(feature)
        self.accept(8, 13)
        task = self.submitted[-1]
        original = feature.geometry().asWkt()
        self.iface.setActiveLayer(other)
        task.finish()
        self.key(Qt.Key.Key_B)
        self.assertEqual(other.getFeature(feature.id()).geometry().asWkt(), original)
        self.iface.setActiveLayer(self.vector)
        self.accept(8, 2)
        self.accept(8, 13)
        self.vector.undoStack().undo()
        self.submitted[-1].finish()
        self.assertFalse(self.tool.has_active_trace())
        self.assertEqual(self.vector.featureCount(), 0)

    def test_startup_malformed_preferences_and_reactivation(self):
        self.plugin.dockwidget.close()
        settings = QSettings()
        for key, value in [
            ("preview/width", "nan"),
            ("snap/tolerance", "inf"),
            ("snap2/tolerance", 1e300),
            ("trace/smooth", "bad"),
            ("color/value", "bad"),
        ]:
            settings.setValue("RasterScribe/" + key, value)
        self.plugin.run()
        self.tool = self.plugin.tool_identify
        self.assertTrue(self.tool.smooth_line)
        self.assertEqual(self.plugin.dockwidget.previewWidthSpinBox.value(), 0.5)
        widget = self.plugin.dockwidget.checkBoxColor
        widget.blockSignals(True)
        with self.plugin._blocked(widget):
            pass
        self.assertTrue(widget.signalsBlocked())
        widget.blockSignals(False)
        self.canvas.setMapTool(self.iface.pan_tool)
        self.plugin.run()
        self.plugin.run()
        self.assertIs(self.plugin.last_maptool, self.iface.pan_tool)
        self.tool.raster_layer_has_changed(self.raster)
        self.commit_segment()


class WorkerThreadTest(TraceFixture):
    def test_external_task_cancellation_releases_pending_without_tool_cancel(self):
        from ..pointtool_raster import read_rgb

        entered, release = threading.Event(), threading.Event()

        def blocked(*args):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test barrier timed out")
            return read_rgb(*args)

        with patch("raster_scribe.pointtool_raster.read_rgb", side_effect=blocked):
            try:
                self.add_anchors()
                wait_until(entered.is_set)
                self.plugin.task_controller._task.cancel()
                release.set()
                wait_until(lambda: not self.tool.tracking_is_active)
                self.assertEqual(len(self.tool.anchors), 1)
                self.assertEqual(self.vector.featureCount(), 0)
                self.accept(8, 13)
                wait_until(lambda: not self.tool.tracking_is_active)
                self.assertEqual(self.vector.featureCount(), 1)
            finally:
                release.set()
                wait_until(lambda: QgsApplication.taskManager().countActiveTasks() == 0)

    def test_read_runs_off_gui_with_independent_handle_and_drains_after_close(self):
        from ..pointtool_raster import read_rgb

        entered, release = threading.Event(), threading.Event()
        captured = []
        gui_thread = threading.get_ident()
        gui_dataset = self.tool.raster_context.raster_sampler.dataset

        def blocked(dataset, *args):
            captured.append((threading.get_ident(), dataset is gui_dataset))
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test barrier timed out")
            return read_rgb(dataset, *args)

        with patch("raster_scribe.pointtool_raster.read_rgb", side_effect=blocked):
            try:
                self.add_anchors()
                wait_until(entered.is_set)
                old = self.plugin.task_controller._task
                self.plugin.dockwidget.close()
                self.plugin.run()
                self.tool = self.plugin.tool_identify
                self.tool.raster_layer_has_changed(self.raster)
                self.add_anchors()
                self.assertIs(self.plugin.task_controller._task, old)
                self.assertEqual(len(captured), 1)
                release.set()
                wait_until(lambda: not self.tool.tracking_is_active)
                self.assertEqual(self.vector.featureCount(), 1)
            finally:
                release.set()
                wait_until(lambda: QgsApplication.taskManager().countActiveTasks() == 0)
        self.assertTrue(
            all(thread != gui_thread and not shared for thread, shared in captured)
        )

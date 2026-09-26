"""Algorithm switches revoke all unaccepted work while preserving the draft."""

from dataclasses import replace

from qgis.core import QgsPointXY
from qgis.PyQt.QtCore import QPoint

from .tracing_fixture import TraceFixture


class EnhancedTracingLifecycleTest(TraceFixture):
    def setUp(self):
        super().setUp()
        self.submitted = self.control_tasks()

    def preview(self, *, start=True):
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        screen = self.tool.toCanvasCoordinates(QgsPointXY(*endpoint.xy))
        self.tool.last_mouse_event_pos = QPoint(screen)
        request = self.tool.make_request(endpoint, screen)
        self.tool.preview_controller.queue(request)
        if start:
            self.tool.preview_controller.ensure_inflight_started()
        return request, endpoint, screen

    def test_algorithm_is_part_of_identity_even_with_same_revision_and_endpoints(self):
        self.accept(8, 2)
        request, _, _ = self.preview()
        enhanced = replace(request, enhanced_tracing=True)
        self.assertNotEqual(request.key, enhanced.key)
        self.assertFalse(request.worker_input().enhanced_tracing)
        self.assertTrue(enhanced.worker_input().enhanced_tracing)
        self.assertIsNone(self.scheduler.matching(enhanced))
        self.submitted[-1].finish()
        self.assertIsNone(self.tool.preview_controller.matching_cached(enhanced))

    def test_cached_preview_is_discarded_and_replaced_before_click(self):
        self.accept(8, 2)
        request, _, _ = self.preview()
        self.submitted[-1].finish()
        preview = self.tool.preview_controller
        stale_result = preview._cached_result
        self.assertTrue(preview._rubber_band.isVisible())
        self.tool.enhanced_tracing = True
        self.assertFalse(preview._rubber_band.isVisible())
        self.assertIsNone(preview._cached_result)
        self.assertTrue(preview._pending_request.enhanced_tracing)
        self.assertGreater(preview._pending_request.revision, request.revision)
        self.tool._on_result(request, stale_result)
        self.assertFalse(preview._rubber_band.isVisible())
        self.assertIsNone(self.tool.session.geometry)
        preview.ensure_inflight_started()
        self.assertEqual(len(self.submitted), 2)
        self.assertTrue(self.submitted[-1].work.enhanced_tracing)
        self.submitted[-1].finish()
        self.assertTrue(preview._rubber_band.isVisible())
        current = preview._cached_request
        self.tool.accept_click(current.goal.xy, current.screen_pos)
        self.assertEqual(len(self.submitted), 2)  # Adopt only the new cached result.
        self.assertEqual(len(self.tool.anchors), 2)

    def test_switch_replaces_debounced_preview_before_any_worker_starts(self):
        self.accept(8, 2)
        old, _, _ = self.preview(start=False)
        self.assertFalse(self.submitted)
        self.tool.enhanced_tracing = True
        preview = self.tool.preview_controller
        self.assertNotEqual(preview._pending_request.request_id, old.request_id)
        self.assertTrue(preview._pending_request.enhanced_tracing)
        preview.ensure_inflight_started()
        self.assertEqual(len(self.submitted), 1)
        self.assertTrue(self.submitted[0].work.enhanced_tracing)
        self.submitted[0].finish()

    def test_rapid_switches_replace_queued_work_and_reject_late_success(self):
        self.accept(8, 2)
        self.preview()
        old = self.submitted[0]
        preview = self.tool.preview_controller
        self.tool.enhanced_tracing = True
        preview.ensure_inflight_started()
        self.assertTrue(old.cancelled)
        self.assertTrue(self.scheduler._queued.request.enhanced_tracing)
        self.assertEqual(len(self.submitted), 1)
        self.tool.enhanced_tracing = False
        preview.ensure_inflight_started()
        self.assertFalse(self.scheduler._queued.request.enhanced_tracing)
        replacement_id = self.scheduler._queued.request.request_id
        old.finish()  # Deliver actual success despite cancellation.
        self.assertEqual(len(self.submitted), 2)
        self.assertEqual(self.submitted[-1].work.request_id, replacement_id)
        self.assertFalse(self.submitted[-1].work.enhanced_tracing)
        self.assertFalse(preview._rubber_band.isVisible())
        self.assertEqual(len(self.tool.anchors), 1)
        self.assertIsNone(self.tool.session.geometry)
        self.submitted[-1].finish()
        self.assertTrue(preview._rubber_band.isVisible())
        self.assertFalse(preview._cached_request.enhanced_tracing)

    def test_clicked_pending_segment_is_cancelled_without_losing_accepted_draft(self):
        for adopted in (False, True):
            with self.subTest(adopted=adopted):
                self.tool.finish_session()
                self.tool.last_mouse_event_pos = None
                self.tool.enhanced_tracing = False
                self.accept(8, 2)
                self.accept(8, 6)
                self.submitted[-1].finish()
                previous = bytes(self.tool.session.geometry.asWkb())
                anchors = self.tool.anchors
                if adopted:
                    _, endpoint, screen = self.preview()
                    self.tool.accept_click(endpoint.xy, screen)
                    self.assertTrue(self.tool.session.pending.adopted_preview)
                else:
                    self.accept(8, 13)
                    point = self.tool.to_coords(8, 13)
                    self.tool.last_mouse_event_pos = self.tool.toCanvasCoordinates(
                        point
                    )
                    self.assertFalse(self.tool.session.pending.adopted_preview)
                old = self.submitted[-1]
                self.tool.enhanced_tracing = True
                self.assertTrue(old.cancelled)
                self.assertIsNone(self.tool.session.pending)
                self.assertFalse(self.tool._pending_markers)
                self.assertEqual(self.tool.anchors, anchors)
                self.assertEqual(bytes(self.tool.session.geometry.asWkb()), previous)
                preview = self.tool.preview_controller
                preview.ensure_inflight_started()
                old.finish()
                self.assertEqual(self.tool.anchors, anchors)
                self.assertEqual(bytes(self.tool.session.geometry.asWkb()), previous)
                self.assertTrue(self.submitted[-1].work.enhanced_tracing)
                self.submitted[-1].finish()
                # The replacement is only a preview; the cancelled click stays cancelled.
                self.assertEqual(self.tool.anchors, anchors)
                self.assertIsNone(self.tool.session.pending)
                self.assertTrue(preview._rubber_band.isVisible())
                current = preview._cached_request
                self.tool.accept_click(current.goal.xy, current.screen_pos)
                self.assertEqual(len(self.tool.anchors), len(anchors) + 1)

    def test_setting_same_algorithm_does_not_cancel_or_restart_work(self):
        self.accept(8, 2)
        self.preview()
        task = self.submitted[-1]
        revision = self.tool.session.revision
        self.tool.enhanced_tracing = False
        self.assertEqual(self.tool.session.revision, revision)
        self.assertFalse(task.cancelled)
        self.assertIsNone(self.scheduler._queued)
        task.finish()

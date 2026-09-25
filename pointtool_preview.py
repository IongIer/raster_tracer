"""Preview debounce, one numerical result cache and canvas rendering."""

import math
from dataclasses import replace

from qgis.core import Qgis, QgsGeometry
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor


class TracePreviewController:
    def __init__(self, tool):
        self._tool = tool
        self._enabled = True
        self._timer = QTimer(tool)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._execute_preview_request)
        self._pending_request = None
        self._current_request = None
        self._cached_request = None
        self._cached_result = None
        self._pixel_tolerance = 8
        self._rubber_band = QgsRubberBand(tool.canvas(), Qgis.GeometryType.Line)
        self._rubber_band.setColor(QColor(255, 20, 147))
        self._rubber_band.setWidth(2.7)
        self._rubber_band.setLineStyle(Qt.PenStyle.DashLine)
        self._rubber_band.hide()

    @property
    def enabled(self):
        return self._enabled

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if not enabled:
            self._tool.clear_preview()
        elif self._tool.last_mouse_event_pos is not None and not self._tool.suspended:
            self._tool._hover(self._tool.last_mouse_event_pos)

    def set_color(self, color):
        color = QColor(color)
        if color.isValid():
            self._rubber_band.setColor(color)

    def set_width(self, width):
        try:
            width = float(width)
        except (TypeError, ValueError, OverflowError):
            return
        if math.isfinite(width) and width > 0:
            self._rubber_band.setWidth(width)

    def clear(self, cancel=True):
        self._timer.stop()
        self._pending_request = self._current_request = None
        self._cached_request = self._cached_result = None
        self._rubber_band.hide()
        if cancel:
            self._tool.task_controller.cancel(
                self._tool.owner, speculative_only=True, evict=False
            )

    def dispose(self):
        self.clear(cancel=False)
        self._timer.timeout.disconnect(self._execute_preview_request)
        self._tool.canvas().scene().removeItem(self._rubber_band)

    def _matches(self, old, new):
        if (
            old is None
            or old.key != new.key
            or old.screen_pos is None
            or new.screen_pos is None
        ):
            return False
        return (
            sum((a - b) ** 2 for a, b in zip(old.screen_pos, new.screen_pos))
            <= self._pixel_tolerance**2
        )

    def matching_cached(self, request):
        if self.enabled and self._matches(self._cached_request, request):
            return self._cached_result
        return None

    def matching_request(self, request):
        if not self.enabled:
            return None
        if self._matches(self._pending_request, request):
            return self._pending_request
        candidate = self._tool.task_controller.matching(request)
        return candidate if self._matches(candidate, request) else None

    def queue(self, request):
        if not self.enabled or self._tool.tracking_is_active:
            return
        cached = self.matching_cached(request)
        if cached is not None:
            self._timer.stop()
            self._pending_request = None
            self._current_request = request
            self.receive(request, replace(cached, request_id=request.request_id))
            return
        matching = self.matching_request(request)
        if matching is not None:
            if matching is not self._pending_request:
                self._timer.stop()
                self._pending_request = None
            # Same numerical work can render the latest exact hover coordinates.
            self._current_request = replace(request, request_id=matching.request_id)
            return
        self._current_request = self._pending_request = request
        self._rubber_band.hide()
        self._timer.start(100)

    def ensure_inflight_started(self):
        self._timer.stop()
        self._execute_preview_request()

    def start_adopted(self, request_id):
        self._timer.stop()
        if (
            self._pending_request is not None
            and self._pending_request.request_id == request_id
        ):
            request, self._pending_request = self._pending_request, None
            self._tool.task_controller.submit(
                request, self._tool._on_result, committed=True
            )
        else:
            self._tool.task_controller.adopt(request_id)

    def _execute_preview_request(self):
        request, self._pending_request = self._pending_request, None
        if request is None or not self.enabled or self._tool.tracking_is_active:
            return
        if self._tool._valid_request(request):
            self._tool.task_controller.submit(request, self._tool._on_result)

    def receive(self, request, result):
        current = self._current_request
        if (
            not self.enabled
            or current is None
            or current.request_id != request.request_id
        ):
            return
        if result.status != "success":
            self._cached_request = self._cached_result = None
            self._rubber_band.hide()
            return
        try:
            points = self._tool.build_path_points(current, result, in_layer_crs=False)
            geometry = QgsGeometry.fromPolylineXY(points)
        except Exception:
            self._rubber_band.hide()
            return
        self._cached_request, self._cached_result = current, result
        self._rubber_band.setToGeometry(geometry, None)
        self._rubber_band.show()

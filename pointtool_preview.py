"""Preview controller for RasterTracer PointTool."""

from typing import Any, Dict, Iterable, Optional, Tuple

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsApplication, QgsGeometry, QgsPointXY, QgsWkbTypes
from qgis.gui import QgsRubberBand

from .astar import FindPathTask
from .exceptions import OutsideMapError
from .line_simplification import simplify, smooth


PreviewRequest = Dict[str, Any]


class TracePreviewController:
    """Encapsulates preview state and behaviour for ``PointTool``."""

    def __init__(self, tool):
        self._tool = tool

        self._enabled = True
        self._interval_ms = 200
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._execute_preview_request)

        self._pending_request: Optional[PreviewRequest] = None
        self._last_request: Optional[PreviewRequest] = None
        self._task: Optional[FindPathTask] = None
        self._sequence = 0

        self._rubber_band = QgsRubberBand(tool.canvas(), QgsWkbTypes.LineGeometry)
        self._color = QColor(255, 20, 147)
        self._width = 2.7
        self._rubber_band.setColor(self._color)
        self._rubber_band.setWidth(self._width)
        self._rubber_band.setLineStyle(Qt.DashLine)
        self._rubber_band.hide()

        self._cached_path: Optional[Iterable[Tuple[int, int]]] = None
        self._cached_request: Optional[PreviewRequest] = None
        self._cached_pos: Optional[Tuple[float, float]] = None
        self._pixel_tolerance = 8

        self._commit_request: Optional[PreviewRequest] = None
        self._commit_vlayer = None

    # ------------------------------------------------------------------
    # Public API used by PointTool
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._timer.stop()
        self._pending_request = None
        self._cancel_preview_task()
        self._sequence += 1
        if not enabled:
            self._last_request = None
            self._rubber_band.hide()
            self._clear_preview_cache()
            self._commit_request = None
            self._commit_vlayer = None

    def set_color(self, color) -> None:
        if color is None:
            return
        if not isinstance(color, QColor):
            color = QColor(color)
        self._color = QColor(color)
        self._rubber_band.setColor(self._color)

    def set_width(self, width) -> None:
        try:
            width_value = float(width)
        except (TypeError, ValueError):
            return
        self._width = width_value
        self._rubber_band.setWidth(self._width)

    def clear(self) -> None:
        self._timer.stop()
        self._pending_request = None
        self._last_request = None
        self._cancel_preview_task()
        self._sequence += 1
        self._rubber_band.hide()
        self._clear_preview_cache()
        self._commit_request = None
        self._commit_vlayer = None

    def take_path_if_valid(self, start_point, end_point, click_pos):
        if (not self._enabled) or self._cached_path is None:
            return None
        if not self._rubber_band.isVisible():
            return None
        if not self._request_matches_points(self._cached_request, start_point, end_point):
            return None
        if click_pos is None:
            return None
        cached_pos = self._cached_pos
        if cached_pos is None:
            return None
        dx = click_pos.x() - cached_pos[0]
        dy = click_pos.y() - cached_pos[1]
        if (dx * dx + dy * dy) > (self._pixel_tolerance ** 2):
            return None
        return [tuple(pt) for pt in self._cached_path]

    def has_inflight_preview_for(self, start_point, end_point) -> bool:
        if self._request_matches_points(self._pending_request, start_point, end_point):
            return True
        if self._task is not None and self._request_matches_points(self._last_request, start_point, end_point):
            return True
        return False

    def request_commit(self, request: PreviewRequest, vlayer) -> None:
        self._commit_request = dict(request)
        self._commit_vlayer = vlayer

    def clear_commit(self) -> None:
        self._commit_request = None
        self._commit_vlayer = None

    def ensure_inflight_started(self) -> None:
        if self._task is not None:
            return
        if self._pending_request is None:
            return
        self._timer.stop()
        self._execute_preview_request()

    def queue(self, start, goal, screen_pos=None) -> None:
        if not self._enabled:
            return
        request = {
            "start": tuple(start),
            "goal": tuple(goal),
        }
        if screen_pos is not None:
            request["screen_pos"] = (screen_pos.x(), screen_pos.y())
        if (
                self._pending_request is not None and
                self._requests_equivalent(self._pending_request, request)
        ):
            return
        if (
                self._task is None and
                self._requests_equivalent(self._last_request, request)
        ):
            return
        self._pending_request = request
        if self._timer.isActive():
            self._timer.stop()
        self._timer.start(self._interval_ms)

    def restart_last_request(self) -> None:
        if not self._enabled or self._last_request is None:
            return
        self._pending_request = dict(self._last_request)
        if self._timer.isActive():
            self._timer.stop()
        self._timer.start(self._interval_ms)

    def consume_commit(self, global_path) -> None:
        request = self._commit_request
        vlayer = self._commit_vlayer
        self._commit_request = None
        self._commit_vlayer = None
        if vlayer is None or not global_path:
            self._tool.tracking_is_active = False
            return
        path_to_draw = [tuple(node) for node in global_path]
        self._tool.tracking_is_active = True
        self.clear()
        self._tool.draw_path(path_to_draw, vlayer, was_tracing=True)

    def fallback_after_failure(self) -> None:
        if self._commit_request is None:
            return
        request = self._commit_request
        vlayer = self._commit_vlayer
        self._commit_request = None
        self._commit_vlayer = None
        if vlayer is None:
            self._tool.tracking_is_active = False
            return
        start = request.get("start")
        goal = request.get("goal")
        if start is None or goal is None:
            self._tool.tracking_is_active = False
            return
        self.clear()
        try:
            self._tool.trace_over_image(start, goal, do_it_as_task=True, vlayer=vlayer)
        except OutsideMapError:
            self._tool.tracking_is_active = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clear_preview_cache(self) -> None:
        self._cached_path = None
        self._cached_request = None
        self._cached_pos = None

    def _cancel_preview_task(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _requests_equivalent(self, request_a, request_b) -> bool:
        if not request_a or not request_b:
            return False
        start_a = tuple(request_a.get("start") or ())
        goal_a = tuple(request_a.get("goal") or ())
        start_b = tuple(request_b.get("start") or ())
        goal_b = tuple(request_b.get("goal") or ())
        return start_a == start_b and goal_a == goal_b

    def _request_matches_points(self, request, start_point, end_point) -> bool:
        if not request:
            return False
        return (
            tuple(request.get("start") or ()) == tuple(start_point) and
            tuple(request.get("goal") or ()) == tuple(end_point)
        )

    def _execute_preview_request(self) -> None:
        if not self._enabled:
            return
        request = self._pending_request
        self._pending_request = None
        if request is None:
            return
        request = dict(request)
        self._last_request = request
        self._start_preview_task(request)

    def _start_preview_task(self, request: PreviewRequest) -> None:
        start = request["start"]
        goal = request["goal"]
        try:
            preparation = self._tool._prepare_pathfinding(start, goal, reason="preview")
        except OutsideMapError:
            self._last_request = None
            self._rubber_band.hide()
            return

        grid = preparation["grid"]
        local_start = preparation["local_start"]
        local_goal = preparation["local_goal"]

        self._cancel_preview_task()
        self._sequence += 1
        current_sequence = self._sequence

        current_request = dict(request)

        def callback(
                path,
                _vlayer,
                seq=current_sequence,
                origin=preparation["origin"],
                req=current_request):
            origin_i, origin_j = origin
            self._preview_task_callback(path, origin_i, origin_j, seq, req)

        task = FindPathTask(grid, local_start, local_goal, callback, None)
        self._task = task
        QgsApplication.taskManager().addTask(task)

    def _preview_task_callback(self, path, origin_i, origin_j, sequence_id, request) -> None:
        if sequence_id != self._sequence or not self._enabled:
            return
        self._task = None
        if not path:
            self._rubber_band.hide()
            self._cached_path = None
            if (
                self._commit_request is not None and
                self._requests_equivalent(self._commit_request, request)
            ):
                self.fallback_after_failure()
            return

        global_path = [
            (int(local_i + origin_i), int(local_j + origin_j))
            for local_i, local_j in path
        ]

        self._cached_path = [tuple(node) for node in global_path]
        self._cached_request = dict(request)
        self._cached_pos = request.get("screen_pos")

        path_for_preview = list(global_path)

        if self._tool.smooth_line and len(path_for_preview) > 2:
            smoothed = smooth(path_for_preview, size=5)
            smoothed = simplify(smoothed)
        else:
            smoothed = path_for_preview

        points = []
        for global_i, global_j in smoothed:
            pt_xy = self._tool.to_coords(global_i, global_j)
            if not isinstance(pt_xy, QgsPointXY):
                pt_xy = QgsPointXY(pt_xy[0], pt_xy[1])
            points.append(pt_xy)

        if not points:
            self._rubber_band.hide()
            if (
                self._commit_request is not None and
                self._requests_equivalent(self._commit_request, request)
            ):
                self.fallback_after_failure()
            return

        geometry = QgsGeometry.fromPolylineXY(points)
        self._rubber_band.setToGeometry(geometry, None)
        if self._enabled:
            self._rubber_band.show()

        if (
            self._commit_request is not None and
            self._requests_equivalent(self._commit_request, request)
        ):
            self.consume_commit(global_path)

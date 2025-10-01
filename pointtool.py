'''
Main functionality of raster tracer.
'''

from enum import Enum
from collections import namedtuple
import os
import time

import numpy as np

from qgis.core import QgsPointXY, QgsPoint, QgsGeometry, QgsFeature, \
                      QgsVectorLayer, QgsProject, QgsWkbTypes, QgsApplication, \
                      QgsRectangle, QgsSpatialIndex, QgsMessageLog
from qgis.gui import QgsMapToolEmitPoint, QgsMapToolEdit, \
                     QgsRubberBand, QgsVertexMarker, QgsMapTool
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis
from qgis.core import QgsCoordinateTransform


from .astar import FindPathTask, FindPathFunction
from .line_simplification import smooth, simplify
from .utils import get_whole_raster, PossiblyIndexedImageError
from .pointtool_states import WaitingFirstPointState
from .exceptions import OutsideMapError

# An point on the map where the user clicked along the line
Anchor = namedtuple('Anchor', ['x', 'y', 'i', 'j'])

# Flag for experimental Autofollowing mode
ALLOW_AUTO_FOLLOWING = False

PROFILE_ENABLED = os.environ.get("RASTER_TRACER_PROFILE", "0") == "1"


class TracingModes(Enum):
    '''
    Possible Tracing Modes for Pointtool.
    LINE - straight line from start to end.
    PATH - tracing along color from start to end.
    AUTO - auto tracing mode along color in the given direction.
    '''

    LINE = 1
    PATH = 2
    AUTO = 3

    def next(self):
        '''
        Switches between LINE and PATH
        '''
        cls = self.__class__
        members = list(cls)

        if not ALLOW_AUTO_FOLLOWING:
            return members[0] if self.value == 2 else members[1]

        index = members.index(self) + 1
        if index >= len(members):
            index = 0
        return members[index]

    def is_tracing(self):
        '''
        Returns True if mode is PATH
        '''
        return True if self.value == 2 else False

    def is_auto(self):
        '''
        Returns True if mode is PATH
        '''
        return True if self.value == 3 else False


# Line styles for the rubber band
RUBBERBAND_LINE_STYLES = {
    TracingModes.PATH: Qt.DotLine,
    TracingModes.LINE: Qt.SolidLine,
    TracingModes.AUTO: Qt.DashDotLine,
    }


class PointTool(QgsMapToolEdit):
    '''
    Implementation of interactions of the user with the main map.
    Will called every time the user clicks on the map
    or hovers the mouse over the map.
    '''

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.clear_preview()
        self.deactivated.emit()

    def __init__(self, canvas, iface, turn_off_snap, smooth=False):
        '''
        canvas - link to the QgsCanvas of the application
        iface - link to the Qgis Interface
        turn_off_snap - flag sets snapping to the nearest color
        smooth - flag sets smoothing of the traced path
        '''

        self.iface = iface

        # list of Anchors for current line
        self.anchors = []

        # for keeping track of mouse event for rubber band updating
        self.last_mouse_event_pos = None

        self.tracing_mode = TracingModes.PATH

        self.turn_off_snap = turn_off_snap
        self.smooth_line = smooth

        # possible variants: gray_diff, as_is, color_diff (using v from hsv)
        self.grid_conversion = "gray_diff"

        # QApplication.restoreOverrideCursor()
        # QApplication.setOverrideCursor(Qt.CrossCursor)
        QgsMapToolEmitPoint.__init__(self, canvas)

        self.rlayer = None
        self.grid_changed = None
        self.snap_tolerance = None # snap to color
        self.snap2_tolerance = None # snap to itself
        self.vlayer = None
        self.grid = None
        self.sample = None
        self.raster_sampler = None
        self.window_origin = None
        self.window_shape = None
        self.window_padding = 1024
        self.min_window_size = 512
        self.trace_color_value = None
        self.preview_enabled = True
        self.preview_interval_ms = 200
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._execute_preview_request)
        self.preview_pending_request = None
        self.preview_last_request = None
        self.preview_task = None
        self.preview_sequence = 0
        self.preview_rubber_band = QgsRubberBand(self.canvas(), QgsWkbTypes.LineGeometry)
        self.preview_color = QColor(255, 20, 147)
        self.preview_width = 2.7
        self.preview_rubber_band.setColor(self.preview_color)
        self.preview_rubber_band.setWidth(self.preview_width)
        self.preview_rubber_band.setLineStyle(Qt.DashLine)
        self.preview_rubber_band.hide()

        self.tracking_is_active = False

        # False = not a polygon
        self.rubber_band = QgsRubberBand(self.canvas(), QgsWkbTypes.LineGeometry)
        self.markers = []
        self.marker_snap = QgsVertexMarker(self.canvas())
        self.marker_snap.setColor(QColor(255, 0, 255))

        self.find_path_task = None

        self.change_state(WaitingFirstPointState)

        self.last_vlayer = None

    def display_message(self,
                        title,
                        message,
                        level='Info',
                        duration=2,
                        ):
        '''
        Shows message bar to the user.
        `level` receives one of four possible string values:
            Info, Warning, Critical, Success
        '''

        LEVELS = {
            'Info': Qgis.Info,
            'Warning': Qgis.Warning,
            'Critical': Qgis.Critical,
            'Success': Qgis.Success,
        }

        self.iface.messageBar().pushMessage(
            title,
            message,
            LEVELS[level],
            duration)

    def change_state(self, state):
        self.state = state(self)

    def snap_tolerance_changed(self, snap_tolerance):
        self.snap_tolerance = snap_tolerance
        self.clear_preview()
        if snap_tolerance is None:
            self.marker_snap.hide()
        else:
            self.marker_snap.show()

    def snap2_tolerance_changed(self, snap_tolerance):
        self.snap2_tolerance = snap_tolerance**2
        # if snap_tolerance is None:
        #     self.marker_snap.hide()
        # else:
        #     self.marker_snap.show()

    def trace_color_changed(self, color):
        if color is False:
            self.trace_color_value = None
        else:
            r0, g0, b0, _ = color.getRgb()
            self.trace_color_value = (float(r0), float(g0), float(b0))

        self._recompute_trace_grid(reason="manual")
        if self.preview_enabled and self.preview_last_request is not None:
            self.preview_pending_request = self.preview_last_request
            if self.preview_timer.isActive():
                self.preview_timer.stop()
            self.preview_timer.start(self.preview_interval_ms)

    def set_preview_enabled(self, enabled):
        self.preview_enabled = enabled
        self.preview_timer.stop()
        self.preview_pending_request = None
        self._cancel_preview_task()
        self.preview_sequence += 1
        if not enabled:
            self.preview_last_request = None
            self.preview_rubber_band.hide()

    def set_preview_color(self, color):
        if color is None:
            return
        if not isinstance(color, QColor):
            color = QColor(color)
        self.preview_color = QColor(color)
        self.preview_rubber_band.setColor(self.preview_color)

    def set_preview_width(self, width):
        try:
            width = float(width)
        except (TypeError, ValueError):
            return
        self.preview_width = width
        self.preview_rubber_band.setWidth(self.preview_width)

    def clear_preview(self):
        self.preview_timer.stop()
        self.preview_pending_request = None
        self.preview_last_request = None
        self._cancel_preview_task()
        self.preview_sequence += 1
        self.preview_rubber_band.hide()

    def _cancel_preview_task(self):
        if self.preview_task is not None:
            self.preview_task.cancel()
            self.preview_task = None

    def _queue_preview(self, start, goal):
        if not self.preview_enabled:
            return
        if start == goal:
            self.clear_preview()
            return
        request = {
            "start": start,
            "goal": goal,
        }
        if request == self.preview_pending_request and self.preview_timer.isActive():
            return
        if request == self.preview_last_request and self.preview_task is None:
            return
        self.preview_pending_request = request
        if self.preview_timer.isActive():
            self.preview_timer.stop()
        self.preview_timer.start(self.preview_interval_ms)

    def _execute_preview_request(self):
        if not self.preview_enabled:
            return
        request = self.preview_pending_request
        self.preview_pending_request = None
        if request is None:
            return
        self.preview_last_request = request
        self._start_preview_task(request)

    def _start_preview_task(self, request):
        start = request["start"]
        goal = request["goal"]
        try:
            preparation = self._prepare_pathfinding(start, goal, reason="preview")
        except OutsideMapError:
            self.preview_last_request = None
            self.preview_rubber_band.hide()
            return

        grid = preparation["grid"]
        local_start = preparation["local_start"]
        local_goal = preparation["local_goal"]
        origin_i, origin_j = preparation["origin"]

        self._cancel_preview_task()
        self.preview_sequence += 1
        current_sequence = self.preview_sequence

        def callback(path, _vlayer, seq=current_sequence, origin=preparation["origin"]):
            self._preview_task_callback(path, origin[0], origin[1], seq)

        task = FindPathTask(grid, local_start, local_goal, callback, None)
        self.preview_task = task
        QgsApplication.taskManager().addTask(task)

    def _preview_task_callback(self, path, origin_i, origin_j, sequence_id):
        if sequence_id != self.preview_sequence or not self.preview_enabled:
            return
        self.preview_task = None
        if not path:
            self.preview_rubber_band.hide()
            return

        global_path = [
            (local_i + origin_i, local_j + origin_j) for local_i, local_j in path
        ]

        if self.smooth_line and len(global_path) > 2:
            smoothed = smooth(global_path, size=5)
            smoothed = simplify(smoothed)
        else:
            smoothed = global_path

        points = []
        for global_i, global_j in smoothed:
            pt_xy = self.to_coords(global_i, global_j)
            if not isinstance(pt_xy, QgsPointXY):
                pt_xy = QgsPointXY(pt_xy[0], pt_xy[1])
            points.append(pt_xy)

        if not points:
            self.preview_rubber_band.hide()
            return

        geometry = QgsGeometry.fromPolylineXY(points)
        self.preview_rubber_band.setToGeometry(geometry, None)
        if self.preview_enabled:
            self.preview_rubber_band.show()

    def _ensure_sampler(self):
        if self.raster_sampler is None and self.rlayer is not None:
            try:
                sampler = get_whole_raster(
                    self.rlayer,
                    QgsProject.instance(),
                )
            except PossiblyIndexedImageError:
                self.display_message(
                    "Missing Layer",
                    "Can't trace indexed or gray image",
                    level='Critical',
                    duration=2,
                    )
                self.raster_sampler = None
            else:
                self.raster_sampler = sampler
                self.to_indexes = self.raster_sampler.to_indexes
                self.to_coords = self.raster_sampler.to_coords
                self.to_coords_provider = self.raster_sampler.to_coords_provider
                self.to_coords_provider2 = self.raster_sampler.to_coords_provider2
                self.sample = None
                self.grid = None
                self.grid_changed = None
                self.window_origin = None
                self.window_shape = None
                self._recompute_trace_grid(reason="lazy-load")

    def _indices_inside_window(self, index):
        if self.window_origin is None or self.window_shape is None:
            return False
        origin_i, origin_j = self.window_origin
        height, width = self.window_shape
        i, j = index
        return (
            origin_i <= i < origin_i + height and
            origin_j <= j < origin_j + width
        )

    def _compute_window_bounds(self, indices, padding):
        if self.raster_sampler is None:
            return None

        height = self.raster_sampler.height
        width = self.raster_sampler.width

        clamped_i = []
        clamped_j = []
        for i, j in indices:
            clamped_i.append(max(0, min(int(i), height - 1)))
            clamped_j.append(max(0, min(int(j), width - 1)))

        if not clamped_i or not clamped_j:
            return None

        min_i = min(clamped_i)
        max_i = max(clamped_i)
        min_j = min(clamped_j)
        max_j = max(clamped_j)

        target_padding = max(int(padding), 0)

        i_min = max(0, min_i - target_padding)
        i_max = min(height, max_i + target_padding + 1)
        j_min = max(0, min_j - target_padding)
        j_max = min(width, max_j + target_padding + 1)

        min_height = min(self.min_window_size, height)
        min_width = min(self.min_window_size, width)

        current_height = i_max - i_min
        if current_height < min_height:
            needed = min_height - current_height
            extend_top = min(i_min, needed // 2)
            extend_bottom = min(height - i_max, needed - extend_top)
            i_min -= extend_top
            i_max += extend_bottom
            i_min = max(0, i_min)
            i_max = min(height, i_max)

        current_width = j_max - j_min
        if current_width < min_width:
            needed = min_width - current_width
            extend_left = min(j_min, needed // 2)
            extend_right = min(width - j_max, needed - extend_left)
            j_min -= extend_left
            j_max += extend_right
            j_min = max(0, j_min)
            j_max = min(width, j_max)

        return int(i_min), int(i_max), int(j_min), int(j_max)

    def _load_window(self, bounds, reason):
        if bounds is None or self.raster_sampler is None:
            return

        i_min, i_max, j_min, j_max = bounds
        load_start = time.perf_counter() if PROFILE_ENABLED else None

        bands, origin, shape = self.raster_sampler.read_window(
            i_min,
            i_max,
            j_min,
            j_max,
        )

        if bands is None or shape == (0, 0):
            return

        prep_start = time.perf_counter() if PROFILE_ENABLED else None
        cleaned_bands = []
        for band in bands:
            cleaned = np.nan_to_num(band, copy=False)
            cleaned_bands.append(cleaned)
        prep_duration = (
            time.perf_counter() - prep_start
        ) if PROFILE_ENABLED else None

        grid_start = time.perf_counter() if PROFILE_ENABLED else None
        grid = cleaned_bands[0] + cleaned_bands[1] + cleaned_bands[2]
        grid_duration = (
            time.perf_counter() - grid_start
        ) if PROFILE_ENABLED else None

        self.sample = tuple(cleaned_bands)
        self.grid = grid
        self.window_origin = origin
        self.window_shape = shape
        self.grid_changed = None

        total_duration = (
            time.perf_counter() - load_start
        ) if PROFILE_ENABLED else None

        if PROFILE_ENABLED:
            color_bytes = sum(arr.nbytes for arr in self.sample)
            grid_bytes = self.grid.nbytes if isinstance(self.grid, np.ndarray) else 0

            def _fmt(value):
                return f"{value:.2f}s" if value is not None else "n/a"

            QgsMessageLog.logMessage(
                (
                    "[profiling] window_prepare "
                    f"reason={reason} origin={origin} shape={shape} "
                    f"prep={_fmt(prep_duration)} grid_sum={_fmt(grid_duration)} "
                    f"total={_fmt(total_duration)} color_mb={(color_bytes / (1024 ** 2)):.1f} "
                    f"grid_mb={(grid_bytes / (1024 ** 2)):.1f}"
                ),
                "RasterTracer",
                Qgis.Info,
            )

        self._recompute_trace_grid(reason=f"window:{reason}")

    def _ensure_window_for_indices(self, indices, reason, padding=None):
        if not indices:
            return

        self._ensure_sampler()
        if self.raster_sampler is None:
            return

        if padding is None:
            padding = self.window_padding

        if all(self._indices_inside_window(index) for index in indices):
            return

        bounds = self._compute_window_bounds(indices, padding)
        self._load_window(bounds, reason)

    def _to_local_indices(self, i, j):
        if self.window_origin is None:
            raise OutsideMapError
        origin_i, origin_j = self.window_origin
        local_i = i - origin_i
        local_j = j - origin_j
        if (
            local_i < 0 or local_j < 0 or
            self.window_shape is None or
            local_i >= self.window_shape[0] or
            local_j >= self.window_shape[1]
        ):
            raise OutsideMapError
        return local_i, local_j

    def _recompute_trace_grid(self, reason):
        if not PROFILE_ENABLED and self.sample is None and self.trace_color_value is None:
            # fast path to avoid work when nothing is ready and profiling off
            self.grid_changed = None
            return

        start_time = time.perf_counter() if PROFILE_ENABLED else None
        diff_duration = None

        if self.sample is None:
            self.grid_changed = None
            state = "no-sample"
        elif self.trace_color_value is None:
            self.grid_changed = None
            state = "cleared"
        else:
            compute_start = time.perf_counter() if PROFILE_ENABLED else None
            r, g, b = self.sample
            r0, g0, b0 = self.trace_color_value
            self.grid_changed = np.abs((r0 - r) ** 2 + (g0 - g) ** 2 +
                                       (b0 - b) ** 2)
            if PROFILE_ENABLED:
                diff_duration = time.perf_counter() - compute_start
            state = "computed"

        if PROFILE_ENABLED:
            total_duration = time.perf_counter() - start_time if start_time is not None else None
            diff_text = f"{diff_duration:.2f}s" if diff_duration is not None else "n/a"
            total_text = f"{total_duration:.2f}s" if total_duration is not None else "n/a"
            grid_changed_bytes = (
                self.grid_changed.nbytes if isinstance(self.grid_changed, np.ndarray) else 0
            )
            QgsMessageLog.logMessage(
                (
                    "[profiling] trace_color_changed "
                    f"state={state} reason={reason} diff={diff_text} "
                    f"total={total_text} grid_changed_mb="
                    f"{(grid_changed_bytes / (1024 ** 2)):.1f}"
                ),
                "RasterTracer",
                Qgis.Info,
            )

    def get_current_vector_layer(self):
        try:
            vlayer = self.iface.layerTreeView().selectedLayers()[0]
            if isinstance(vlayer, QgsVectorLayer):
                if vlayer.wkbType() == QgsWkbTypes.MultiLineString:
                    # if self.last_vlayer:
                    #     if vlayer != self.last_vlayer:
                    #         self.create_spatial_index_for_vlayer(vlayer)
                    # else:
                    #     self.create_spatial_index_for_vlayer(vlayer)
                    # self.last_vlayer = vlayer
                    return vlayer
                else:
                    self.display_message(
                        " ",
                        "The active layer must be" +
                        " a MultiLineString vector layer",
                        level='Warning',
                        duration=2,
                        )
                    return None
            else:
                self.display_message(
                    "Missing Layer",
                    "Please select vector layer to draw",
                    level='Warning',
                    duration=2,
                    )
                return None
        except IndexError:
            self.display_message(
                "Missing Layer",
                "Please select vector layer to draw",
                level='Warning',
                duration=2,
                )
            return None

    def raster_layer_has_changed(self, raster_layer):
        self.rlayer = raster_layer
        if self.rlayer is None:
            self.display_message(
                "Missing Layer",
                "Please select raster layer to trace",
                level='Warning',
                duration=2,
                )
            return

        self.clear_preview()

        total_start = time.perf_counter() if PROFILE_ENABLED else None

        try:
            self.raster_sampler = get_whole_raster(
                self.rlayer,
                QgsProject.instance(),
            )
        except PossiblyIndexedImageError:
            self.display_message(
                "Missing Layer",
                "Can't trace indexed or gray image",
                level='Critical',
                duration=2,
                )
            self.raster_sampler = None
            self.sample = None
            self.grid = None
            self.grid_changed = None
            self.window_origin = None
            self.window_shape = None
            return

        self.to_indexes = self.raster_sampler.to_indexes
        self.to_coords = self.raster_sampler.to_coords
        self.to_coords_provider = self.raster_sampler.to_coords_provider
        self.to_coords_provider2 = self.raster_sampler.to_coords_provider2

        self.sample = None
        self.grid = None
        self.grid_changed = None
        self.window_origin = None
        self.window_shape = None

        self._recompute_trace_grid(reason="raster-change")

        if PROFILE_ENABLED:
            total_duration = time.perf_counter() - total_start if total_start is not None else None
            total_text = f"{total_duration:.2f}s" if total_duration is not None else "n/a"
            raster_size = (
                self.raster_sampler.height if self.raster_sampler else 0,
                self.raster_sampler.width if self.raster_sampler else 0,
            )
            QgsMessageLog.logMessage(
                (
                    "[profiling] raster_layer_has_changed "
                    f"size={raster_size} total={total_text}"
                ),
                "RasterTracer",
                Qgis.Info,
            )

    def remove_last_anchor_point(self, undo_edit=True, redraw=True):
        '''
        Removes last anchor point and last marker point
        '''

        self.clear_preview()

        # check if we have at least one feature to delete
        vlayer = self.get_current_vector_layer()
        if vlayer is None:
            return
        if vlayer.featureCount() < 1:
            return

        # remove last marker
        if self.markers:
            last_marker = self.markers.pop()
            self.canvas().scene().removeItem(last_marker)

        # remove last anchor
        if self.anchors:
            self.anchors.pop()

        if undo_edit:
            # it's a very ugly way of triggering single undo event
            self.iface.editMenu().actions()[0].trigger()

        if redraw:
            self.update_rubber_band()
            self.redraw()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Backspace or e.key() == Qt.Key_B:
            # delete last segment if backspace is pressed
            self.remove_last_anchor_point()
        elif e.key() == Qt.Key_A:
            # change tracing mode
            self.tracing_mode = self.tracing_mode.next()
            self.update_rubber_band()
            if not self.tracing_mode.is_tracing():
                self.clear_preview()
        elif e.key() == Qt.Key_S:
            # toggle snap mode
            self.turn_off_snap()
            self.clear_preview()
        elif e.key() == Qt.Key_Escape:
            # Abort tracing process
            self.abort_tracing_process()

    def add_anchor_points(self, x1, y1, i1, j1):
        '''
        Adds anchor points and markers to self.
        '''

        anchor = Anchor(x1, y1, i1, j1)
        self.anchors.append(anchor)

        marker = QgsVertexMarker(self.canvas())
        marker.setCenter(QgsPointXY(x1, y1))
        self.markers.append(marker)

    def _anchor_indices(self, anchor):
        if hasattr(anchor, 'i'):
            return anchor.i, anchor.j
        return anchor[2], anchor[3]

    def _prepare_pathfinding(self, start, goal, reason):
        self._ensure_window_for_indices([start, goal], reason=reason)

        if self.sample is None or self.grid is None:
            raise OutsideMapError

        try:
            local_start = self._to_local_indices(*start)
            local_goal = self._to_local_indices(*goal)
        except OutsideMapError:
            self._ensure_window_for_indices(
                [start, goal],
                reason=f"{reason}-grow",
                padding=self.window_padding * 2,
            )
            local_start = self._to_local_indices(*start)
            local_goal = self._to_local_indices(*goal)

        r, g, b = self.sample

        try:
            r0 = r[local_goal]
            g0 = g[local_goal]
            b0 = b[local_goal]
        except IndexError:
            raise OutsideMapError

        if self.grid_changed is None:
            grid_to_use = np.abs((r0 - r) ** 2 + (g0 - g) ** 2 + (b0 - b) ** 2)
        else:
            grid_to_use = self.grid_changed

        grid_for_path = grid_to_use.astype(np.dtype('l'))
        origin_i, origin_j = self.window_origin

        return {
            "grid": grid_for_path,
            "local_start": local_start,
            "local_goal": local_goal,
            "origin": (origin_i, origin_j),
        }

    def trace_over_image(self,
                         start,
                         goal,
                         do_it_as_task=False,
                         vlayer=None):
        '''
        performs tracing
        '''

        preparation = self._prepare_pathfinding(start, goal, reason="trace")

        grid_for_path = preparation["grid"]
        local_start = preparation["local_start"]
        local_goal = preparation["local_goal"]
        origin_i, origin_j = preparation["origin"]

        if do_it_as_task:
            # dirty hack to avoid QGIS crashing
            self.find_path_task = FindPathTask(
                grid_for_path,
                local_start,
                local_goal,
                lambda path, layer: self._task_path_callback(
                    path,
                    layer,
                    origin_i,
                    origin_j,
                ),
                vlayer,
                )

            QgsApplication.taskManager().addTask(
                self.find_path_task,
                )
            self.tracking_is_active = True
        else:
            path, cost = FindPathFunction(
                grid_for_path,
                local_start,
                local_goal,
                )
            global_path = [
                (i + origin_i, j + origin_j) for i, j in path
            ]
            return global_path, cost

    def _task_path_callback(self, path, vlayer, origin_i, origin_j):
        if path is not None:
            path = [(i + origin_i, j + origin_j) for i, j in path]
        self.draw_path(path, vlayer)

    def trace(self, x1, y1, i1, j1, vlayer):
        '''
        Traces path from last point to given point.
        In case tracing is inactive just creates
        straight line.
        '''

        if self.tracing_mode.is_tracing():
            if self.snap_tolerance is not None:
                try:
                    i1, j1 = self.snap(i1, j1)
                except OutsideMapError:
                    return

            _, _, i0, j0 = self.anchors[-2]
            start_point = i0, j0
            end_point = i1, j1
            try:
                self.clear_preview()
                self.trace_over_image(start_point,
                                      end_point,
                                      do_it_as_task=True,
                                      vlayer=vlayer)
            except OutsideMapError:
                pass
        else:
            self.draw_path(
                None,
                vlayer,
                was_tracing=False,
                x1=x1,
                y1=y1,
                )

    def snap_to_itself(self, x, y, sq_tolerance=1):
        '''
        finds a nearest segment line to the current vlayer
        '''

        pt = QgsPointXY(x, y)
        # nearest_feature_id = self.spIndex.nearestNeighbor(pt, 1, tolerance)[0]
        vlayer = self.get_current_vector_layer()
        # feature = vlayer.getFeature(nearest_feature_id)
        for feature in vlayer.getFeatures():
            closest_point, _, _, _, sq_distance = feature.geometry().closestVertex(pt)
            if sq_distance < sq_tolerance:
                return closest_point.x(), closest_point.y()
        return x, y

    def snap(self, i, j):
        if self.snap_tolerance is None:
            return i, j
        if not self.tracing_mode.is_tracing():
            return i, j
        if self.grid_changed is None:
            return i, j

        self._ensure_window_for_indices([(i, j)], reason="snap")

        if self.grid_changed is None:
            return i, j

        local_i, local_j = self._to_local_indices(i, j)

        size_i, size_j = self.grid.shape
        size = self.snap_tolerance

        if (
            local_i < size or
            local_j < size or
            local_i + size > size_i or
            local_j + size > size_j
        ):
            raise OutsideMapError

        grid_small = self.grid_changed[
            local_i - size: local_i + size,
            local_j - size: local_j + size,
        ]

        smallest_cells = np.where(grid_small == np.amin(grid_small))
        coordinates = list(zip(smallest_cells[0], smallest_cells[1]))

        offsets = [(ci - size, cj - size) for ci, cj in coordinates]

        if len(offsets) == 1:
            offset_i, offset_j = offsets[0]
        else:
            lengths = [(di ** 2 + dj ** 2) for di, dj in offsets]
            best_index = lengths.index(min(lengths))
            offset_i, offset_j = offsets[best_index]

        return i + offset_i, j + offset_j

    def canvasReleaseEvent(self, mouseEvent):
        '''
        Method where the actual tracing is performed
        after the user clicked on the map
        '''

        vlayer = self.get_current_vector_layer()

        if vlayer is None:
            return

        if not vlayer.isEditable():
            self.display_message(
                "Edit mode",
                "Please begin editing vector layer to trace",
                level='Warning',
                duration=2,
                )
            return

        if self.rlayer is None:
            self.display_message(
                "Missing Layer",
                "Please select raster layer to trace",
                level='Warning',
                duration=2,
                )
            return

        if mouseEvent.button() == Qt.RightButton:
            self.state.click_rmb(mouseEvent, vlayer)
        elif mouseEvent.button() == Qt.LeftButton:
            self.state.click_lmb(mouseEvent, vlayer)

        return

    def draw_path(self, path, vlayer, was_tracing=True,\
                  x1=None, y1=None):
        '''
        Draws a path after tracer found it.
        '''

        self.preview_rubber_band.hide()
        self.preview_pending_request = None
        self.preview_last_request = None

        transform = QgsCoordinateTransform(QgsProject.instance().crs(),
                                           vlayer.crs(),
                                           QgsProject.instance())
        if was_tracing:
            if self.smooth_line:
                path = smooth(path, size=5)
                path = simplify(path)
            vlayer = self.get_current_vector_layer()
            current_last_point = self.to_coords(*path[-1])
            path_ref = [transform.transform(*self.to_coords_provider(i, j)) for i, j in path]
            x0, y0, _, _ = self.anchors[-2]
            last_point = transform.transform(*self.to_coords_provider2(x0, y0))
            path_ref = [last_point] + path_ref[1:]
        else:
            x0, y0, _i, _j = self.anchors[-2]
            current_last_point = (x1, y1)
            path_ref = [transform.transform(*self.to_coords_provider2(x0, y0)),
                        transform.transform(*self.to_coords_provider2(x1, y1))]


        self.ready = False
        if len(self.anchors) == 2:
            vlayer.beginEditCommand("Adding new line")
            add_feature_to_vlayer(vlayer, path_ref)
            vlayer.endEditCommand()
        else:
            vlayer.beginEditCommand("Adding new segment to the line")
            add_to_last_feature(vlayer, path_ref)
            vlayer.endEditCommand()
        _, _, current_last_point_i, current_last_point_j = self.anchors[-1]
        last_x = current_last_point.x() if hasattr(current_last_point, 'x') else current_last_point[0]
        last_y = current_last_point.y() if hasattr(current_last_point, 'y') else current_last_point[1]
        self.anchors[-1] = Anchor(
            last_x,
            last_y,
            current_last_point_i,
            current_last_point_j,
        )
        self.redraw()
        self.tracking_is_active = False


    def update_rubber_band(self):
        # this is very ugly but I can't make another way
        if self.last_mouse_event_pos is None:
            return

        if not self.anchors:
            return

        x0, y0, _, _ = self.anchors[-1]
        qgsPoint = self.toMapCoordinates(self.last_mouse_event_pos)
        x1, y1 = qgsPoint.x(), qgsPoint.y()
        points = [QgsPoint(x0, y0), QgsPoint(x1, y1)]

        self.rubber_band.setColor(QColor(255, 0, 0))
        self.rubber_band.setWidth(3)

        self.rubber_band.setLineStyle(
            RUBBERBAND_LINE_STYLES[self.tracing_mode],
            )

        vlayer = self.get_current_vector_layer()
        if vlayer is None:
            return

        self.rubber_band.setToGeometry(
            QgsGeometry.fromPolyline(points),
            self.vlayer,
            )

    def canvasMoveEvent(self, mouseEvent):
        '''
        Store the mouse position for the correct
        updating of the rubber band
        '''

        # we need at least one point to draw
        if not self.anchors:
            self.clear_preview()
            return

        qgs_point = self.toMapCoordinates(mouseEvent.pos())
        x1, y1 = qgs_point.x(), qgs_point.y()

        preview_goal = None

        if self.tracing_mode.is_tracing() and self.to_indexes is not None:
            try:
                base_i, base_j = self.to_indexes(x1, y1)
            except Exception:  # pylint: disable=broad-except
                base_i = None
                base_j = None

            if base_i is not None and base_j is not None:
                target_i, target_j = base_i, base_j
                if self.snap_tolerance is not None:
                    try:
                        target_i, target_j = self.snap(base_i, base_j)
                    except OutsideMapError:
                        self.clear_preview()
                        return
                    snap_point = self.to_coords(target_i, target_j)
                    if hasattr(snap_point, 'x'):
                        self.marker_snap.setCenter(QgsPointXY(snap_point.x(), snap_point.y()))
                    else:
                        self.marker_snap.setCenter(QgsPointXY(snap_point[0], snap_point[1]))
                preview_goal = (target_i, target_j)
        else:
            self.clear_preview()

        self.last_mouse_event_pos = mouseEvent.pos()
        self.update_rubber_band()
        self.redraw()

        if (self.preview_enabled and self.tracing_mode.is_tracing() and
                preview_goal is not None and len(self.anchors) >= 1):
            start_anchor = self.anchors[-1]
            start = self._anchor_indices(start_anchor)
            self._queue_preview(start, preview_goal)
        elif self.preview_enabled:
            self.clear_preview()

    def abort_tracing_process(self):
        '''
        Terminate background process of tracing raster
        after the user hits Esc.
        '''

        self.clear_preview()

        # check if we have any tasks
        if self.find_path_task is None:
            return

        self.tracking_is_active = False

        try:
            # send terminate signal to the task
            self.find_path_task.cancel()
            self.find_path_task = None
        except RuntimeError:
            return
        else:
            self.remove_last_anchor_point(
                    undo_edit=False,
                    )

    def redraw(self):
        # If caching is enabled, a simple canvas refresh might not be
        # sufficient to trigger a redraw and you must clear the cached image
        # for the layer
        if self.iface.mapCanvas().isCachingEnabled():
            vlayer = self.get_current_vector_layer()
            if vlayer is None:
                return
            vlayer.triggerRepaint()

        self.iface.mapCanvas().refresh()
        QgsApplication.processEvents()

    def pan(self, x, y):
        '''
        Move the canvas to the x, y position
        '''
        currExt = self.iface.mapCanvas().extent()
        canvasCenter = currExt.center()
        dx = x - canvasCenter.x()
        dy = y - canvasCenter.y()
        xMin = currExt.xMinimum() + dx
        xMax = currExt.xMaximum() + dx
        yMin = currExt.yMinimum() + dy
        yMax = currExt.yMaximum() + dy
        newRect = QgsRectangle(xMin, yMin, xMax, yMax)
        self.iface.mapCanvas().setExtent(newRect)

    def add_last_feature_to_spindex(self, vlayer):
        '''
        Adds last feature to spatial index
        '''
        features = list(vlayer.getFeatures())
        last_feature = features[-1]
        self.spIndex.insertFeature(last_feature)

    def create_spatial_index_for_vlayer(self, vlayer):
        '''
        Creates spatial index for the vlayer
        '''

        self.spIndex = QgsSpatialIndex()
        # features = [f for f in vlayer]
        self.spIndex.addFeatures(vlayer.getFeatures())



def add_to_last_feature(vlayer, points):
    '''
    Adds points to the last line feature in the vlayer
    vlayer - QgsLayer of type MultiLine string
    points - list of points
    '''
    features = list(vlayer.getFeatures())
    last_feature = features[-1]
    fid = last_feature.id()
    geom = last_feature.geometry()
    new_points = [QgsPointXY(x, y) for x, y in points]

    if geom.isMultipart():
        multiline = geom.asMultiPolyline()
        if not multiline:
            multiline = [[]]
        multiline[-1].extend(new_points)
        new_geom = QgsGeometry.fromMultiPolylineXY(multiline)
    else:
        polyline = geom.asPolyline()
        if polyline is None:
            polyline = []
        polyline.extend(new_points)
        new_geom = QgsGeometry.fromPolylineXY(polyline)

    vlayer.changeGeometry(fid, new_geom)


def add_feature_to_vlayer(vlayer, points):
    '''
    Adds new line feature to the vlayer
    '''

    feat = QgsFeature(vlayer.fields())
    polyline = [QgsPoint(x, y) for x, y in points]
    feat.setGeometry(QgsGeometry.fromPolyline(polyline))
    vlayer.addFeature(feat)

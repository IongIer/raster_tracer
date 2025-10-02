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
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis
from qgis.core import QgsCoordinateTransform


from .astar import FindPathFunction
from .line_simplification import smooth, simplify
from .pointtool_states import WaitingFirstPointState
from .exceptions import OutsideMapError
from .pointtool_preview import TracePreviewController
from .pointtool_raster import RasterTracingContext
from .pointtool_tasks import TraceTaskController

# An point on the map where the user clicked along the line
Anchor = namedtuple('Anchor', ['x', 'y', 'i', 'j'])

# Flag for experimental Autofollowing mode
ALLOW_AUTO_FOLLOWING = False

SHORTCUT_KEYS = {
    Qt.Key_A,
    Qt.Key_B,
    Qt.Key_Backspace,
    Qt.Key_S,
    Qt.Key_Escape,
    Qt.Key_T,
}

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

    def __init__(
            self,
            canvas,
            iface,
            turn_off_snap,
            smooth=False,
            ensure_trace_color_enabled=None,
            set_trace_color=None):
        '''
        canvas - link to the QgsCanvas of the application
        iface - link to the Qgis Interface
        turn_off_snap - flag sets snapping to the nearest color
        smooth - flag sets smoothing of the traced path
        ensure_trace_color_enabled - callback enabling trace-color mode in UI
        set_trace_color - callback syncing sampled color back to UI control
        '''

        self.iface = iface

        # list of Anchors for current line
        self.anchors = []

        # for keeping track of mouse event for rubber band updating
        self.last_mouse_event_pos = None

        self.tracing_mode = TracingModes.PATH

        self.turn_off_snap = turn_off_snap
        self.smooth_line = smooth
        self._enable_trace_color_cb = ensure_trace_color_enabled
        self._set_trace_color_cb = set_trace_color

        # possible variants: gray_diff, as_is, color_diff (using v from hsv)
        self.grid_conversion = "gray_diff"

        # QApplication.restoreOverrideCursor()
        # QApplication.setOverrideCursor(Qt.CrossCursor)
        QgsMapToolEmitPoint.__init__(self, canvas)

        self.rlayer = None
        self.snap_tolerance = None # snap to color
        self.snap2_tolerance = None # snap to itself
        self.vlayer = None
        self.to_indexes = None
        self.to_coords = None
        self.to_coords_provider = None
        self.to_coords_provider2 = None
        self.trace_color_value = None
        self.preview_controller = TracePreviewController(self)
        self.task_controller = TraceTaskController(self)
        self.raster_context = RasterTracingContext(self)

        self.tracking_is_active = False

        # False = not a polygon
        self.rubber_band = QgsRubberBand(self.canvas(), QgsWkbTypes.LineGeometry)
        self.markers = []
        self.marker_snap = QgsVertexMarker(self.canvas())
        self.marker_snap.setColor(QColor(255, 0, 255))

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
        if snap_tolerance is None:
            self.snap2_tolerance = None
            return

        try:
            snap_value = float(snap_tolerance)
        except (TypeError, ValueError):
            return

        self.snap2_tolerance = snap_value ** 2
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
        self.preview_controller.restart_last_request()

    def set_preview_enabled(self, enabled):
        self.preview_controller.set_enabled(enabled)

    def set_preview_color(self, color):
        self.preview_controller.set_color(color)

    def set_preview_width(self, width):
        self.preview_controller.set_width(width)

    def clear_preview(self):
        self.preview_controller.clear()

    def _take_preview_path_if_valid(self, start_point, end_point, click_pos):
        return self.preview_controller.take_path_if_valid(start_point, end_point, click_pos)

    def _has_inflight_preview_for(self, start_point, end_point):
        return self.preview_controller.has_inflight_preview_for(start_point, end_point)

    def _ensure_preview_inflight_started(self):
        self.preview_controller.ensure_inflight_started()

    def _queue_preview(self, start, goal, screen_pos=None):
        self.preview_controller.queue(start, goal, screen_pos)

    def _ensure_sampler(self):
        self.raster_context.ensure_sampler()

    def _ensure_window_for_indices(self, indices, reason, padding=None):
        self.raster_context.ensure_window_for_indices(indices, reason, padding)

    def _to_local_indices(self, i, j):
        return self.raster_context.to_local_indices(i, j)

    def _recompute_trace_grid(self, reason):
        self.raster_context.recompute_trace_grid(reason)

    def _prepare_pathfinding(self, start, goal, reason):
        return self.raster_context.prepare_pathfinding(start, goal, reason)

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
            self.raster_context.set_sampler_for_layer(None)
            return

        self.clear_preview()

        total_start = time.perf_counter() if PROFILE_ENABLED else None

        loaded = self.raster_context.set_sampler_for_layer(self.rlayer)
        if not loaded:
            return

        if PROFILE_ENABLED:
            total_duration = time.perf_counter() - total_start if total_start is not None else None
            total_text = f"{total_duration:.2f}s" if total_duration is not None else "n/a"
            sampler = self.raster_context.raster_sampler
            raster_size = (
                sampler.height if sampler else 0,
                sampler.width if sampler else 0,
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
        elif e.key() == Qt.Key_T:
            self._handle_trace_color_shortcut()

    def add_anchor_points(self, x1, y1, i1, j1):
        '''
        Adds anchor points and markers to self.
        '''

        anchor = Anchor(x1, y1, i1, j1)
        self.anchors.append(anchor)

        marker = QgsVertexMarker(self.canvas())
        marker.setCenter(QgsPointXY(x1, y1))
        self.markers.append(marker)

    def handled_shortcut_keys(self):
        return SHORTCUT_KEYS

    def _handle_trace_color_shortcut(self):
        if self.to_indexes is None:
            QgsMessageLog.logMessage(
                "[shortcut] Ignoring 'T' – no raster selected",
                "RasterTracer",
                Qgis.Info,
            )
            return

        if self.last_mouse_event_pos is None:
            return

        qgs_point = self.toMapCoordinates(self.last_mouse_event_pos)
        x, y = qgs_point.x(), qgs_point.y()

        try:
            i, j = self.to_indexes(x, y)
        except Exception:  # pylint: disable=broad-except
            QgsMessageLog.logMessage(
                "[shortcut] Ignoring 'T' – point outside raster extent",
                "RasterTracer",
                Qgis.Info,
            )
            return

        color = self._sample_color_at_indices(i, j)
        if color is None:
            return

        if self._enable_trace_color_cb is not None:
            self._enable_trace_color_cb()

        if self._set_trace_color_cb is not None:
            self._set_trace_color_cb(color)
        else:
            self.trace_color_changed(color)

        if not self.tracing_mode.is_tracing():
            self.tracing_mode = TracingModes.PATH
            self.update_rubber_band()

    def _sample_color_at_indices(self, i, j):
        return self.raster_context.sample_color_at_indices(i, j)

    def has_active_trace(self):
        return bool(self.anchors) or self.tracking_is_active

    def _anchor_indices(self, anchor):
        if hasattr(anchor, 'i'):
            return anchor.i, anchor.j
        return anchor[2], anchor[3]

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
            def callback(path, layer):
                self._task_path_callback(
                    path,
                    layer,
                    origin_i,
                    origin_j,
                )

            self.task_controller.start(
                grid_for_path,
                local_start,
                local_goal,
                callback,
                vlayer,
            )
            self.tracking_is_active = True
        else:
            path, cost = FindPathFunction(
                grid_for_path,
                local_start,
                local_goal,
                )
            if path is None or cost is None:
                return None, None
            global_path = [
                (i + origin_i, j + origin_j) for i, j in path
            ]
            return global_path, cost

    def _task_path_callback(self, path, vlayer, origin_i, origin_j):
        if path is None:
            self.tracking_is_active = False
            self.preview_controller.clear()
            self.display_message(
                "No Path",
                "Unable to find a path between the selected points.",
                level='Warning',
                duration=2,
            )
            return

        path = [(i + origin_i, j + origin_j) for i, j in path]
        self.draw_path(path, vlayer)

    def trace(self, x1, y1, i1, j1, vlayer, click_pos=None):
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
            start_point = (int(i0), int(j0))
            end_point = (int(i1), int(j1))

            preview_path = self._take_preview_path_if_valid(start_point, end_point, click_pos)
            if preview_path is not None:
                self.tracking_is_active = True
                self.preview_controller.clear()
                self.draw_path(preview_path, vlayer, was_tracing=True)
                return

            if (
                self.preview_controller.enabled and
                self._has_inflight_preview_for(start_point, end_point)
            ):
                commit_request = {
                    "start": start_point,
                    "goal": end_point,
                }
                self.preview_controller.request_commit(commit_request, vlayer)
                self.tracking_is_active = True
                self._ensure_preview_inflight_started()
                return

            self.preview_controller.clear_commit()
            try:
                self.clear_preview()
                self.trace_over_image(start_point,
                                      end_point,
                                      do_it_as_task=True,
                                      vlayer=vlayer)
            except OutsideMapError:
                pass
        else:
            self.preview_controller.clear_commit()
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
        if self.raster_context.grid_changed is None:
            return i, j

        self._ensure_window_for_indices([(i, j)], reason="snap")

        if self.raster_context.grid_changed is None:
            return i, j

        local_i, local_j = self._to_local_indices(i, j)

        grid = self.raster_context.grid
        if grid is None:
            return i, j
        size_i, size_j = grid.shape
        size = self.snap_tolerance

        if (
            local_i < size or
            local_j < size or
            local_i + size > size_i or
            local_j + size > size_j
        ):
            raise OutsideMapError

        grid_small = self.raster_context.grid_changed[
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

        self.preview_controller.clear()

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

        if (self.preview_controller.enabled and self.tracing_mode.is_tracing() and
                preview_goal is not None and len(self.anchors) >= 1):
            start_anchor = self.anchors[-1]
            start = self._anchor_indices(start_anchor)
            self._queue_preview(start, preview_goal, mouseEvent.pos())
        elif self.preview_controller.enabled:
            self.clear_preview()

    def abort_tracing_process(self):
        '''
        Terminate background process of tracing raster
        after the user hits Esc.
        '''

        self.clear_preview()

        # check if we have any tasks
        if not self.task_controller.active:
            return

        self.tracking_is_active = False

        if self.task_controller.cancel():
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

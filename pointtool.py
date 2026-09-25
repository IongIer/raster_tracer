"""GUI input, context validation and transactional geometry edits."""

import math
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from itertools import chain

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsFeatureRequest,
    QgsGeometry,
    QgsLineString,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsVectorLayer,
    QgsVectorLayerUtils,
    QgsVertexId,
)
from qgis.gui import QgsMapToolEdit, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import QCoreApplication, QPoint, Qt
from qgis.PyQt.QtGui import QColor

from .exceptions import OutsideMapError
from .pointtool_preview import TracePreviewController
from .pointtool_raster import RasterTracingContext
from .pointtool_session import (
    ResolvedEndpoint,
    TraceContext,
    TraceRequest,
    TraceResult,
    TraceSession,
    as_xy,
    new_id,
)
from .pointtool_states import WaitingFirstPointState, WaitingMiddlePointState
from .pointtool_tasks import TraceTaskController, execute_request
from .utils import get_coords_from_raster_indxs, get_indxs_from_raster_coords

DENSE_LINE_SPACING = 5.0  # Preserve existing spacing in vector layer units.
SHORTCUT_KEYS = {
    Qt.Key.Key_A,
    Qt.Key.Key_B,
    Qt.Key.Key_S,
    Qt.Key.Key_Escape,
    Qt.Key.Key_T,
    Qt.Key.Key_D,
}


class TracingModes(Enum):
    LINE = 1
    PATH = 2
    DENSE_LINE = 4

    def is_tracing(self):
        return self == TracingModes.PATH


def append_path_points(existing, points):
    """Extend an owned line without converting its existing vertices in Python."""
    if existing.wkbType() == Qgis.WkbType.MultiLineString:
        geometry = QgsGeometry(existing)
        multi = geometry.get()  # Detach before modifying the shared geometry.
        if multi.numGeometries() == 0:
            raise ValueError("Target line is empty")
        line = multi.lineStringN(multi.numGeometries() - 1)
        if line.numPoints() == 0:
            raise ValueError("Target line is empty")
        seam = QgsPointXY(line.endPoint())
        tail = points[1:] if seam == points[0] else points
        # Native append may replace its shared endpoint. Anchor it at the exact
        # old coordinate, including when QgsPointXY equality is only approximate.
        line.append(QgsLineString([seam, *tail]))
        return geometry
    # Retain the existing curve segmentization and Z/M conversion behavior.
    lines = existing.asMultiPolyline()
    if not lines or not lines[-1]:
        raise ValueError("Target line is empty")
    lines[-1].extend(points[1:] if lines[-1][-1] == points[0] else points)
    return QgsGeometry.fromMultiPolylineXY(lines)


def truncate_path_points(existing, count):
    """Drop an appended segment without retaining a copy of every prefix."""
    geometry = QgsGeometry(existing)
    multi = geometry.get()
    line = multi.lineStringN(multi.numGeometries() - 1)
    if not 2 <= count <= line.numPoints():
        raise ValueError("Invalid segment boundary")
    for index in range(line.numPoints() - 1, count - 1, -1):
        line.deleteVertex(QgsVertexId(0, 0, index))
    return geometry


class RasterScribePointTool(QgsMapToolEdit):
    def __init__(
        self,
        canvas,
        iface,
        turn_off_snap,
        smooth=False,
        ensure_trace_color_enabled=None,
        set_trace_color=None,
        scheduler=None,
    ):
        super().__init__(canvas)
        self.iface = iface
        self.session = TraceSession()
        self.owner = new_id()
        self.disposed = False
        self.suspended = True
        self._own_edit = False
        self._finishing = False
        self._connections = []
        self._raster_connections = []
        self._target_connections = []
        self._context = None
        self.rlayer = None
        self.to_indexes = self.to_coords = None
        self.last_mouse_event_pos = None
        self._tracing_mode = TracingModes.PATH
        self._smooth_line = bool(smooth)
        self.snap_tolerance = self.snap2_tolerance = None
        self.trace_color_value = None
        self.turn_off_snap = turn_off_snap
        self._enable_trace_color_cb = ensure_trace_color_enabled
        self._set_trace_color_cb = set_trace_color
        self.task_controller = scheduler or TraceTaskController()
        self.raster_context = RasterTracingContext(self)
        self.rubber_band = QgsRubberBand(canvas, Qgis.GeometryType.Line)
        self.draft_band = self.createRubberBand(Qgis.GeometryType.Line)
        self.draft_band.hide()
        self.marker_snap = QgsVertexMarker(canvas)
        self.marker_snap.setColor(QColor(255, 0, 255))
        self.marker_snap.hide()
        self.markers = []
        self._pending_markers = {}
        self.preview_controller = TracePreviewController(self)
        self._sync_state()
        project = QgsProject.instance()
        self._connect(
            canvas, "destinationCrsChanged", self._context_changed, self._connections
        )
        self._connect(project, "crsChanged", self._context_changed, self._connections)
        self._connect(
            project, "transformContextChanged", self._context_changed, self._connections
        )
        self._connect(
            project, "layersWillBeRemoved", self._layers_removed, self._connections
        )
        self._connect(
            iface.layerTreeView(),
            "currentLayerChanged",
            self._target_changed,
            self._connections,
        )

    @staticmethod
    def _connect(obj, name, callback, collection):
        signal = getattr(obj, name, None)
        if signal is not None:
            signal.connect(callback)
            collection.append((signal, callback))

    @staticmethod
    def _disconnect(collection):
        while collection:
            signal, callback = collection.pop()
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass

    @property
    def anchors(self):
        return tuple(self.session.anchors)

    @property
    def tracking_is_active(self):
        return self.session.pending is not None

    @property
    def smooth_line(self):
        return self._smooth_line

    @smooth_line.setter
    def smooth_line(self, value):
        if bool(value) != self._smooth_line:
            self._smooth_line = bool(value)
            self._semantic_changed()

    @property
    def tracing_mode(self):
        return self._tracing_mode

    @tracing_mode.setter
    def tracing_mode(self, value):
        if value != self._tracing_mode:
            self._tracing_mode = value
            self._semantic_changed()

    def activate(self):
        if self.disposed:
            return
        self.suspended = False
        # A suspended tool can be reused after project changes.
        self.raster_context.set_sampler_for_layer(self.rlayer)
        super().activate()

    def deactivate(self):
        self.suspend()
        super().deactivate()  # QgsMapTool emits deactivated exactly once.

    def suspend(self):
        if self.disposed:
            return True
        self.suspended = True
        finished = self.finish_session()
        self.last_mouse_event_pos = None
        self.marker_snap.hide()
        return finished

    def dispose(self):
        if self.disposed:
            return True
        if not self.suspend():
            return False
        self.disposed = True
        self._disconnect(self._connections)
        self._disconnect(self._raster_connections)
        self._disconnect(self._target_connections)
        self.preview_controller.dispose()
        for item in (self.rubber_band, self.draft_band, self.marker_snap):
            self.canvas().scene().removeItem(item)
        self.raster_context.reset()
        self.rlayer = None
        self.turn_off_snap = self._enable_trace_color_cb = self._set_trace_color_cb = (
            None
        )
        return True

    def _sync_state(self):
        cls = (
            WaitingMiddlePointState if self.session.anchors else WaitingFirstPointState
        )
        self.state = cls(self)

    def _remove_pending_markers(self):
        for marker in self._pending_markers.values():
            self.canvas().scene().removeItem(marker)
        self._pending_markers.clear()

    def _cancel_inflight_segment(self, *, evict=True):
        self.session.invalidate()  # Revoke identity before requesting cancellation.
        self.task_controller.cancel(self.owner, evict=evict)
        self.preview_controller.clear(cancel=False)
        self._remove_pending_markers()
        self.update_rubber_band()

    def _reset_session(self):
        self.session.reset()
        self.task_controller.cancel(self.owner)
        self.preview_controller.clear(cancel=False)
        self._remove_pending_markers()
        while self.markers:
            self.canvas().scene().removeItem(self.markers.pop())
        self.rubber_band.hide()
        self.draft_band.reset(Qgis.GeometryType.Line)
        self.marker_snap.hide()
        self._context = None
        self._disconnect(self._target_connections)
        self._sync_state()

    def finish_session(self, *args):
        """Publish the completed draft in one short, independently undoable edit."""
        if self._finishing:
            return False
        self._cancel_inflight_segment()
        geometry = self.session.geometry
        if geometry is None:
            self._reset_session()
            return True
        layer = QgsProject.instance().mapLayer(self.session.target_id)
        command_open = False
        self._finishing = True
        try:
            if layer is None or not layer.isEditable():
                raise ValueError("The draft's target layer is not editable")
            if layer.isEditCommandActive():
                raise ValueError(
                    "Finish the other layer edit before finishing the trace"
                )
            with self._editing():
                feature = QgsVectorLayerUtils.createFeature(layer, geometry)
                layer.beginEditCommand(self.tr("Trace raster line"))
                command_open = True
                if not layer.addFeature(feature):
                    raise RuntimeError(
                        "Geometry write failed; the draft is still available"
                    )
                layer.endEditCommand()
                command_open = False
            self._reset_session()
            layer.triggerRepaint()
            return True
        except Exception as error:
            if command_open:
                with self._editing():
                    layer.destroyEditCommand()
            self.report_failure("error", str(error))
            return False
        finally:
            self._finishing = False

    def _discard_session(self, *args):
        self._reset_session()

    def _before_modified_check(self):
        # Buffered-group saves query dirty layers without beforeCommitChanges.
        # This is QGIS's hook for including last-minute edits in that query.
        if (
            not self._own_edit
            and not self._finishing
            and QgsProject.instance().transactionMode()
            == Qgis.TransactionMode.BufferedGroups
        ):
            self.finish_session()

    def _semantic_changed(self):
        if self.disposed:
            return
        hover = self.last_mouse_event_pos
        self._cancel_inflight_segment(evict=False)
        if hover is not None and self.anchors and not self.suspended:
            self._hover(hover)

    def _context_changed(self, *args):
        if self.disposed:
            return
        if self.finish_session():
            self.raster_context.set_sampler_for_layer(self.rlayer)

    def _target_changed(self, *args):
        if self.disposed or not self.session.target_id:
            return
        layer = self.get_current_vector_layer()
        if layer is None or layer.id() != self.session.target_id:
            self.finish_session()

    def _external_edit(self, *args):
        if not self.disposed and not self._own_edit:
            # The draft is independent of layer edits. Do not publish from a
            # signal emitted inside QUndoStack.push/undo/redo.
            self._cancel_inflight_segment()

    def _layers_removed(self, ids):
        if self.disposed:
            return
        if self.rlayer is not None and self.rlayer.id() in ids:
            self.raster_layer_has_changed(None)
        elif self.session.target_id in ids:
            self.finish_session()

    def raster_layer_has_changed(self, layer):
        if self.disposed:
            return
        if not self.finish_session():
            return
        self._disconnect(self._raster_connections)
        self.rlayer = layer
        self.raster_context.set_sampler_for_layer(layer)
        if layer is not None:
            for signal in ("crsChanged", "dataChanged", "isValidChanged"):
                self._connect(
                    layer, signal, self._context_changed, self._raster_connections
                )

    def _observe_target(self, layer):
        self._disconnect(self._target_connections)
        for signal in (
            "geometryChanged",
            "featureAdded",
            "featureDeleted",
            "attributeValueChanged",
            "updatedFields",
        ):
            self._connect(layer, signal, self._external_edit, self._target_connections)
        for signal in ("beforeCommitChanges", "editingStopped"):
            self._connect(layer, signal, self.finish_session, self._target_connections)
        self._connect(
            layer, "beforeRollBack", self._discard_session, self._target_connections
        )
        self._connect(
            layer,
            "beforeModifiedCheck",
            self._before_modified_check,
            self._target_connections,
        )
        self._connect(
            layer, "crsChanged", self._context_changed, self._target_connections
        )
        self._connect(
            layer.undoStack(),
            "indexChanged",
            self._external_edit,
            self._target_connections,
        )

    def get_current_vector_layer(self):
        layer = self.iface.activeLayer()
        if (
            isinstance(layer, QgsVectorLayer)
            and layer.isValid()
            and layer.wkbType()
            in (Qgis.WkbType.MultiLineString, Qgis.WkbType.MultiCurve)
        ):
            return layer
        return None

    def _capture_context(self, layer):
        sampler = self.raster_context.raster_sampler
        if sampler is None:
            raise OutsideMapError("No RGB raster selected")
        working = QgsCoordinateReferenceSystem(
            self.canvas().mapSettings().destinationCrs()
        )
        vector = QgsCoordinateReferenceSystem(layer.crs())
        raster = QgsCoordinateReferenceSystem(self.rlayer.crs())
        transform_context = QgsCoordinateTransformContext(
            QgsProject.instance().transformContext()
        )
        return TraceContext(
            self.rlayer.id(),
            sampler.source,
            raster,
            working,
            vector,
            transform_context,
            sampler.geo_ref,
            QgsCoordinateTransform(raster, working, transform_context),
            QgsCoordinateTransform(working, vector, transform_context),
        )

    def _valid_request(self, request):
        # These primitive guards must precede access to possibly deleted Qt objects.
        if (
            self.disposed
            or self.suspended
            or request.owner != self.owner
            or not self.session.matches(request)
        ):
            return False
        project = QgsProject.instance()
        layer = project.mapLayer(request.target_id)
        raster = project.mapLayer(request.context.raster_id)
        if (
            layer is None
            or raster is None
            or not layer.isEditable()
            or not raster.isValid()
        ):
            return False
        if layer is not self.get_current_vector_layer():
            return False
        sampler = self.raster_context.raster_sampler
        ctx = request.context
        if (
            sampler is None
            or sampler.source != ctx.source
            or raster.source() != ctx.source.path
            or layer.crs() != ctx.vector_crs
            or raster.crs() != ctx.raster_crs
            or self.canvas().mapSettings().destinationCrs() != ctx.working_crs
            or project.transformContext() != ctx.transform_context
        ):
            return False
        return True

    def resolve_endpoint(self, point):
        x, y = as_xy(point)
        if not all(math.isfinite(v) for v in (x, y)) or self.to_indexes is None:
            raise OutsideMapError("No valid raster endpoint")

        def indexes(x, y):
            if self.tracing_mode.is_tracing():
                return self.to_indexes(x, y)
            sampler = self.raster_context.raster_sampler
            return get_indxs_from_raster_coords(
                sampler.geo_ref, sampler.trfm_to_src.transform(x, y)
            )

        i, j = indexes(x, y)
        if self.tracing_mode.is_tracing():
            if self.snap_tolerance is not None and self.trace_color_value is not None:
                i, j = self.snap(i, j)
                x, y = as_xy(self.to_coords(i, j))
        if self.snap2_tolerance is not None:
            x, y = self.snap_to_itself(x, y, self.snap2_tolerance)
        i, j = indexes(x, y)
        return ResolvedEndpoint(x, y, i, j)

    def snap(self, i, j):
        return self.raster_context.snap(
            i, j, self.snap_tolerance, self.trace_color_value
        )

    def snap_to_itself(self, x, y, tolerance=1):
        """Nearest vertex in canvas map units, including the tolerance boundary.

        Ties prefer the lowest feature ID, then the last vertex in its geometry.
        """
        layer = self.get_current_vector_layer()
        if (
            layer is None
            or tolerance is None
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            return x, y
        working = self.canvas().mapSettings().destinationCrs()
        context = QgsProject.instance().transformContext()
        try:
            to_layer = QgsCoordinateTransform(working, layer.crs(), context)
            to_map = QgsCoordinateTransform(layer.crs(), working, context)
            rectangle = QgsRectangle(
                x - tolerance, y - tolerance, x + tolerance, y + tolerance
            )
            rectangle = to_layer.transformBoundingBox(rectangle)
            if not all(
                math.isfinite(v)
                for v in (
                    rectangle.xMinimum(),
                    rectangle.xMaximum(),
                    rectangle.yMinimum(),
                    rectangle.yMaximum(),
                )
            ):
                return x, y
            request = (
                QgsFeatureRequest().setSubsetOfAttributes([]).setFilterRect(rectangle)
            )
            target = QgsPointXY(x, y)
            threshold = tolerance**2
            best = None
            candidates = ((f.id(), f.geometry()) for f in layer.getFeatures(request))
            draft = self.session.geometry
            if (
                draft is not None
                and self.session.target_id == layer.id()
                and draft.boundingBox().intersects(rectangle)
            ):
                # A draft behaves like the newest feature for equal distances.
                candidates = chain(candidates, ((-math.inf, draft),))
            for feature_id, geometry in candidates:
                if not to_map.isShortCircuited():
                    # Compare in map units without changing the layer or draft.
                    # Layer-space nearest vertices can differ after projection.
                    geometry = QgsGeometry(geometry)
                    if (
                        geometry.transform(to_map)
                        != Qgis.GeometryOperationResult.Success
                    ):
                        continue
                vertex, index, _, _, distance = geometry.closestVertex(target)
                if index >= 0 and 0 <= distance <= threshold:
                    key = (distance, feature_id, -index)
                    if best is None or key < best[0]:
                        best = key, as_xy(vertex)
            return best[1] if best is not None else (x, y)
        except Exception:
            return x, y

    def snap_tolerance_changed(self, value):
        if value is not None:
            value = int(value)
            if not 0 <= value <= 99:
                raise ValueError("Color snap radius must be between 0 and 99")
        if value != self.snap_tolerance:
            self.snap_tolerance = value
            self._semantic_changed()

    def snap2_tolerance_changed(self, value):
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                return
        if value != self.snap2_tolerance:
            self.snap2_tolerance = value
            self._semantic_changed()

    def trace_color_changed(self, color):
        value = None if color is False else tuple(float(v) for v in color.getRgb()[:3])
        if value != self.trace_color_value:
            self.trace_color_value = value
            self._semantic_changed()

    def set_preview_enabled(self, value):
        self.preview_controller.set_enabled(value)

    def set_preview_color(self, color):
        self.preview_controller.set_color(color)

    def set_preview_width(self, width):
        self.preview_controller.set_width(width)

    def clear_preview(self):
        pending = self.session.pending
        if pending is not None and pending.adopted_preview:
            self._cancel_inflight_segment()
        else:
            self.preview_controller.clear()

    def make_request(self, endpoint, screen_pos=None):
        if not self.anchors or self._context is None:
            raise OutsideMapError("No initial anchor")
        start = self.anchors[-1]
        bounds = (
            self.raster_context.compute_window_bounds([start.pixel, endpoint.pixel])
            if self.tracing_mode.is_tracing()
            else ()
        )
        return TraceRequest(
            new_id(),
            self.owner,
            self.session.session_id,
            self.session.revision,
            self._context,
            start,
            endpoint,
            self.session.target_id,
            self.tracing_mode.name,
            self.smooth_line,
            self.trace_color_value,
            bounds,
            as_xy(screen_pos) if screen_pos is not None else None,
        )

    def accept_click(self, point, screen_pos=None):
        if self.disposed or self.suspended or self.tracking_is_active:
            return
        self._target_changed()
        layer = self.get_current_vector_layer()
        if layer is None or not layer.isEditable():
            self.report_failure("invalid_input")
            return
        if self.anchors and self.session.target_id != layer.id():
            return  # A failed publication must keep its original draft/context.
        try:
            endpoint = self.resolve_endpoint(point)
            if not self.anchors:
                self._context = self._capture_context(layer)
                self.session.bind(layer.id(), endpoint)
                self._observe_target(layer)
                self.markers.append(self._marker(endpoint))
                self._sync_state()
                return
            if endpoint.xy == self.anchors[-1].xy:
                return
            request = self.make_request(endpoint, screen_pos)
            if not self._valid_request(request):
                self.finish_session()
                return
            cached = self.preview_controller.matching_cached(request)
            adopted = self.preview_controller.matching_request(request)
            if adopted is not None and cached is None:
                request = replace(request, request_id=adopted.request_id)
            self.session.begin(
                request, adopted=adopted is not None or cached is not None
            )
            self._pending_markers[request.request_id] = self._marker(endpoint)
            if request.mode != "PATH":
                self._on_result(request, TraceResult(request.request_id, "success"))
            elif cached is not None:
                self._on_result(request, replace(cached, request_id=request.request_id))
            elif adopted is not None:
                self.preview_controller.start_adopted(request.request_id)
            else:
                self.preview_controller.clear()
                if not self.task_controller.submit(
                    request, self._on_result, committed=True
                ):
                    self._finish_pending(request.request_id)
        except Exception as error:
            if self.session.pending is not None:
                self._finish_pending(self.session.pending.request.request_id)
            self.report_failure("invalid_input", str(error))

    def _marker(self, endpoint):
        marker = QgsVertexMarker(self.canvas())
        marker.setCenter(QgsPointXY(*endpoint.xy))
        return marker

    def _finish_pending(self, request_id):
        if not self.session.finish(request_id):
            return
        marker = self._pending_markers.pop(request_id, None)
        if marker is not None:
            self.canvas().scene().removeItem(marker)
        self.preview_controller.clear(cancel=False)
        self.update_rubber_band()

    def _on_result(self, request, result):
        if result.request_id != request.request_id:
            return
        if not self._valid_request(request):
            if (
                not self.disposed
                and not self.suspended
                and self.session.matches(request)
            ):
                self.finish_session()
            return
        pending = self.session.pending
        if pending is not None and pending.request.request_id == result.request_id:
            request = pending.request  # Accepted click's exact map endpoints.
            if result.status == "success":
                self._commit(request, result)
            else:
                self._finish_pending(result.request_id)
                if result.status != "cancelled":
                    self.report_failure(result.status, result.diagnostics)
        elif pending is None:
            self.preview_controller.receive(request, result)

    def build_path_points(self, request, result, *, in_layer_crs):
        """Share path/endpoints, converting only to the caller's required CRS."""
        ctx = request.context
        if request.mode == "PATH":
            top, left = result.origin
            points = [
                as_xy(
                    ctx.raster_to_map.transform(
                        *get_coords_from_raster_indxs(ctx.geo_ref, (i + top, j + left))
                    )
                )
                for i, j in result.processed
            ]
            if not points:
                raise ValueError("Empty path")
            if len(points) == 1:
                points.append(points[0])
            points[0], points[-1] = request.start.xy, request.goal.xy
        else:
            points = [request.start.xy, request.goal.xy]
        if not all(math.isfinite(v) for point in points for v in point):
            raise ValueError("Nonfinite transformed geometry")
        if in_layer_crs:
            points = [as_xy(ctx.map_to_vector.transform(*p)) for p in points]
            if request.mode == "DENSE_LINE":
                a, b = points
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                fractions = (
                    [0.5]
                    if 0 < length < DENSE_LINE_SPACING
                    else [
                        step * DENSE_LINE_SPACING / length
                        for step in range(1, int(length // DENSE_LINE_SPACING) + 1)
                        if step * DENSE_LINE_SPACING < length
                    ]
                )
                points = (
                    [a]
                    + [
                        (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                        for f in fractions
                    ]
                    + [b]
                )
            if not all(math.isfinite(v) for point in points for v in point):
                raise ValueError("Nonfinite transformed geometry")
        return [QgsPointXY(*p) for p in points]

    @contextmanager
    def _editing(self):
        previous = self._own_edit
        self._own_edit = True
        try:
            yield
        finally:
            self._own_edit = previous

    def _commit(self, request, result):
        try:
            if not self._valid_request(request):
                self._finish_pending(request.request_id)
                return
            points = self.build_path_points(request, result, in_layer_crs=True)
            previous = self.session.geometry
            if previous is None:
                previous_vertices = 0
                geometry = QgsGeometry.fromMultiPolylineXY([points])
            else:
                previous_vertices = previous.constGet().nCoordinates()
                geometry = append_path_points(previous, points)
            # Validate again after transforms before replacing the owned draft.
            if not self._valid_request(request):
                self._finish_pending(request.request_id)
                return
            self.session.accept(request, geometry, previous_vertices)
            self.markers.append(self._pending_markers.pop(request.request_id))
            self.preview_controller.clear()
            self._show_draft()
            self.update_rubber_band()
        except Exception as error:
            self._finish_pending(request.request_id)
            self.report_failure("error", str(error))

    def _show_draft(self):
        geometry = self.session.geometry
        if geometry is None:
            self.draft_band.reset(Qgis.GeometryType.Line)
            return
        layer = QgsProject.instance().mapLayer(self.session.target_id)
        self.draft_band.setToGeometry(geometry, layer)
        self.draft_band.show()

    def remove_last_anchor_point(self):
        if self.tracking_is_active:
            self._cancel_inflight_segment()
            return
        if not self.anchors:
            return
        if not self.session.segment_vertices:
            self._discard_session()
            return
        self._cancel_inflight_segment()
        count = self.session.segment_vertices[-1]
        geometry = truncate_path_points(self.session.geometry, count) if count else None
        self.session.undo(geometry)
        self.canvas().scene().removeItem(self.markers.pop())
        self._sync_state()
        self._show_draft()
        self.update_rubber_band()

    def keyPressEvent(self, event):
        if self.disposed or self.suspended:
            return
        key = event.key()
        if key == Qt.Key.Key_B:
            self.remove_last_anchor_point()
        elif key == Qt.Key.Key_Escape:
            self._cancel_inflight_segment()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_D):
            mode = TracingModes.LINE if key == Qt.Key.Key_A else TracingModes.DENSE_LINE
            self.tracing_mode = TracingModes.PATH if self.tracing_mode == mode else mode
        elif key == Qt.Key.Key_S:
            self.turn_off_snap()
        elif key == Qt.Key.Key_T:
            self._handle_trace_color_shortcut()
        self.update_rubber_band()

    def _handle_trace_color_shortcut(self):
        if self.last_mouse_event_pos is None or self.to_indexes is None:
            return
        try:
            point = self.toMapCoordinates(self.last_mouse_event_pos)
            color = self.raster_context.sample_color_at_indices(
                *self.to_indexes(*as_xy(point))
            )
            if color is None:
                return
            if self._enable_trace_color_cb is not None:
                self._enable_trace_color_cb()
            if self._set_trace_color_cb is not None:
                self._set_trace_color_cb(color)
            else:
                self.trace_color_changed(color)
            self.tracing_mode = TracingModes.PATH
        except Exception as error:
            self.report_failure("invalid_input", str(error))

    def handled_shortcut_keys(self):
        return SHORTCUT_KEYS

    def has_active_trace(self):
        return not self.disposed and not self.suspended and bool(self.anchors)

    def canvasReleaseEvent(self, event):
        if self.disposed or self.suspended:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.state.click_rmb(event, self.get_current_vector_layer())
        elif event.button() == Qt.MouseButton.LeftButton:
            self.state.click_lmb(event, self.get_current_vector_layer())

    def canvasMoveEvent(self, event):
        if not self.disposed and not self.suspended:
            self._hover(event.pos())

    def _hover(self, screen_pos):
        self.last_mouse_event_pos = QPoint(screen_pos)
        if self.tracking_is_active:
            return
        if not self.anchors:
            self.marker_snap.hide()
            return
        try:
            endpoint = self.resolve_endpoint(self.toMapCoordinates(screen_pos))
            if self.snap_tolerance is not None or self.snap2_tolerance is not None:
                self.marker_snap.setCenter(QgsPointXY(*endpoint.xy))
                self.marker_snap.show()
            else:
                self.marker_snap.hide()
            self.update_rubber_band(endpoint)
            if self.tracing_mode.is_tracing():
                self.preview_controller.queue(self.make_request(endpoint, screen_pos))
        except Exception:
            self.marker_snap.hide()
            self.preview_controller.clear()

    def update_rubber_band(self, endpoint=None):
        if (
            self.disposed
            or self.suspended
            or not self.anchors
            or self.last_mouse_event_pos is None
        ):
            self.rubber_band.hide()
            return
        end = (
            endpoint.xy
            if endpoint is not None
            else as_xy(self.toMapCoordinates(self.last_mouse_event_pos))
        )
        self.rubber_band.setColor(
            QColor(0, 102, 255)
            if self.tracing_mode == TracingModes.DENSE_LINE
            else QColor(255, 0, 0)
        )
        self.rubber_band.setWidth(3)
        self.rubber_band.setLineStyle(
            Qt.PenStyle.DotLine
            if self.tracing_mode.is_tracing()
            else Qt.PenStyle.SolidLine
        )
        self.rubber_band.setToGeometry(
            QgsGeometry.fromPolylineXY(
                [QgsPointXY(*self.anchors[-1].xy), QgsPointXY(*end)]
            ),
            None,
        )
        self.rubber_band.show()

    def trace_over_image(self, start, goal):
        """Synchronous numerical probe; ordinary input always uses the scheduler."""
        from .pointtool_session import WorkerRequest

        sampler = self.raster_context.raster_sampler
        bounds = self.raster_context.compute_window_bounds([start, goal])
        work = WorkerRequest(
            new_id(),
            sampler.source,
            bounds,
            start,
            goal,
            self.trace_color_value,
            self.smooth_line,
        )
        result = execute_request(work)
        if result.status != "success":
            return None, None
        return [
            (i + result.origin[0], j + result.origin[1]) for i, j in result.path
        ], result.cost

    def tr(self, message):
        return QCoreApplication.translate("RasterScribePointTool", message)

    def report_failure(self, status, detail=""):
        if self.disposed or status == "cancelled":
            return
        if status == "resource_limit":
            message = self.tr("Tracing limit reached. Please choose a shorter segment.")
        elif status == "no_path":
            message = self.tr("Unable to find a path between the selected points.")
        elif status == "invalid_input":
            message = self.tr(
                "Cannot trace this endpoint. Check the editable vector layer and valid RGB raster pixels."
            )
        else:
            message = self.tr(
                "Unable to finish the segment. Previous geometry has been preserved."
            )
        self.iface.messageBar().pushMessage(
            self.tr("Raster Scribe"), message, Qgis.MessageLevel.Warning, 3
        )
        if detail:
            QgsMessageLog.logMessage(detail, "Raster Scribe", Qgis.MessageLevel.Warning)

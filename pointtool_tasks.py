"""One plugin-owned scheduler for preview and committed raster work."""

import cProfile
import io
import math
import os
import pstats
import time
import weakref
from dataclasses import dataclass, replace

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsGeometry,
    QgsLineString,
    QgsMessageLog,
    QgsTask,
)

from .astar import _find_path_core
from .exceptions import (
    InvalidRasterError,
    OutsideMapError,
    ResourceLimitError,
    TraceCancelled,
)
from .line_simplification import smooth
from .pointtool_raster import check_cancel, prepare_snapshot, validate_budget
from .pointtool_session import TraceResult

PROFILE_ENABLED = os.environ.get("RASTER_SCRIBE_PROFILE", "0") == "1"


def postprocess_path(path):
    """Keep the moving mean, then simplify within a quarter pixel in QGIS."""
    points = smooth(path, size=5)
    if len(points) <= 2:
        return tuple(points)
    # Owned entirely by this worker; no layer, project or GUI objects are used.
    line = QgsLineString(points).simplifyByDistance(0.25)
    if line is None:
        raise ValueError("Path simplification failed")
    result = [(point.x(), point.y()) for point in QgsGeometry(line).asPolyline()]
    if len(result) < 2 or not all(math.isfinite(v) for point in result for v in point):
        raise ValueError("Path simplification returned invalid coordinates")
    # Native simplification can remove the start of a closed line. Keep its
    # remaining vertices as well as the original anchor in that case.
    if points[0] == points[-1] and result[0] != points[0]:
        result.insert(0, points[0])
    result[0], result[-1] = points[0], points[-1]
    return tuple(result)


def execute_request(work, snapshot=None, cancel=lambda: False):
    """All large reads, costs, search and smoothing execute in the worker."""
    timings = {}
    try:
        snapshot, graph, valid, timings = prepare_snapshot(work, snapshot, cancel)
        top, _, left, _ = work.bounds
        start = (work.start[0] - top, work.start[1] - left)
        goal = (work.goal[0] - top, work.goal[1] - left)
        result = _find_path_core(graph, start, goal, cancel, valid)
        timings.update(
            search=result.profile_stats["duration"], nodes=result.profile_stats["nodes"]
        )
        check_cancel(cancel)
        started = time.perf_counter()
        path = tuple(result.path or ())
        processed = postprocess_path(path) if work.smoothing else path
        timings["postprocess"] = time.perf_counter() - started
        check_cancel(cancel)
        return TraceResult(
            work.request_id,
            result.status,
            path,
            result.cost,
            (top, left),
            processed,
            snapshot,
            timings=timings,
        )
    except TraceCancelled:
        # Only fully prepared immutable snapshots reach this local variable.
        # Keeping one does not authorize delivery of its revoked path result.
        return TraceResult(
            work.request_id, "cancelled", snapshot=snapshot, timings=timings
        )
    except ResourceLimitError as error:
        status, detail = "resource_limit", str(error)
    except (InvalidRasterError, OutsideMapError, ValueError) as error:
        status, detail = "invalid_input", str(error)
    except Exception as error:
        status, detail = "error", f"{type(error).__name__}: {error}"
    return TraceResult(work.request_id, status, diagnostics=detail, timings=timings)


class FindPathTask(QgsTask):
    def __init__(self, work, snapshot, callback):
        super().__init__("Raster Scribe", QgsTask.Flag.CanCancel)
        self.work = work
        self.snapshot = snapshot
        self.callback = callback
        self.outcome = None
        self._terminal_snapshot = None
        self._reported = False
        self.profile = ""

    def run(self):
        profiler = cProfile.Profile() if PROFILE_ENABLED else None
        try:
            if profiler:
                profiler.enable()
            self.outcome = execute_request(self.work, self.snapshot, self.isCanceled)
        except Exception as error:
            self.outcome = TraceResult(
                self.work.request_id, "error", diagnostics=str(error)
            )
        finally:
            self.snapshot = None
            if profiler:
                profiler.disable()
                output = io.StringIO()
                pstats.Stats(profiler, stream=output).sort_stats("cumtime").print_stats(
                    15
                )
                self.profile = output.getvalue()
        return self.outcome.status == "success"

    def finished(self, success):
        if self._reported:
            return
        self._reported = True
        outcome = self.outcome
        if self.isCanceled():
            outcome = TraceResult(
                self.work.request_id,
                "cancelled",
                snapshot=outcome.snapshot if outcome is not None else None,
                timings=outcome.timings if outcome is not None else {},
            )
        elif outcome is None or (not success and outcome.status == "success"):
            outcome = TraceResult(
                self.work.request_id, "error", diagnostics="Task terminated"
            )
        # Keep the handoff on the task, not in callback arguments. A terminal
        # callback may immediately start another worker after evicting this
        # cache; this frame and instrumentation wrappers must not retain it.
        self._terminal_snapshot = outcome.snapshot
        outcome = replace(outcome, snapshot=None)
        self.outcome = self.snapshot = None
        callback, self.callback = self.callback, None
        if PROFILE_ENABLED:
            QgsMessageLog.logMessage(
                f"[profiling] {outcome.timings}\n{self.profile}",
                "Raster Scribe",
                Qgis.MessageLevel.Info,
            )
        try:
            callback(self, outcome)
        finally:
            self._terminal_snapshot = None


@dataclass
class ScheduledRequest:
    request: object
    callback: object
    committed: bool
    cache_generation: int = 0


class TraceTaskController:
    """Keep allocation ownership until terminal notification, including close/reopen.

    Descriptors and callbacks are GUI-only. A task receives a numerical subset
    and a scheduler terminal callback; it never dereferences GUI state in run().
    """

    def __init__(self, task_factory=FindPathTask, submit=None):
        self._task = None
        self._running = None
        self._queued = None
        self._snapshot = None
        self._closed = False
        self._cache_generation = 0
        self._factory = task_factory
        self._submit = submit or QgsApplication.taskManager().addTask

    @property
    def active(self):
        return self._task is not None

    def matching(self, request):
        for job in (self._queued, self._running):
            if (
                job is not None
                and job.callback is not None
                and job.request.key == request.key
            ):
                return job.request
        return None

    def adopt(self, request_id):
        for job in (self._queued, self._running):
            if (
                job
                and job.request.request_id == request_id
                and job.callback is not None
            ):
                job.committed = True
                return True
        return False

    def submit(self, request, callback, committed=False):
        if self._closed:
            return False
        job = ScheduledRequest(request, weakref.WeakMethod(callback), committed)
        if self._running is not None:
            if not committed and (
                (self._running.committed and self._running.callback is not None)
                or (self._queued is not None and self._queued.committed)
            ):
                return False
            # Revoke callback before cooperative cancellation. Keep task ownership.
            self._queued = job
            self._running.callback = None
            self._task.cancel()
        else:
            self._start(job)
        return True

    def _start(self, job):
        snapshot, self._snapshot = self._snapshot, None
        work = job.request.worker_input()
        if snapshot is not None and not snapshot.covers(work.source, work.bounds):
            snapshot = None
        # Unused cache is evicted before any new allocation. On fixed-color
        # changes drop the old cost but retain reusable immutable RGB/mask.
        if snapshot is not None and work.color is not None:
            if snapshot.cost_key != (True, work.color, "float64/int64/masked-v1"):
                snapshot = replace(snapshot, cost=None, cost_key=None)
        self._running = job
        job.cache_generation = self._cache_generation
        task = self._factory(work, snapshot, self._terminal)
        self._task = task
        try:
            validate_budget(work.bounds)
            self._submit(task)
        except Exception as error:
            status = (
                "resource_limit" if isinstance(error, ResourceLimitError) else "error"
            )
            self._terminal(
                task, TraceResult(work.request_id, status, diagnostics=str(error))
            )

    def _terminal(self, task, result):
        if task is not self._task:
            return
        job = self._running
        self._task = self._running = None
        if not self._closed and job.cache_generation == self._cache_generation:
            self._snapshot = (
                getattr(task, "_terminal_snapshot", None) or result.snapshot
            )
        task._terminal_snapshot = task.snapshot = task.outcome = None
        # The preview retains only the path, never a second large cache owner.
        result = replace(result, snapshot=None)
        callback = job.callback() if job.callback is not None else None
        try:
            if callback is not None and not self._closed:
                callback(job.request, result)
        finally:
            # Release previous task numerical refs before replacement allocation.
            task.snapshot = task.outcome = None
            if self._queued is not None and not self._closed and self._task is None:
                next_job, self._queued = self._queued, None
                self._start(next_job)

    def cancel(self, owner=None, speculative_only=False, evict=True):
        def matches(job):
            return (
                job is not None
                and (owner is None or job.request.owner == owner)
                and not (speculative_only and job.committed)
            )

        if matches(self._queued):
            self._queued = None
        if matches(self._running):
            self._running.callback = None
            self._task.cancel()
        if evict:
            # A late cancelled task must not repopulate a cache explicitly
            # cleared by finishing, switching source/context or unloading.
            self._cache_generation += 1
            self._snapshot = None

    def shutdown(self):
        self._closed = True
        self.cancel()

"""Logical tracing state and explicit GUI/worker request boundaries."""

from dataclasses import dataclass, field
from itertools import count
from typing import Optional

_ids = count(1)


def new_id():
    return next(_ids)


def as_xy(point):
    if hasattr(point, "x"):
        return point.x(), point.y()
    return point[0], point[1]


@dataclass(frozen=True)
class ResolvedEndpoint:
    x: float
    y: float
    i: int
    j: int

    @property
    def xy(self):
        return self.x, self.y

    @property
    def pixel(self):
        return self.i, self.j


@dataclass(frozen=True)
class TraceContext:
    raster_id: str
    source: object
    raster_crs: object
    working_crs: object
    vector_crs: object
    transform_context: object
    geo_ref: tuple
    raster_to_map: object
    map_to_vector: object


@dataclass(frozen=True)
class RasterSourceSpec:
    path: str
    height: int
    width: int
    revision: int
    bands: tuple = (1, 2, 3)
    open_options: tuple = ()


@dataclass(frozen=True)
class WorkerRequest:
    """Only plain values; never layers, providers, transforms or UI callbacks."""

    request_id: int
    source: RasterSourceSpec
    bounds: tuple
    start: tuple
    goal: tuple
    color: Optional[tuple]
    smoothing: bool


@dataclass(frozen=True)
class TraceRequest:
    request_id: int
    owner: int
    session_id: int
    revision: int
    context: TraceContext
    start: ResolvedEndpoint
    goal: ResolvedEndpoint
    target_id: str
    mode: str
    smoothing: bool
    color: Optional[tuple]
    bounds: tuple
    screen_pos: Optional[tuple] = None

    @property
    def key(self):
        return (
            self.owner,
            self.session_id,
            self.revision,
            self.context.source,
            self.start.pixel,
            self.goal.pixel,
            self.target_id,
            self.mode,
            self.smoothing,
            self.color,
            self.bounds,
        )

    def worker_input(self):
        return WorkerRequest(
            self.request_id,
            self.context.source,
            self.bounds,
            self.start.pixel,
            self.goal.pixel,
            self.color,
            self.smoothing,
        )


@dataclass(frozen=True)
class TraceResult:
    request_id: int
    status: str
    path: tuple = ()
    cost: Optional[int] = None
    origin: tuple = (0, 0)
    processed: tuple = ()
    snapshot: object = None
    diagnostics: str = ""
    timings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PendingSegment:
    request: TraceRequest
    adopted_preview: bool = False


class TraceSession:
    def __init__(self):
        self.session_id = new_id()
        self.revision = 0
        self.target_id = None
        self.geometry = None
        self.anchors = []
        self.pending = None
        self.segment_vertices = []

    def invalidate(self):
        self.revision += 1
        self.pending = None

    def reset(self):
        self.invalidate()
        self.session_id = new_id()
        self.target_id = None
        self.geometry = None
        self.anchors.clear()
        self.segment_vertices.clear()

    def bind(self, target_id, anchor):
        self.target_id = target_id
        self.anchors.append(anchor)

    def begin(self, request, adopted=False):
        if self.pending is not None:
            raise RuntimeError("A segment is already pending")
        self.pending = PendingSegment(request, adopted)

    def matches(self, request):
        return (
            request.session_id == self.session_id
            and request.revision == self.revision
            and request.target_id == self.target_id
        )

    def finish(self, request_id):
        if self.pending is not None and self.pending.request.request_id == request_id:
            self.pending = None
            return True
        return False

    def accept(self, request, geometry, previous_vertices):
        if not self.finish(request.request_id):
            return False
        self.geometry = geometry
        self.anchors.append(request.goal)
        self.segment_vertices.append(previous_vertices)
        self.revision += 1
        return True

    def undo(self, geometry):
        self.segment_vertices.pop()
        self.anchors.pop()
        self.geometry = geometry
        self.revision += 1

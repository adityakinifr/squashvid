"""Squash court geometry: World Squash Federation singles defaults, adjustable per venue.

World frame, metres: x runs across the court from the left side wall (as seen
from behind the back wall), y runs from the back wall to the front wall, z is
up. Heights in the WSF specification are measured to the underside of each
50 mm line, so line centres sit 25 mm above the quoted height.

Real club courts drift from the specification (court 5 in the first test
footage has its service-box inner lines ~16 cm closer to the centre), so every
derived quantity reads from a CourtSpec that calibration can adjust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

LINE_W = 0.05
HALF_LINE = LINE_W / 2


@dataclass(frozen=True)
class CourtSpec:
    width: float = 6.40
    length: float = 9.75
    short_line_near_edge: float = 4.26  # back wall to the short line's nearest edge
    box_depth: float = 1.60  # internal, along the court
    box_width: float = 1.60  # internal, from the side wall
    front_out: float = 4.57
    service_line: float = 1.78
    tin_top: float = 0.48  # top of the 50 mm board line; PSA events use 0.43
    back_out: float = 2.13

    # Derived line centres
    @property
    def short_y(self) -> float:
        return self.short_line_near_edge + HALF_LINE

    @property
    def box_back_y(self) -> float:
        return self.short_line_near_edge - self.box_depth - HALF_LINE

    @property
    def box_inner_left_x(self) -> float:
        return self.box_width + HALF_LINE

    @property
    def box_inner_right_x(self) -> float:
        return self.width - self.box_width - HALF_LINE

    @property
    def half_x(self) -> float:
        return self.width / 2

    @property
    def t_point(self) -> np.ndarray:
        return np.array([self.half_x, self.short_y, 0.0])

    @property
    def front_out_z(self) -> float:
        return self.front_out + HALF_LINE

    @property
    def service_z(self) -> float:
        return self.service_line + HALF_LINE

    @property
    def tin_z(self) -> float:
        return self.tin_top - HALF_LINE

    @property
    def back_out_z(self) -> float:
        return self.back_out + HALF_LINE

    def side_out_z(self, y: float | np.ndarray) -> float | np.ndarray:
        """Centre height of the diagonal side-wall out line at distance y from the back wall."""
        return self.back_out_z + (self.front_out - self.back_out) * np.asarray(y) / self.length

    def with_(self, **changes) -> "CourtSpec":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


WSF = CourtSpec()


@dataclass(frozen=True)
class CourtLine:
    name: str
    surface: str  # floor | front | left | right | back
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    kind: str = "paint"  # paint: a painted line; edge: a surface junction

    def sample(self, step: float) -> np.ndarray:
        a = np.asarray(self.a, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        return a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]


def court_lines(spec: CourtSpec = WSF) -> list[CourtLine]:
    """Centrelines of every marking a back-wall camera can see, plus the wall-floor junctions."""
    s = spec
    L, W = s.length, s.width
    return [
        CourtLine("short_line", "floor", (0, s.short_y, 0), (W, s.short_y, 0)),
        CourtLine("half_court", "floor", (s.half_x, 0, 0), (s.half_x, s.short_line_near_edge, 0)),
        CourtLine("box_back_left", "floor", (0, s.box_back_y, 0), (s.box_inner_left_x, s.box_back_y, 0)),
        CourtLine("box_inner_left", "floor", (s.box_inner_left_x, s.box_back_y, 0), (s.box_inner_left_x, s.short_line_near_edge, 0)),
        CourtLine("box_back_right", "floor", (s.box_inner_right_x, s.box_back_y, 0), (W, s.box_back_y, 0)),
        CourtLine("box_inner_right", "floor", (s.box_inner_right_x, s.box_back_y, 0), (s.box_inner_right_x, s.short_line_near_edge, 0)),
        CourtLine("front_out", "front", (0, L, s.front_out_z), (W, L, s.front_out_z)),
        CourtLine("service_line", "front", (0, L, s.service_z), (W, L, s.service_z)),
        CourtLine("tin", "front", (0, L, s.tin_z), (W, L, s.tin_z)),
        CourtLine("side_out_left", "left", (0, 0, s.back_out_z), (0, L, s.front_out_z)),
        CourtLine("side_out_right", "right", (W, 0, s.back_out_z), (W, L, s.front_out_z)),
        CourtLine("front_floor_edge", "front", (0, L, 0), (W, L, 0), "edge"),
        CourtLine("left_floor_edge", "left", (0, 0, 0), (0, L, 0), "edge"),
        CourtLine("right_floor_edge", "right", (W, 0, 0), (W, L, 0), "edge"),
        CourtLine("front_left_corner", "front", (0, L, 0), (0, L, s.front_out_z), "edge"),
        CourtLine("front_right_corner", "front", (W, L, 0), (W, L, s.front_out_z), "edge"),
    ]


def landmarks(spec: CourtSpec = WSF) -> dict[str, tuple[float, float, float]]:
    """Named points used for click calibration and held-out checks."""
    s = spec
    L, W = s.length, s.width
    return {
        "short_x_left_wall": (0.0, s.short_y, 0.0),
        "box_back_x_left_wall": (0.0, s.box_back_y, 0.0),
        "box_left_inner_x_back": (s.box_inner_left_x, s.box_back_y, 0.0),
        "box_left_inner_x_short": (s.box_inner_left_x, s.short_y, 0.0),
        "t_point": (s.half_x, s.short_y, 0.0),
        "box_right_inner_x_short": (s.box_inner_right_x, s.short_y, 0.0),
        "box_right_inner_x_back": (s.box_inner_right_x, s.box_back_y, 0.0),
        "short_x_right_wall": (W, s.short_y, 0.0),
        "service_line_left": (0.0, L, s.service_z),
        "service_line_right": (W, L, s.service_z),
        "tin_left": (0.0, L, s.tin_z),
        "tin_right": (W, L, s.tin_z),
        "front_out_left": (0.0, L, s.front_out_z),
        "front_out_right": (W, L, s.front_out_z),
        "front_floor_left": (0.0, L, 0.0),
        "front_floor_right": (W, L, 0.0),
    }

"""Region classification for ball points, on a synthetic camera behind the back wall."""

import cv2
import numpy as np
import pytest

from squashvid.ball.regions import REGIONS, classify, classify_rays
from squashvid.geom import court
from squashvid.geom.camera import Camera

SPEC = court.WSF.with_(width=6.15, length=9.94)


def camera(centre=(3.0, -2.0, 1.4)) -> Camera:
    # Looking straight down the court: image x = world x, image y = -world z.
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    C = np.array(centre, float)
    return Camera(500.0, 512.0, 288.0, cv2.Rodrigues(R)[0].reshape(3), -R @ C)


@pytest.mark.parametrize(
    "point, region",
    [
        ((3.0, 9.94, 2.0), "front_wall"),
        ((3.0, 9.94, 5.5), "air"),  # above the front out line
        ((0.0, 6.0, 1.0), "side_walls"),
        ((6.15, 3.0, 1.5), "side_walls"),
        ((0.0, 8.0, 4.8), "air"),  # above the sloping side out line
        ((3.0, 2.5, 0.0), "floor_back"),
        ((3.0, 7.5, 0.0), "floor_front"),
    ],
)
def test_surface_behind_point(point, region):
    cam = camera()
    uv = cam.project(np.array([point]))
    assert classify(cam, uv, SPEC, door=())[0] == region


def test_first_surface_wins_over_farther_ones():
    # A ray that grazes the floor just behind the short line must not be called front wall.
    cam = camera()
    uv = cam.project(np.array([[3.0, SPEC.short_y - 0.2, 0.0]]))
    assert classify(cam, uv, SPEC, door=())[0] == "floor_back"


def test_door_band_overrides_any_surface():
    cam = camera()
    door = ((1.0, 1.4),)
    behind_band = cam.project(np.array([[1.2, 0.0, 1.0], [1.2, 0.0, 0.2]]))  # through the band to wall and floor
    clear = cam.project(np.array([[2.0, 0.0, 1.0]]))
    assert list(classify(cam, behind_band, SPEC, door=door)) == ["door_strip", "door_strip"]
    assert classify(cam, clear, SPEC, door=door)[0] != "door_strip"


def test_rays_leaving_the_court_are_air():
    cam = camera()
    up = np.array([[0.0, 0.2, 1.0]]) / np.linalg.norm([0.0, 0.2, 1.0])
    assert classify_rays(cam.centre, up, SPEC, door=())[0] == "air"


def test_every_pixel_gets_a_known_region():
    cam = camera()
    ys, xs = np.mgrid[0:576:16, 0:1024:16]
    names = classify(cam, np.column_stack([xs.ravel(), ys.ravel()]), SPEC)
    assert set(names) <= set(REGIONS)
    assert {"front_wall", "side_walls", "floor_back"} <= set(names)


def test_empty_input():
    assert len(classify(camera(), np.zeros((0, 2)), SPEC)) == 0

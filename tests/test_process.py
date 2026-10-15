import json

import pytest

from squashvid.ball import clips
from squashvid.process import Stage, execute, plan


def test_registered_stem_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(clips, "WORK", tmp_path)
    video = tmp_path / "new.mov"
    video.write_bytes(b"")
    reg = clips.register("new_clip", video)
    assert reg == tmp_path / "clips.json" and json.loads(reg.read_text()) == {"new_clip": str(video.resolve())}
    assert clips.clip_path("new_clip") == video.resolve()
    clips.register("other", video)
    assert set(clips.registered()) == {"new_clip", "other"}


def test_existing_stems_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(clips, "WORK", tmp_path)
    clips.register("new_clip", tmp_path / "x.mov")
    for orig, copy in clips.COPIES.items():
        assert clips.clip_path(orig).name == f"{orig}.MOV" and clips.clip_path(copy).name == f"{copy}.mp4"
        assert clips.is_original(orig) and not clips.is_original(copy)
        assert clips.venue_path(orig) == clips.VENUE_1080 and clips.venue_path(copy) == clips.VENUE


def test_unknown_stem_errors_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(clips, "WORK", tmp_path)
    with pytest.raises(KeyError, match="unknown clip 'nope'"):
        clips.clip_path("nope")


def test_venue_by_frame_size_and_unknown_court(tmp_path):
    from squashvid.calib.venue import Venue

    for src in (clips.VENUE, clips.VENUE_1080):
        Venue.load(src).save(tmp_path / src.name)
    assert clips.venue_for_size(1920, 1080, tmp_path).name == clips.VENUE_1080.name
    assert clips.venue_for_size(1024, 576, tmp_path).name == clips.VENUE.name
    with pytest.raises(FileNotFoundError, match="court 5 only"):
        clips.venue_for_size(1280, 720, tmp_path)


def _stages(tmp_path, ran):
    def stage(name):
        out = tmp_path / f"{name}.out"

        def run():
            ran.append(name)
            out.write_text("x")

        return Stage(name, lambda: [out], run)

    return [stage(n) for n in ("a", "b", "c")]


def test_stages_run_then_skip(tmp_path):
    ran = []
    stages = _stages(tmp_path, ran)
    execute(stages, out=lambda *_: None)
    assert ran == ["a", "b", "c"]
    rows = execute(stages, out=lambda *_: None)
    assert ran == ["a", "b", "c"] and [r["status"] for r in rows] == ["skipped"] * 3


def test_missing_output_reruns_only_that_stage(tmp_path):
    ran = []
    stages = _stages(tmp_path, ran)
    execute(stages, out=lambda *_: None)
    (tmp_path / "b.out").unlink()
    execute(stages, out=lambda *_: None)
    assert ran == ["a", "b", "c", "b"]


def test_from_and_only_force(tmp_path):
    ran = []
    stages = _stages(tmp_path, ran)
    execute(stages, out=lambda *_: None)
    ran.clear()
    execute(stages, start="b", out=lambda *_: None)
    assert ran == ["b", "c"]
    ran.clear()
    execute(stages, only="a", out=lambda *_: None)
    assert ran == ["a"]
    with pytest.raises(ValueError, match="unknown stage"):
        plan(stages, start="zzz")
    with pytest.raises(ValueError, match="not both"):
        plan(stages, start="a", only="b")


def test_incomplete_output_counts_as_pending(tmp_path):
    out = tmp_path / "ball.parquet"
    out.write_text("x")
    partial = Stage("ball", lambda: [out], lambda: None, complete=lambda: False)
    assert [go for _, go in plan([partial])] == [True]


def test_failing_stage_names_itself(tmp_path):
    def boom():
        raise ValueError("bad")

    with pytest.raises(RuntimeError, match="stage 'a' failed.*bad"):
        execute([Stage("a", lambda: [tmp_path / "a"], boom)], out=lambda *_: None)

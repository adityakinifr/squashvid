"""Player tracking: detection, tracklets, closed-world two-player identity, foot point and court trajectories.

Squash always has exactly two players, so identity is solved offline as a
two-label assignment over short, conservative tracklets rather than trusted to
a generic multi-object tracker. Positions live in court metres through the
per-frame camera from `squashvid.calib`.
"""

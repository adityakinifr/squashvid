"""Top-down 2D pose with RTMPose-m (body7, 256x192) through rtmlib on the CoreML execution provider.

Players are 60-250 px tall, so a top-down crop upsampled to 256x192 beats a
bottom-up model. Torso, hips and knees are reliable at this bitrate; wrists and
ankles are noisy, which is why the foot point keeps a fallback ladder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from squashvid.players.detect import THIRD_PARTY

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
N_KPT = 17


def model_path() -> Path:
    hits = sorted(THIRD_PARTY.glob("rtmpose_m/**/end2end.onnx"))
    if not hits:
        raise FileNotFoundError(
            "RTMPose-m ONNX missing: unzip rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip "
            f"from download.openmmlab.com into {THIRD_PARTY / 'rtmpose_m'}"
        )
    return hits[0]


class PoseModel:
    """RTMPose-m keypoints for a list of boxes in one frame."""

    def __init__(self, device: str = "mps"):
        from rtmlib import RTMPose

        self.model = RTMPose(str(model_path()), model_input_size=(192, 256), backend="onnxruntime", device=device)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(keypoints (N, 17, 2) in image pixels, scores (N, 17))."""
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        if not len(boxes):
            return np.zeros((0, N_KPT, 2), np.float32), np.zeros((0, N_KPT), np.float32)
        k, s = self.model(image, boxes.tolist())
        return np.asarray(k, np.float32).reshape(-1, N_KPT, 2), np.asarray(s, np.float32).reshape(-1, N_KPT)

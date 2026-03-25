#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import cv2

try:
    from pupil_apriltags import Detector
except ImportError as e:
    raise SystemExit("pip install pupil-apriltags") from e


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


@dataclass
class BoardConfig:
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.162
    horizontal_inner_gap_m: float = 0.306
    vertical_inner_gap_m: float = 0.024

    # (top-left, top-right, bottom-left, bottom-right)
    tag_ids: Tuple[int, int, int, int] = (0, 1, 2, 3)

    min_decision_margin: float = 25.0
    nthreads: int = 4
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25


class AprilTagBoardPoseEstimator:
    """
    Estimates T_board_in_cam from a 2x2 AprilTag board using solvePnP.

    board_frame:
      origin = center of the 4 tag centers
      +x = right across board
      +y = up across board
      +z = out of board plane
    """

    def __init__(self, cfg: BoardConfig):
        self.cfg = cfg
        self.detector = Detector(
            families=cfg.tag_family,
            nthreads=cfg.nthreads,
            quad_decimate=cfg.quad_decimate,
            quad_sigma=cfg.quad_sigma,
            refine_edges=cfg.refine_edges,
            decode_sharpening=cfg.decode_sharpening,
        )
        self._tag_centers = self._make_tag_centers()

    def _make_tag_centers(self) -> Dict[int, np.ndarray]:
        s = self.cfg.tag_size_m
        dx = 0.5 * (s + self.cfg.horizontal_inner_gap_m)
        dy = 0.5 * (s + self.cfg.vertical_inner_gap_m)

        tl, tr, bl, br = self.cfg.tag_ids
        return {
            tl: np.array([-dx, +dy, 0.0], dtype=np.float32),
            tr: np.array([+dx, +dy, 0.0], dtype=np.float32),
            bl: np.array([-dx, -dy, 0.0], dtype=np.float32),
            br: np.array([+dx, -dy, 0.0], dtype=np.float32),
        }

    def _tag_object_corners(self, tag_id: int) -> np.ndarray:
        """
        Returns 4x3 object points for one tag, in the same corner order
        as pupil_apriltags detection.corners:
            [top-left, top-right, bottom-right, bottom-left]
        """
        c = self._tag_centers[tag_id]
        h = self.cfg.tag_size_m / 2.0

        pts = np.array([
            [c[0] - h, c[1] + h, 0.0],  # top-left
            [c[0] + h, c[1] + h, 0.0],  # top-right
            [c[0] + h, c[1] - h, 0.0],  # bottom-right
            [c[0] - h, c[1] - h, 0.0],  # bottom-left
        ], dtype=np.float32)
        return pts

    def detect(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray, estimate_tag_pose=False)

    def estimate(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Dict[str, Any]:
        detections = self.detect(bgr)

        all_detected_ids: List[int] = []
        obj_pts: List[np.ndarray] = []
        img_pts: List[np.ndarray] = []
        used_ids: List[int] = []

        for d in detections:
            tid = int(d.tag_id)
            all_detected_ids.append(tid)

            if tid not in self._tag_centers:
                continue
            if float(d.decision_margin) < self.cfg.min_decision_margin:
                continue

            obj_pts.append(self._tag_object_corners(tid))
            img_pts.append(np.asarray(d.corners, dtype=np.float32))
            used_ids.append(tid)

        all_detected_ids = sorted(set(all_detected_ids))
        used_ids_unique = sorted(set(used_ids))

        if len(obj_pts) == 0:
            return {
                "ok": False,
                "reason": "no usable board tags found",
                "used_ids": [],
                "all_detected_ids": all_detected_ids,
            }

        # Relaxed requirement: at least 3 tags
        if len(used_ids_unique) < 3:
            return {
                "ok": False,
                "reason": "need at least 3 board tags",
                "used_ids": used_ids_unique,
                "all_detected_ids": all_detected_ids,
            }

        obj = np.concatenate(obj_pts, axis=0).astype(np.float32)
        img = np.concatenate(img_pts, axis=0).astype(np.float32)

        K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)

        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return {
                "ok": False,
                "reason": "solvePnP failed",
                "used_ids": used_ids_unique,
                "all_detected_ids": all_detected_ids,
            }

        R, _ = cv2.Rodrigues(rvec)
        T_board_in_cam = Rt_to_T(R, tvec.reshape(3))

        return {
            "ok": True,
            "T_board_in_cam": T_board_in_cam,
            "used_ids": used_ids_unique,
            "all_detected_ids": all_detected_ids,
            "num_corners": int(obj.shape[0]),
            "rvec": rvec,
            "tvec": tvec,
        }
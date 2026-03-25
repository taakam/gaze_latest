#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from realsense_stream import RealSenseStream, RealSenseConfig
from board_pose import BoardConfig, AprilTagBoardPoseEstimator

# -----------------------------
# USER CONFIG
# -----------------------------
SHOW_WINDOW = True

BOARD_TAG_SIZE_M = 0.162
BOARD_HORIZONTAL_INNER_GAP_M = 0.306
BOARD_VERTICAL_INNER_GAP_M = 0.024

# IMPORTANT:
# Put the REAL IDs in this order:
# (top-left, top-right, bottom-left, bottom-right)
BOARD_TAG_IDS = (0, 1, 2, 3)

MIN_DECISION_MARGIN = 25.0

# depth patch radius around tag center pixel
PATCH_RADIUS = 2   # 2 => 5x5 patch

# -----------------------------
# HELPERS
# -----------------------------
def pixel_to_rs_xyz(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.array([X, Y, Z], dtype=np.float32)


def median_depth_in_patch(depth_m: np.ndarray, u: int, v: int, r: int) -> Optional[float]:
    H, W = depth_m.shape[:2]

    u0 = max(0, u - r)
    u1 = min(W, u + r + 1)
    v0 = max(0, v - r)
    v1 = min(H, v + r + 1)

    patch = depth_m[v0:v1, u0:u1]
    vals = patch[np.isfinite(patch)]
    vals = vals[(vals > 0.05) & (vals < 5.0)]

    if vals.size == 0:
        return None

    return float(np.median(vals))


def assign_roles_by_board_ids(points_by_id: Dict[int, np.ndarray], tag_ids: Tuple[int, int, int, int]):
    tl_id, tr_id, bl_id, br_id = tag_ids

    if not all(tid in points_by_id for tid in tag_ids):
        return None

    tl = points_by_id[tl_id]
    tr = points_by_id[tr_id]
    bl = points_by_id[bl_id]
    br = points_by_id[br_id]

    center = 0.25 * (tl + tr + bl + br)

    return {
        "tl": tl,
        "tr": tr,
        "bl": bl,
        "br": br,
        "center": center.astype(np.float32),
    }


def draw_tag_overlay(img, det, xyz: Optional[np.ndarray]):
    corners = np.asarray(det.corners, dtype=np.int32)
    center = np.asarray(det.center, dtype=np.int32)
    tid = int(det.tag_id)
    dm = float(det.decision_margin)

    # outline
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    # center
    cv2.circle(img, tuple(center), 5, (0, 255, 255), -1)

    label = f"id={tid} dm={dm:.1f}"
    cv2.putText(
        img,
        label,
        (center[0] + 8, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if xyz is not None:
        txt = f"[{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]"
        cv2.putText(
            img,
            txt,
            (center[0] + 8, center[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


# -----------------------------
# MAIN
# -----------------------------
def main():
    rs_cfg = RealSenseConfig(
        color_wh_fps=(1280, 720, 30),
        depth_wh_fps=(848, 480, 30),
        align_to_color=True,
        enable_filters=False,
        max_depth_m=4.0,
    )

    board_cfg = BoardConfig(
        tag_family="tag36h11",
        tag_size_m=BOARD_TAG_SIZE_M,
        horizontal_inner_gap_m=BOARD_HORIZONTAL_INNER_GAP_M,
        vertical_inner_gap_m=BOARD_VERTICAL_INNER_GAP_M,
        tag_ids=BOARD_TAG_IDS,
        min_decision_margin=MIN_DECISION_MARGIN,
    )
    estimator = AprilTagBoardPoseEstimator(board_cfg)

    if SHOW_WINDOW:
        cv2.namedWindow("RS tag centers", cv2.WINDOW_NORMAL)

    print("\nRunning RS point estimator...")
    print("Press 'p' to print the current 4 tag centers + board center in RS frame.")
    print("Press 'q' or ESC to quit.\n")

    with RealSenseStream(rs_cfg) as rs_cam:
        while True:
            color_bgr, depth_m, meta = rs_cam.read()
            if color_bgr is None or depth_m is None:
                continue

            K_rs = np.asarray(meta["K_color"], dtype=np.float32).reshape(3, 3)

            detections = estimator.detect(color_bgr)

            vis = color_bgr.copy()
            points_by_id: Dict[int, np.ndarray] = {}

            for d in detections:
                tid = int(d.tag_id)
                dm = float(d.decision_margin)

                xyz = None

                if tid in BOARD_TAG_IDS and dm >= MIN_DECISION_MARGIN:
                    u = int(round(float(d.center[0])))
                    v = int(round(float(d.center[1])))

                    z = median_depth_in_patch(depth_m, u, v, PATCH_RADIUS)
                    if z is not None:
                        xyz = pixel_to_rs_xyz(u, v, z, K_rs)
                        points_by_id[tid] = xyz

                draw_tag_overlay(vis, d, xyz)

            roles = assign_roles_by_board_ids(points_by_id, BOARD_TAG_IDS)

            if roles is not None:
                center_rs = roles["center"]

                # project board center back to image for overlay
                fx = float(K_rs[0, 0])
                fy = float(K_rs[1, 1])
                cx = float(K_rs[0, 2])
                cy = float(K_rs[1, 2])

                X, Y, Z = float(center_rs[0]), float(center_rs[1]), float(center_rs[2])
                if Z > 1e-6:
                    u_c = int(round(fx * X / Z + cx))
                    v_c = int(round(fy * Y / Z + cy))

                    if 0 <= u_c < vis.shape[1] and 0 <= v_c < vis.shape[0]:
                        cv2.circle(vis, (u_c, v_c), 7, (255, 0, 255), -1)
                        cv2.putText(
                            vis,
                            f"board_center [{center_rs[0]:+.3f}, {center_rs[1]:+.3f}, {center_rs[2]:+.3f}]",
                            (u_c + 10, v_c),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            used_ids = sorted(points_by_id.keys())
            cv2.putText(
                vis,
                f"usable IDs: {used_ids}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "Press p to print RS coordinates",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            if SHOW_WINDOW:
                cv2.imshow("RS tag centers", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("p"):
                print("\n--- CURRENT RS POINTS ---")
                if roles is None:
                    print("Not all required tag centers are available yet.")
                    print(f"Currently usable IDs: {used_ids}")
                else:
                    print(f"TL tag center RS: {roles['tl']}")
                    print(f"TR tag center RS: {roles['tr']}")
                    print(f"BL tag center RS: {roles['bl']}")
                    print(f"BR tag center RS: {roles['br']}")
                    print(f"Board center RS : {roles['center']}")
                    print("-------------------------")

    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
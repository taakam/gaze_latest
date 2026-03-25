#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from realsense_stream import RealSenseStream, RealSenseConfig
from board_pose import AprilTagBoardPoseEstimator, BoardConfig, inv_T


# -----------------------------
# USER SETTINGS
# -----------------------------
OUTPUT_PATH = "T_cam_in_board.npy"
NUM_VALID_SAMPLES = 40
SHOW_WINDOW = True

BOARD_TAG_SIZE_M = 0.162
BOARD_HORIZONTAL_INNER_GAP_M = 0.306
BOARD_VERTICAL_INNER_GAP_M = 0.024

# IMPORTANT: set this to the REAL tag order on your board:
# (top-left, top-right, bottom-left, bottom-right)
BOARD_TAG_IDS = (0, 1, 2, 3)


def average_transforms(T_list: List[np.ndarray]) -> np.ndarray:
    translations = []
    rotations = []

    for T in T_list:
        translations.append(T[:3, 3])
        rotations.append(T[:3, :3])

    t_avg = np.mean(np.stack(translations, axis=0), axis=0)

    R_mean = np.mean(np.stack(rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_avg = U @ Vt

    # Ensure proper rotation matrix
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    T_avg = np.eye(4, dtype=np.float32)
    T_avg[:3, :3] = R_avg.astype(np.float32)
    T_avg[:3, 3] = t_avg.astype(np.float32)
    return T_avg


def draw_axes(img: np.ndarray, K: np.ndarray, dist: np.ndarray, T_board_in_cam: np.ndarray, axis_len=0.08):
    R = T_board_in_cam[:3, :3]
    t = T_board_in_cam[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    obj_pts = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=np.float32)

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    o = tuple(img_pts[0])
    x = tuple(img_pts[1])
    y = tuple(img_pts[2])
    z = tuple(img_pts[3])

    cv2.line(img, o, x, (0, 0, 255), 2)   # x = red
    cv2.line(img, o, y, (0, 255, 0), 2)   # y = green
    cv2.line(img, o, z, (255, 0, 0), 2)   # z = blue
    cv2.putText(img, "x", x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


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
        min_decision_margin=25.0,
    )
    board_est = AprilTagBoardPoseEstimator(board_cfg)

    T_samples: List[np.ndarray] = []
    last_capture_time = 0.0
    capture_interval_s = 0.2

    if SHOW_WINDOW:
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    print("\nCalibration started.")
    print(f"Need {NUM_VALID_SAMPLES} valid samples.")
    print("Keep the camera fixed, board flat, and all tags visible.")
    print("Press 'c' to capture manually, 'a' to auto-capture, 'q' to quit.\n")

    auto_capture = True

    with RealSenseStream(rs_cfg) as rs_cam:
        while True:
            rs_color_bgr, _, rs_meta = rs_cam.read()
            if rs_color_bgr is None:
                continue

            K_rs = np.asarray(rs_meta["K_color"], dtype=np.float32).reshape(3, 3)
            dist_rs = np.asarray(
                rs_meta.get("dist_color", np.zeros((5, 1))),
                dtype=np.float32
            ).reshape(-1, 1)

            board_out = board_est.estimate(rs_color_bgr, K_rs, dist_rs)

            vis = rs_color_bgr.copy()

            if board_out.get("ok", False):
                T_board_in_cam = board_out["T_board_in_cam"]
                T_cam_in_board = inv_T(T_board_in_cam)

                draw_axes(vis, K_rs, dist_rs, T_board_in_cam)

                used_ids = board_out.get("used_ids", [])
                t = T_cam_in_board[:3, 3]

                cv2.putText(
                    vis,
                    f"BOARD OK used_ids={used_ids}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    f"T_cam_in_board t=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                now = time.time()
                if auto_capture and (now - last_capture_time) > capture_interval_s:
                    T_samples.append(T_cam_in_board.copy())
                    last_capture_time = now
                    print(f"[{len(T_samples)}/{NUM_VALID_SAMPLES}] captured")

            else:
                reason = board_out.get("reason", "unknown")
                used_ids = board_out.get("used_ids", [])
                all_detected = board_out.get("all_detected_ids", [])
                cv2.putText(
                    vis,
                    f"BOARD FAIL: {reason}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    f"used_ids={used_ids} all_detected={all_detected}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                vis,
                f"samples: {len(T_samples)}/{NUM_VALID_SAMPLES}   auto_capture={'ON' if auto_capture else 'OFF'}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "keys: a=toggle auto  c=capture once  s=save if enough  q=quit",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if SHOW_WINDOW:
                cv2.imshow("Calibration", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("a"):
                auto_capture = not auto_capture
                print(f"auto_capture = {auto_capture}")
            elif key == ord("c"):
                if board_out.get("ok", False):
                    T_board_in_cam = board_out["T_board_in_cam"]
                    T_cam_in_board = inv_T(T_board_in_cam)
                    T_samples.append(T_cam_in_board.copy())
                    print(f"[{len(T_samples)}/{NUM_VALID_SAMPLES}] captured manually")
                else:
                    print("Manual capture failed: board pose not valid")
            elif key == ord("s"):
                if len(T_samples) < 3:
                    print("Not enough samples to save")
                    continue
                T_avg = average_transforms(T_samples)
                np.save(OUTPUT_PATH, T_avg)
                print(f"\nSaved averaged transform to {OUTPUT_PATH}")
                print("T_cam_in_board =\n", T_avg)

            if len(T_samples) >= NUM_VALID_SAMPLES:
                T_avg = average_transforms(T_samples)
                np.save(OUTPUT_PATH, T_avg)
                print(f"\nSaved averaged transform to {OUTPUT_PATH}")
                print("T_cam_in_board =\n", T_avg)
                break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    if Path(OUTPUT_PATH).exists():
        print(f"\nCalibration complete. File saved: {OUTPUT_PATH}")
        print("Use it at runtime with:")
        print("T_cam_in_board = np.load('T_cam_in_board.npy')")


if __name__ == "__main__":
    main()
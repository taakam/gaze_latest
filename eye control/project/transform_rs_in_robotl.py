#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def solve_rigid_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Solve for T such that:
        p_dst ~= R @ p_src + t

    Inputs:
        src_pts: Nx3 points in source frame  (RS frame)
        dst_pts: Nx3 points in destination frame (robot frame)

    Returns:
        T: 4x4 homogeneous transform mapping source -> destination
    """
    assert src_pts.shape == dst_pts.shape
    assert src_pts.shape[1] == 3
    assert src_pts.shape[0] >= 3

    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_centroid - R @ src_centroid

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply homogeneous transform T to Nx3 points.
    """
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    out_h = (T @ pts_h.T).T
    return out_h[:, :3]


def print_matrix(name: str, M: np.ndarray):
    print(f"{name} = np.array([")
    for row in M:
        print("    [" + ", ".join(f"{v: .6f}" for v in row) + "],")
    print("], dtype=np.float32)\n")


def main():
    # ============================================================
    # 1. RS-FRAME POINTS
    #    Order must exactly match robot points below:
    #    [CENTER, BR, TR, TL, BL]
    # ============================================================
    rs_points = np.array([
        [-3.4319982e-04,  8.6561978e-02,  1.0280000e+00],  # center
        [ 2.1139884e-01,  1.2837125e-01,  9.1300005e-01],  # BR
        [ 2.4496450e-01,  3.6227970e-02,  1.0790000e+00],  # TR
        [-2.1779560e-01,  4.5742830e-02,  1.1380000e+00],  # TL
        [-2.3984054e-01,  1.3590588e-01,  9.8200005e-01],  # BL
    ], dtype=np.float64)

    # ============================================================
    # 2. ROBOT-FRAME POINTS
    #    Same order as rs_points:
    #    [CENTER, BR, TR, TL, BL]
    # ============================================================
    robot_points = np.array([
        [0.633099,  0.000,   0.2075],  # center / cube top
        [0.867099, -0.093,   0.2075],  # BR
        [0.867099,  0.093,   0.2075],  # TR
        [0.399099,  0.093,   0.2075],  # TL
        [0.399099, -0.093,   0.2075],  # BL
    ], dtype=np.float64)

    # ============================================================
    # 3. Solve transform: RS -> ROBOT
    # ============================================================
    T_rs_in_robot = solve_rigid_transform(rs_points, robot_points)

    # ============================================================
    # 4. Evaluate fit
    # ============================================================
    robot_points_est = transform_points(T_rs_in_robot, rs_points)
    residuals = robot_points_est - robot_points
    errors = np.linalg.norm(residuals, axis=1)

    point_names = ["CENTER", "BR", "TR", "TL", "BL"]

    print("\n=== Estimated Transform: RS -> ROBOT ===\n")
    print_matrix("T_rs_in_robot", T_rs_in_robot.astype(np.float32))

    print("Rotation matrix R:")
    print(T_rs_in_robot[:3, :3], "\n")

    print("Translation t:")
    print(T_rs_in_robot[:3, 3], "\n")

    print("det(R) =", np.linalg.det(T_rs_in_robot[:3, :3]), "\n")

    print("=== Per-point residuals ===")
    for name, est, gt, err in zip(point_names, robot_points_est, robot_points, errors):
        print(f"{name}:")
        print(f"  estimated = [{est[0]:+.6f}, {est[1]:+.6f}, {est[2]:+.6f}]")
        print(f"  target    = [{gt[0]:+.6f}, {gt[1]:+.6f}, {gt[2]:+.6f}]")
        print(f"  error [m] = {err:.6f}\n")

    print(f"Mean error [m]: {np.mean(errors):.6f}")
    print(f"Max  error [m]: {np.max(errors):.6f}\n")

    # ============================================================
    # 5. Save result
    # ============================================================
    np.save("T_rs_in_robot.npy", T_rs_in_robot.astype(np.float32))
    print("Saved transform to: T_rs_in_robot.npy")


if __name__ == "__main__":
    main()
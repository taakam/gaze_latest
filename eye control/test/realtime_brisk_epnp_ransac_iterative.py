#!/usr/bin/env python3
import time
import contextlib
import msgpack
import numpy as np
import cv2
import pyrealsense2 as rs
import pupil_labs.pupil_core_network_client as pcnc


# =========================
# USER SETTINGS
# =========================
PUPIL_ADDR = "127.0.0.1"
PUPIL_PORT = 50020
PUPIL_WORLD_TOPIC = "frame.world"
PUPIL_FRAME_FORMAT = "bgr"          # we request this (must also be enabled in Pupil Capture UI)

WORLD_INTRINSICS_PATH = "world.intrinsics"

# RealSense profiles
COLOR_W, COLOR_H, FPS = 1280, 720, 30
DEPTH_W, DEPTH_H = 848, 480

# BRISK
BRISK_THRESH = 30
BRISK_OCTAVES = 3
MATCH_RATIO = 0.75

# Depth filtering
MAX_DEPTH_M = 4.0

# Pose thresholds
RANSAC_REPROJ_ERR = 4.0
REFINE_REPROJ_ERR = 3.0
MIN_PNP_INLIERS = 25
MIN_CORRESPONDENCES = 12

# Runtime throttling
PNP_HZ = 12.0
MAX_REFINE_ITERS = 5
INLIER_STABLE_EPS = 2

SHOW_REALSENSE = True
PRINT_PAYLOAD_KEYS_ONCE = True   # keep True until it works


# =========================
# HELPERS
# =========================
def K_from_rs_intr(intr: rs.intrinsics) -> np.ndarray:
    return np.array([[intr.fx, 0, intr.ppx],
                     [0, intr.fy, intr.ppy],
                     [0, 0, 1]], dtype=np.float32)

def backproject(u: int, v: int, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.array([X, Y, z], dtype=np.float32)

def ema(prev: np.ndarray | None, new: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return new if prev is None else prev * (1 - alpha) + new * alpha

def load_pupil_intrinsics(path: str, wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        intr_db = msgpack.unpack(fh, raw=True)

    w, h = wh
    key1 = f"({w}, {h})".encode("utf-8")
    key2 = f"({w},{h})".encode("utf-8")

    if key1 in intr_db:
        entry = intr_db[key1]
    elif key2 in intr_db:
        entry = intr_db[key2]
    else:
        available = [k.decode("utf-8", errors="ignore") if isinstance(k, (bytes, bytearray)) else str(k)
                     for k in intr_db.keys()]
        raise KeyError(f"No intrinsics for {wh} in {path}. Available keys: {available}")

    K = np.array(entry[b"camera_matrix"], dtype=np.float32).reshape(3, 3)
    D_raw = np.array(entry[b"dist_coefs"], dtype=np.float32).flatten()
    D = D_raw[:5].reshape(-1, 1).astype(np.float32)
    return K, D

def reprojection_errors(object_pts: np.ndarray, image_pts: np.ndarray,
                        rvec: np.ndarray, tvec: np.ndarray,
                        K: np.ndarray, D: np.ndarray) -> np.ndarray:
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    return np.linalg.norm(proj - image_pts, axis=1)

def epnp_ransac_iterative_refine(object_pts: np.ndarray, image_pts: np.ndarray,
                                K: np.ndarray, D: np.ndarray,
                                ransac_thresh: float = 4.0,
                                refine_thresh: float = 3.0,
                                min_inliers: int = 25,
                                max_iters: int = 5,
                                stable_eps: int = 2):
    if len(object_pts) < MIN_CORRESPONDENCES:
        return None

    obj = object_pts.astype(np.float32)
    img = image_pts.astype(np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=D,
        reprojectionError=ransac_thresh,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None

    prev_inlier_count = len(inliers)
    inlier_idx = inliers[:, 0]

    for _ in range(max_iters):
        errs = reprojection_errors(obj, img, rvec, tvec, K, D)
        new_inlier_idx = np.where(errs < refine_thresh)[0]

        if len(new_inlier_idx) < min_inliers:
            break

        if abs(len(new_inlier_idx) - prev_inlier_count) <= stable_eps:
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=obj[new_inlier_idx],
                imagePoints=img[new_inlier_idx],
                cameraMatrix=K,
                distCoeffs=D,
                rvec=rvec,
                tvec=tvec
            )
            inlier_idx = new_inlier_idx
            break

        ok2, rvec2, tvec2 = cv2.solvePnP(
            objectPoints=obj[new_inlier_idx],
            imagePoints=img[new_inlier_idx],
            cameraMatrix=K,
            distCoeffs=D,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok2:
            break

        rvec, tvec = rvec2, tvec2
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=obj[new_inlier_idx],
            imagePoints=img[new_inlier_idx],
            cameraMatrix=K,
            distCoeffs=D,
            rvec=rvec,
            tvec=tvec
        )
        inlier_idx = new_inlier_idx
        prev_inlier_count = len(inlier_idx)

    final_errs = reprojection_errors(obj, img, rvec, tvec, K, D)
    final_inliers = np.where(final_errs < refine_thresh)[0]
    if len(final_inliers) < min_inliers:
        return None

    med_err = float(np.median(final_errs[final_inliers]))
    return rvec, tvec, final_inliers, med_err


def extract_world_image(payload: dict):
    """
    Robustly extract image from pupil-core-network-client payload.

    We try common key variants:
      - raw BGR bytes: 'bgr', 'image', 'data', 'buffer', 'frame'
      - JPEG bytes: 'jpeg_buffer', 'jpg', 'jpeg'
    and width/height keys:
      - 'width','height' OR 'frame_width','frame_height' OR 'resolution'

    Returns: (bgr_img, (w,h))
    """
    # --- resolve width/height ---
    w = payload.get("width") or payload.get("frame_width")
    h = payload.get("height") or payload.get("frame_height")

    if (w is None or h is None) and "resolution" in payload:
        # sometimes resolution is (w,h)
        res = payload["resolution"]
        if isinstance(res, (tuple, list)) and len(res) == 2:
            w, h = int(res[0]), int(res[1])

    # --- pick the likely image-bytes field ---
    candidate_keys = [
        "bgr", "image", "data", "buffer", "frame",
        "bgr_buffer", "image_buffer",
        "jpeg_buffer", "jpg", "jpeg"
    ]

    img_bytes = None
    used_key = None
    for k in candidate_keys:
        if k in payload:
            img_bytes = payload[k]
            used_key = k
            break

    if img_bytes is None:
        raise KeyError(f"No image bytes key found in payload. Keys={list(payload.keys())}")

    if isinstance(img_bytes, memoryview):
        img_bytes = img_bytes.tobytes()

    # --- if jpeg, decode ---
    if used_key in ("jpeg_buffer", "jpg", "jpeg"):
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("JPEG decode failed.")
        h2, w2 = img.shape[:2]
        return img, (w2, h2)

    # --- otherwise assume raw bgr bytes ---
    if w is None or h is None:
        # try to infer square/known shapes: cannot safely infer -> force user to rely on width/height in payload
        raise KeyError(f"Width/height not found in payload. Keys={list(payload.keys())}")

    w, h = int(w), int(h)
    expected = w * h * 3
    if len(img_bytes) != expected:
        # sometimes stride/padding exists or format isn't actually bgr
        raise RuntimeError(
            f"Raw BGR size mismatch: got {len(img_bytes)} bytes, expected {expected} for {w}x{h}x3. "
            f"Payload keys={list(payload.keys())}"
        )

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)
    return img, (w, h)


# =========================
# MAIN
# =========================
def main():
    global PRINT_PAYLOAD_KEYS_ONCE

    # ---- Connect to Pupil Capture ----
    device = pcnc.Device(PUPIL_ADDR, PUPIL_PORT)
    device.send_notification({"subject": "frame_publishing.set_format", "format": PUPIL_FRAME_FORMAT})
    print(f"[INFO] Requested Pupil Frame Publisher format: {PUPIL_FRAME_FORMAT}")

    # ---- RealSense ----
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K_rgb = K_from_rs_intr(color_intr)

    print("[INFO] RealSense K_rgb:\n", K_rgb)
    print(f"[INFO] RealSense depth scale: {depth_scale} m/unit")

    brisk = cv2.BRISK_create(thresh=BRISK_THRESH, octaves=BRISK_OCTAVES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Throttle PnP
    pnp_period = 1.0 / PNP_HZ
    last_pnp_time = 0.0

    # State
    K_scene = None
    D_scene = None
    t_smooth = None
    last_good_inliers = 0
    last_median_err = None
    good_count = 0

    print("[INFO] Subscribing to", PUPIL_WORLD_TOPIC)

    with contextlib.suppress(KeyboardInterrupt):
        with device.subscribe_in_background(PUPIL_WORLD_TOPIC, buffer_size=1) as sub:
            while True:
                msg = sub.recv_new_message()
                payload = msg.payload

                if PRINT_PAYLOAD_KEYS_ONCE:
                    PRINT_PAYLOAD_KEYS_ONCE = False
                    print("[DEBUG] payload keys:", list(payload.keys()))
                    # also print a few example scalar fields
                    for k in list(payload.keys())[:10]:
                        v = payload[k]
                        if isinstance(v, (int, float, str)):
                            print(f"[DEBUG] {k}={v}")
                    print("[DEBUG] (not printing large buffers)")

                # ---- Decode world frame robustly ----
                scene_bgr, (w, h) = extract_world_image(payload)

                # Load intrinsics once
                if K_scene is None:
                    K_scene, D_scene = load_pupil_intrinsics(WORLD_INTRINSICS_PATH, (w, h))
                    print(f"[INFO] Loaded Pupil intrinsics for {w}x{h}")
                    print("[INFO] K_scene:\n", K_scene)

                # ---- RealSense aligned depth ----
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                rgb_bgr = np.asanyarray(color_frame.get_data())
                depth_u16 = np.asanyarray(depth_frame.get_data())
                depth_m = depth_u16.astype(np.float32) * depth_scale

                # ---- Gray ----
                scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
                rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

                now = time.time()
                do_pnp = (now - last_pnp_time) >= pnp_period

                if do_pnp:
                    last_pnp_time = now

                    kp_s, des_s = brisk.detectAndCompute(scene_gray, None)
                    kp_r, des_r = brisk.detectAndCompute(rgb_gray, None)

                    if des_s is None or des_r is None or len(kp_s) < 50 or len(kp_r) < 50:
                        last_good_inliers = 0
                        last_median_err = None
                        good_count = 0
                    else:
                        matches_knn = bf.knnMatch(des_s, des_r, k=2)
                        good = []
                        for m, n in matches_knn:
                            if m.distance < MATCH_RATIO * n.distance:
                                good.append(m)
                        good_count = len(good)

                        obj_pts, img_pts = [], []
                        for m in good:
                            u_s, v_s = kp_s[m.queryIdx].pt
                            u_r, v_r = kp_r[m.trainIdx].pt
                            ur, vr = int(round(u_r)), int(round(v_r))

                            if (ur < 0 or ur >= depth_m.shape[1] or
                                vr < 0 or vr >= depth_m.shape[0]):
                                continue

                            z = float(depth_m[vr, ur])
                            if z <= 0.0 or z > MAX_DEPTH_M:
                                continue

                            obj_pts.append(backproject(ur, vr, z, K_rgb))
                            img_pts.append([u_s, v_s])

                        if len(obj_pts) >= MIN_CORRESPONDENCES:
                            obj_pts = np.asarray(obj_pts, dtype=np.float32)
                            img_pts = np.asarray(img_pts, dtype=np.float32)

                            result = epnp_ransac_iterative_refine(
                                obj_pts, img_pts, K_scene, D_scene,
                                ransac_thresh=RANSAC_REPROJ_ERR,
                                refine_thresh=REFINE_REPROJ_ERR,
                                min_inliers=MIN_PNP_INLIERS,
                                max_iters=MAX_REFINE_ITERS,
                                stable_eps=INLIER_STABLE_EPS
                            )

                            if result is not None:
                                rvec, tvec, inlier_idx, med_err = result
                                last_good_inliers = len(inlier_idx)
                                last_median_err = med_err
                                t_smooth = ema(t_smooth, tvec.reshape(3), alpha=0.2)

                                print(f"[Pose] good={good_count} inliers={last_good_inliers} "
                                      f"median_err={med_err:.2f}px  t_smooth(m)={t_smooth}")
                            else:
                                last_good_inliers = 0
                                last_median_err = None
                        else:
                            last_good_inliers = 0
                            last_median_err = None

                # ---- Overlay ----
                overlay = scene_bgr.copy()
                ok_pose = last_good_inliers >= MIN_PNP_INLIERS
                status_color = (0, 255, 0) if ok_pose else (0, 0, 255)

                cv2.putText(overlay, f"BRISK good: {good_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                cv2.putText(overlay, f"Inliers: {last_good_inliers}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

                if last_median_err is not None:
                    cv2.putText(overlay, f"Median reproj err: {last_median_err:.2f}px", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                if t_smooth is not None:
                    cv2.putText(overlay, f"t(m): [{t_smooth[0]:.2f}, {t_smooth[1]:.2f}, {t_smooth[2]:.2f}]",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                cv2.imshow("Pupil World (BRISK + EPnP-RANSAC)", overlay)
                if SHOW_REALSENSE:
                    cv2.imshow("RealSense RGB", rgb_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

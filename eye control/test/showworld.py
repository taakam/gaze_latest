import argparse
import contextlib
import logging
import time

import numpy as np
import cv2
import pyrealsense2 as rs
import pupil_labs.pupil_core_network_client as pcnc


# -------------------------
# Decode Pupil frame.world
# -------------------------
def decode_frame(payload) -> np.ndarray:
    """
    Pupil frame publisher (format=bgr) typically sends:
      payload keys: topic, width, height, index, timestamp, format, __raw_data__
    where __raw_data__[0] is usually a numpy array (H,W,3) uint8.
    """
    if "__raw_data__" not in payload:
        raise KeyError(f"No __raw_data__ in payload. Keys={list(payload.keys())}")

    img = payload["__raw_data__"][0]

    # Sometimes wrapped as (img,) depending on msgpack decoding quirks
    if isinstance(img, tuple):
        img = img[0]

    # If it arrives as bytes/memoryview, reconstruct from width/height
    if isinstance(img, (bytes, memoryview)):
        w = int(payload["width"])
        h = int(payload["height"])
        fmt = payload.get("format", "bgr").lower()
        if isinstance(img, memoryview):
            img = img.tobytes()
        if fmt != "bgr":
            raise ValueError(f"Unsupported raw bytes format: {fmt}")
        img = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)

    # Ensure it's a numpy array (some decoders produce array-like)
    img = np.asarray(img)
    return img


def K_from_intr(intr) -> np.ndarray:
    return np.array(
        [[intr.fx, 0, intr.ppx],
         [0, intr.fy, intr.ppy],
         [0, 0, 1]],
        dtype=np.float32,
    )


def main(address: str, port: int, max_frame_rate_hz: int, show: bool):
    # -------------------------
    # Pupil Capture subscription
    # -------------------------
    device = pcnc.Device(address, port)

    # Ask Pupil to publish frames as BGR (as per your setting)
    device.send_notification({"subject": "frame_publishing.set_format", "format": "bgr"})

    print("[INFO] Subscribing to frame.world (Pupil Capture must be running)")
    sub_ctx = device.subscribe_in_background("frame.world", buffer_size=1)

    # -------------------------
    # RealSense setup
    # -------------------------
    COLOR_W, COLOR_H, FPS = 1280, 720, 30
    DEPTH_W, DEPTH_H = 848, 480

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)  # align depth -> color

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    K_rgb = K_from_intr(color_intr)

    print("[INFO] RealSense depth scale:", depth_scale)
    print("[INFO] RealSense K_rgb:\n", K_rgb)

    # -------------------------
    # BRISK + matcher
    # -------------------------
    brisk = cv2.BRISK_create(thresh=30, octaves=3)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ratio = 0.75

    tick_period = 1.0 / max_frame_rate_hz if max_frame_rate_hz > 0 else 0.0
    last_tick = 0.0

    try:
        with contextlib.suppress(KeyboardInterrupt):
            with sub_ctx as sub:
                while True:
                    # Optional throttle (sub already uses buffer_size=1, so it drops old frames)
                    if tick_period > 0:
                        now = time.time()
                        if now - last_tick < tick_period:
                            time.sleep(max(0.0, tick_period - (now - last_tick)))
                        last_tick = time.time()

                    # -------- Pupil world frame --------
                    msg = sub.recv_new_message()
                    payload = msg.payload
                    scene_bgr = decode_frame(payload)

                    # Safety: ensure expected shape
                    if scene_bgr.ndim != 3 or scene_bgr.shape[2] != 3:
                        print("[WARN] Unexpected Pupil frame shape:", scene_bgr.shape)
                        continue

                    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

                    # -------- RealSense frames --------
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)

                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue

                    rgb_bgr = np.asanyarray(color_frame.get_data())
                    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)

                    # -------- BRISK detect+match --------
                    kp_s, des_s = brisk.detectAndCompute(scene_gray, None)
                    kp_r, des_r = brisk.detectAndCompute(rgb_gray, None)

                    good = []
                    if des_s is not None and des_r is not None and len(kp_s) > 30 and len(kp_r) > 30:
                        matches_knn = bf.knnMatch(des_s, des_r, k=2)
                        for m, n in matches_knn:
                            if m.distance < ratio * n.distance:
                                good.append(m)

                    # Depth sanity check for some matches
                    depth_ok = 0
                    for m in good[:50]:
                        u_r, v_r = kp_r[m.trainIdx].pt
                        z = depth_frame.get_distance(int(round(u_r)), int(round(v_r)))
                        if 0.0 < z < 4.0:
                            depth_ok += 1

                    print(
                        f"[BRISK] scene_kp={len(kp_s):4d} rgb_kp={len(kp_r):4d} "
                        f"good={len(good):4d} depth_ok(first50)={depth_ok:3d}"
                    )

                    if show:
                        overlay = scene_bgr.copy()
                        cv2.putText(
                            overlay,
                            f"good matches: {len(good)}  depth_ok(50): {depth_ok}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("Pupil frame.world", overlay)
                        cv2.imshow("RealSense RGB", rgb_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break

    finally:
        pipeline.stop()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=50020)
    parser.add_argument("-fps", "--max-frame-rate", type=int, default=10)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    main(args.address, args.port, args.max_frame_rate, args.show)

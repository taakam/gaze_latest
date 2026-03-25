import argparse
import contextlib
import logging
import time
from matplotlib.pyplot import show
import numpy as np
import pupil_labs.pupil_core_network_client as pupil
import cv2 
import pyrealsense2 as rs


def decode_frame(payload):
    fmt = payload.get("format", "bgr").lower()

    if "__raw_data__" not in payload:
        raise KeyError(f"No __raw_data__ in payload. Keys={list(payload.keys())}")

    img = payload["__raw_data__"][0]

    # Some versions may wrap as (array, ) or similar
    if isinstance(img, tuple):
        img = img[0]

    # If it somehow arrives as bytes/memoryview, fall back to frombuffer
    if isinstance(img, (bytes, memoryview)):
        w = int(payload["width"])
        h = int(payload["height"])
        if isinstance(img, memoryview):
            img = img.tobytes()
        if fmt != "bgr":
            raise ValueError(f"Unsupported raw bytes format: {fmt}")
        img = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)

    # At this point it should be a numpy image
    return img

def K_from_intr(intr):
    return np.array([[intr.fx, 0, intr.ppx], 
                     [0, intr.fy, intr.ppy], 
                     [0, 0, 1]], dtype=np.float32)

def main(address: str, port: int, max_frame_rate_hz: int):
    #Pupil capture stream 
    device = pupil.Device(address, port)
    device.send_notification(
        {"subject": "frame_publishing.set_format", "format": "bgr"}
    )
    world_sub = device.subscribe("frame.world", buffer_size=1)
    
    #RealSense stream
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.bgr8, 30)    cfg.enable_stream(rs.stream.color, 848,480, rs.format.z16, 30)

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K_rgb = K_from_intr(color_intr)
    
    print("[INFO] RealSense depth scale:", depth_scale)
    print("[INFO] RealSense K_rgb:\n", K_rgb)
    
    #BRISK + matcher 
    brisk = cv2.BRISK_create(tresh=30, octaves=3)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ratio = 0.75
    
    last_tick = time.time()
    
    try:
        with contextlib.suppress(KeyboardInterrupt):
            with world_sub:
                while True:
                    #throttle loop (optional)
                    now = time.time()
                    dt = now - last_tick
                    if dt < 1.0 / max_frame_rate_hz:
                        time.sleep(max(0.0, 1.0 / max_frame_rate_hz - dt))
                    last_tick = time.time()
                    
                    #get latest pupil world frame
                    msg = world_sub.recv_new_message()
                    scene_bgr = decode_frame(msg.paylod)
                    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)    
                    
                    
                    #get RealSense frames
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue
                    
                    rgb_bgr = np.asanyarray(color_frame.get_data())
                    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
                    
                    #BRISK detect+compute 
                    kp_s, des_s = brisk.detectAndCompute(scene_gray, None)
                    kp_r, des_r = brisk.detectAndCompute(rgb_gray, None)
                    
                    good = []
                    if des_s is not None and des_r is not None and len(kp_s) > 30 and len(kp_r) > 30:
                        matches_knn = bf.knnMatch(des_s, des_r, k=2)
                        for m, n in matches_knn:
                            if m.distance < ratio * n.distance:
                                good.append(m)
                    
                    # Note : depth is aligned to color here, so (u, v) from rgb frame is valid for depth_frame.
                    depth_ok = 0 
                    if good:
                        for m in good[:50]:
                            u_r, v_r = kp_r[m.trainIdx].pt
                            z = depth_frame.get_distance(int(u_r), int(v_r))
                            if 0.0 < z < 4.0:
                                depth_ok += 1
                    
                    print(f"[BRISK] scene_kp={len(kp_s)}, rgb_kp={len(kp_r)} "
                            f"matches={len(good)}, depth_ok={depth_ok}")
                    
                    if show:
                        overlay = scene_bgr.copy()
                        cv2.putText(overlay, f"good matches: {len(good)}", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.imshow("Pupil frame.world", overlay)
                        cv2.imshow("RealSense RGB", rgb_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                        
                    
                    message = world_sub.recv_new_message()
                    payload = message.payload
                    img = decode_frame(payload)
                    
    finally:
        pipeline.stop()
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

    main(args.address, args.port, args.max_frame_rate)
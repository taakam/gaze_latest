import pyrealsense2 as rs
import numpy as np
import cv2
import sys

# ---------- Config ----------
COLOR_W, COLOR_H, FPS = 1280, 720, 30
DEPTH_W, DEPTH_H = 848, 480

clicked = None  # (x, y)

def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (x, y)

def main():
    global clicked

    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)

    profile = pipeline.start(config)
    
    color_stream = profile.get_stream(rs.stream.color)
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()

    print("[INFO] RealSense COLOR intrinsics:")
    print("  width,height:", color_intr.width, color_intr.height)
    print("  fx, fy:", color_intr.fx, color_intr.fy)
    print("  cx, cy:", color_intr.ppx, color_intr.ppy)
    print("  model:", color_intr.model)
    print("  coeffs:", list(color_intr.coeffs))
    
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
    
    print("[INFO] RealSense DEPTH intrinsics:")
    print("  width,height:", depth_intr.width, depth_intr.height)
    print("  fx, fy:", depth_intr.fx, depth_intr.fy)
    print("  cx, cy:", depth_intr.ppx, depth_intr.ppy)
    print("  model:", depth_intr.model)
    print("  coeffs:", list(depth_intr.coeffs))


    # Align depth to color frame
    align = rs.align(rs.stream.color)

    # Get depth scale (to convert depth units to meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale} meters/unit")

    cv2.namedWindow("RealSense Color", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("RealSense Color", on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())  # uint16

            # Optional: show a depth colormap in another window
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )

            # If user clicked, read depth at that pixel (aligned to color)
            if clicked is not None:
                x, y = clicked
                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z_units = depth[y, x]
                    z_m = z_units * depth_scale
                    print(f"[DEPTH] pixel=({x},{y}) raw={z_units} -> {z_m:.3f} m")

                    # Draw marker
                    cv2.circle(color, (x, y), 6, (0, 255, 0), 2)
                    cv2.putText(color, f"{z_m:.2f}m", (x+10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                clicked = None

            cv2.imshow("RealSense Color", color)
            cv2.imshow("RealSense Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('s'):
                cv2.imwrite("realsense_color.png", color)
                print("[SAVE] realsense_color.png")


    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

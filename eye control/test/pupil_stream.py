import argparse
import contextlib
import logging
import time
import numpy as np
import cv2
import pupil_labs.pupil_core_network_client as pcnc


def decode_frame_pcnc(payload):
    if "__raw_data__" not in payload:
        raise KeyError(f"Missing __raw_data__. Keys={list(payload.keys())}")

    w = int(payload["width"])
    h = int(payload["height"])
    fmt = str(payload.get("format", "bgr")).lower()

    img = payload["__raw_data__"][0]

    if isinstance(img, tuple):
        img = img[0]

    if isinstance(img, np.ndarray):
        return img

    if isinstance(img, memoryview):
        img = img.tobytes()

    if isinstance(img, (bytes, bytearray)):
        if fmt != "bgr":
            raise ValueError(f"Got raw bytes but unsupported format={fmt}")
        return np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)

    raise TypeError(f"Unsupported __raw_data__ type: {type(img)}")


def main(address: str, port: int, max_frame_rate_hz: int):
    device = pcnc.Device(address, port)
    device.send_notification({"subject": "frame_publishing.set_format", "format": "bgr"})

    with contextlib.suppress(KeyboardInterrupt):
        with device.subscribe_in_background("frame.world", buffer_size=1) as sub:
            while True:
                message = sub.recv_new_message()

                # ✅ decode the actual image
                img = decode_frame_pcnc(message.payload)
                
                cv2.imshow("frame.world", img)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

                # headless confirmation
                print(img.shape, img.dtype, "index:", message.payload.get("index"))

                time.sleep(1 / max_frame_rate_hz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=50020)
    parser.add_argument("-fps", "--max-frame-rate", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    main(args.address, args.port, args.max_frame_rate)

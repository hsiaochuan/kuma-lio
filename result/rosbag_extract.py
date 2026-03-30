#!/usr/bin/env python3
import argparse
import os
import sys


# ─────────────────────────────────────────────
# Common utilities
# ─────────────────────────────────────────────

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ts_to_sec(secs: int, nsecs: int) -> float:
    return secs + nsecs * 1e-9

def extract_with_rosbags(bag_path: str, output_dir: str,
                         imu_topics: list, image_topics: list):
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.serde import deserialize_cdr, ros1_to_cdr
        import cv2
        import numpy as np
    except ImportError as e:
        sys.exit(f"[ERROR] Missing dependency: {e}\n"
                 "Please run: pip install rosbags opencv-python numpy")

    IMU_TYPE = "sensor_msgs/msg/Imu"
    IMAGE_TYPE = "sensor_msgs/msg/Image"
    COMP_TYPE = "sensor_msgs/msg/CompressedImage"

    with Reader(bag_path) as reader:
        all_conns = list(reader.connections)

        if not imu_topics:
            imu_topics = [c.topic for c in all_conns
                          if "Imu" in c.msgtype]
        if not image_topics:
            image_topics = [c.topic for c in all_conns
                            if c.msgtype in (IMAGE_TYPE, COMP_TYPE,
                                             "sensor_msgs/msg/CompressedImage")]

        print(f"[INFO] IMU topics: {imu_topics}")
        print(f"[INFO] Image topics: {image_topics}")

        # ── IMU ──────────────────────────────
        imu_files = {}
        for topic in imu_topics:
            safe = topic.lstrip("/").replace("/", "_")
            path = os.path.join(output_dir, f"{safe}.txt")
            ensure_dir(os.path.dirname(path))
            f = open(path, "w")
            f.write("# timestamp(s)  ax ay az  wx wy wz  qx qy qz qw\n")
            imu_files[topic] = f
            print(f"[INFO] IMU → {path}")

        # ── Image ────────────────────────────
        img_count = 0
        img_dirs = {}
        for topic in image_topics:
            safe = topic.lstrip("/").replace("/", "_")
            d = os.path.join(output_dir, safe)
            ensure_dir(d)
            img_dirs[topic] = d
            print(f"[INFO] Image → {d}/")

        wanted_topics = set(imu_topics + image_topics)
        conns = [c for c in all_conns if c.topic in wanted_topics]

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            ts = timestamp * 1e-9  # nanoseconds → seconds
            topic = conn.topic

            try:
                cdr = ros1_to_cdr(rawdata, conn.msgtype)
                msg = deserialize_cdr(cdr, conn.msgtype)
            except Exception as e:
                print(f"[WARN] Deserialization failed ({topic}): {e}")
                continue

            if topic in imu_files:
                a = msg.linear_acceleration
                w = msg.angular_velocity
                q = msg.orientation
                imu_files[topic].write(
                    f"{ts:.9f}  "
                    f"{a.x:.6f} {a.y:.6f} {a.z:.6f}  "
                    f"{w.x:.6f} {w.y:.6f} {w.z:.6f}  "
                    f"{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n"
                )

            elif topic in img_dirs:
                try:
                    if conn.msgtype == COMP_TYPE:
                        buf = np.frombuffer(msg.data, dtype=np.uint8)
                        cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    else:
                        enc = msg.encoding.lower()
                        arr = np.frombuffer(msg.data, dtype=np.uint8)
                        arr = arr.reshape(msg.height, msg.step)[:, :msg.width *
                                                                    (3 if "rgb" in enc or "bgr" in enc else 1)]
                        if enc in ("rgb8", "rgb"):
                            arr = arr.reshape(msg.height, msg.width, 3)
                            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        elif enc in ("bgr8", "bgr"):
                            cv_img = arr.reshape(msg.height, msg.width, 3)
                        elif enc in ("mono8", "8uc1"):
                            cv_img = arr.reshape(msg.height, msg.width)
                        elif enc in ("mono16", "16uc1"):
                            arr16 = np.frombuffer(msg.data, dtype=np.uint16)
                            cv_img = (arr16.reshape(msg.height, msg.width)
                                      >> 8).astype(np.uint8)
                        else:
                            raise ValueError(f"Unsupported encoding: {enc}")

                    fname = os.path.join(img_dirs[topic],
                                         f"{ts:.6f}.jpg")
                    cv2.imwrite(fname, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    img_count = img_count + 1
                except Exception as e:
                    print(f"[WARN] Image processing failed ({topic}): {e}")

    for f in imu_files.values():
        f.close()
    print(f"\n[DONE] Extracted {img_count} images, "
          f"IMU data written to {len(imu_files)} files.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract IMU (→ TXT) and images (→ JPG) from ROS bag"
    )
    parser.add_argument("--bag", required=True, help="Input .bag file path")
    parser.add_argument("--out", default="output", help="Output directory (default ./output)")
    parser.add_argument("--imu", nargs="*", default=[],
                        help="IMU topic name, leave empty for auto-detection")
    parser.add_argument("--image", nargs="*", default=[],
                        help="Image topic name, leave empty for auto-detection")
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        sys.exit(f"[ERROR] File not found: {args.bag}")

    ensure_dir(args.out)
    print(f"[INFO] bag file: {args.bag}")
    print(f"[INFO] output directory: {os.path.abspath(args.out)}")
    extract_with_rosbags(args.bag, args.out, args.imu, args.image)


if __name__ == "__main__":
    main()

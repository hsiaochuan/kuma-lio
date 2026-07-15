#!/usr/bin/env python3
"""
extract_bag_images.py - extract images from a ROS bag file

Supports:
  - sensor_msgs/Image
  - sensor_msgs/CompressedImage
  - multiple image topics
  - time-range filtering (--start / --duration)
  - skip N frames to speed up extraction (--skip)
  - name output images by message timestamp
  - one subfolder per topic

Dependencies:
  pip install tqdm          # optional, for progress bar
  # Requires ROS environment: rosbag, rospy, cv_bridge, sensor_msgs, opencv-python
"""

import os
import sys
import argparse
import numpy as np

# -- dependency check ---------------------------------------------------------
try:
    import rosbag
    import rospy
    from cv_bridge import CvBridge
    import cv2
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Make sure ROS is installed and you have sourced setup.bash, and that cv_bridge / OpenCV are available.")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Supported image message types
IMAGE_MSG_TYPES = {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}


# -- utility functions --------------------------------------------------------

def sanitize_topic(topic: str) -> str:
    """
    Convert a ROS topic name to a safe directory name.
    /camera/image_raw  ->  camera__image_raw
    """
    return topic.strip("/").replace("/", "__")


def stamp_to_str(stamp) -> str:
    """
    Convert rospy.Time / Header.stamp to '0000000000_000000000'.
    Zero-padding ensures filename lexical order == chronological order.
    """
    return f"{stamp.secs:010d}_{stamp.nsecs:09d}"


def decode_image(msg, bridge: CvBridge):
    """
    Convert sensor_msgs/Image or sensor_msgs/CompressedImage to an OpenCV BGR image.
    Returns (cv_img, extension).
    """
    msg_type = type(msg).__name__

    if msg_type == "CompressedImage":
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        cv_img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            raise ValueError("cv2.imdecode failed, data may be corrupted")
        return cv_img, "jpg"

    # sensor_msgs/Image
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    elif enc in ("mono16", "16uc1"):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
    elif "rgb" in enc:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    else:
        # bgr8 / bayer_* / yuv etc. - convert to bgr8
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    return cv_img, "png"


# -- core logic ---------------------------------------------------------------

def extract_images(
        bag_file: str,
        output_dir: str,
        topics: list = None,
        start_offset: float = None,
        duration: float = None,
        skip: int = 0,
) -> None:
    """
    Extract images from a bag file.

    Parameters
    ----
    bag_file     : path to the bag file
    output_dir   : output root directory
    topics       : list of topics to extract; None means all image topics
    start_offset : start offset relative to bag start (seconds); None means from start
    duration     : duration to extract (seconds); None means until bag end
    skip         : number of frames to skip; 0 saves all, N saves 1 out of N+1 frames
    """
    bridge = CvBridge()

    print(f"\n[INFO] Opening bag: {bag_file}")

    with rosbag.Bag(bag_file, "r") as bag:

        # -- 1. bag basic info ------------------------------------------------
        bag_t0 = bag.get_start_time()
        bag_t1 = bag.get_end_time()
        print(f"[INFO] bag time range : {bag_t0:.3f}  ~  {bag_t1:.3f}  "
              f"(duration {bag_t1 - bag_t0:.2f} s)")

        # -- 2. compute extraction time window -------------------------------
        t_start = rospy.Time.from_sec(bag_t0 + (start_offset or 0.0))
        if duration is not None:
            t_end = rospy.Time.from_sec(t_start.to_sec() + duration)
        else:
            t_end = rospy.Time.from_sec(bag_t1)

        # ensure t_end does not exceed bag end
        t_end = rospy.Time.from_sec(min(t_end.to_sec(), bag_t1))

        print(f"[INFO] extraction time window : {t_start.to_sec():.3f}  ~  {t_end.to_sec():.3f}  "
              f"(duration {t_end.to_sec() - t_start.to_sec():.2f} s)")

        # -- 3. discover all image topics in bag -------------------------------
        type_info = bag.get_type_and_topic_info().topics
        bag_img_topics = [
            t for t, info in type_info.items()
            if info.msg_type in IMAGE_MSG_TYPES
        ]

        if not bag_img_topics:
            print("[WARN] no image topics found in bag, exiting.")
            return

        print(f"[INFO] image topics in bag ({len(bag_img_topics)}): {bag_img_topics}")

        # -- 4. determine topics to extract ----------------------------------
        if topics:
            selected = []
            for t in topics:
                if t in bag_img_topics:
                    selected.append(t)
                else:
                    print(f"[WARN] topic {t!r} does not exist or is not an image topic, skipped")
            if not selected:
                print("[ERROR] no valid topics specified, exiting.")
                return
        else:
            selected = bag_img_topics

        print(f"[INFO] topics to extract ({len(selected)}): {selected}")

        # -- 5. create output subdirectories ----------------------------------
        for topic in selected:
            sub_dir = os.path.join(output_dir, sanitize_topic(topic))
            os.makedirs(sub_dir, exist_ok=True)
            print(f"[INFO] output directory: {sub_dir}")

        # -- 6. iterate messages and save images ------------------------------
        # rough estimate of total messages (for progress bar, ignores time filter)
        total_est = sum(type_info[t].message_count for t in selected)

        counts = {t: 0 for t in selected}
        errors = {t: 0 for t in selected}
        frame_indices = {t: 0 for t in selected}  # per-topic frame counters
        processed = 0

        msg_iter = bag.read_messages(
            topics=selected,
            start_time=t_start,
            end_time=t_end,
        )

        if HAS_TQDM:
            msg_iter = tqdm(msg_iter, total=total_est,
                            unit="frames", desc="extracting", dynamic_ncols=True)

        for topic, msg, bag_ts in msg_iter:
            try:
                # check whether to skip this frame
                if skip > 0 and frame_indices[topic] % (skip + 1) != 0:
                    frame_indices[topic] += 1
                    processed += 1
                    continue

                cv_img, ext = decode_image(msg, bridge)

                # prefer the message header timestamp; fall back to bag time
                stamp = (msg.header.stamp
                         if msg.header.stamp.to_sec() > 0
                         else bag_ts)

                filename = f"{stamp_to_str(stamp)}.{ext}"
                out_path = os.path.join(output_dir, sanitize_topic(topic), filename)

                cv2.imwrite(out_path, cv_img)
                counts[topic] += 1
                frame_indices[topic] += 1

            except Exception as exc:
                errors[topic] += 1
                frame_indices[topic] += 1
                ts_str = f"{bag_ts.to_sec():.3f}"
                if HAS_TQDM:
                    tqdm.write(f"[WARN] {topic} @ {ts_str}s failed to process: {exc}")
                else:
                    print(f"\n[WARN] {topic} @ {ts_str}s failed to process: {exc}")

            processed += 1
            if not HAS_TQDM and processed % 100 == 0:
                print(f"[INFO] processed {processed} frames ...")

    # -- 7. summary output --------------------------------------------------
    total_saved = sum(counts.values())
    total_err   = sum(errors.values())
    bar = "═" * 58

    print(f"\n╔{bar}╗")
    print(f"║{'Extraction complete':^58}║")
    print(f"╠{bar}╣")
    for topic in selected:
        folder = os.path.join(output_dir, sanitize_topic(topic))
        print(f"║  topic : {topic}")
        print(f"║  saved : {counts[topic]} frames   failed: {errors[topic]} frames")
        print(f"║  folder: {folder}")
        print(f"╠{bar}╣")
    print(f"║  total saved: {total_saved} frames   total failed: {total_err} frames{' ' * (30 - len(str(total_saved)) - len(str(total_err)))}║")
    print(f"╚{bar}╝\n")


# -- CLI entrypoint ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="extract_bag_images.py",
        description="Extract images from a ROS bag, name by timestamp and store per-topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # extract all image topics from the whole bag
  python extract_bag_images.py record.bag ./output

  # extract only specified topics
  python extract_bag_images.py record.bag ./output \\
      -t /cam0/image_raw /cam1/image_raw

  # start at 10s and extract for 30s
  python extract_bag_images.py record.bag ./output --start 10 --duration 30

  # save 1 in every 3 frames (skip 2 frames)
  python extract_bag_images.py record.bag ./output --skip 2

  # combine: topics + time range + skip
  python extract_bag_images.py record.bag ./output \\
      -t /camera/image_raw --start 5 --duration 60 --skip 4
        """,
    )

    parser.add_argument(
        "--bag_file",
        default="/mnt/data/home/hsiaochuan/data/MCD_VIRAL/raw/camera/tuhh_day_04_d455t.bag",
        help="input ROS bag file path (.bag)",
    )
    parser.add_argument(
        "--output_dir",
        help="output root directory (will be created if missing)",
    )
    parser.add_argument(
        "-t", "--topics",
        nargs="+",
        default=["/d455t/color/image_raw"],
        metavar="TOPIC",
        help="image topics to extract (can specify multiple); default extracts the configured defaults",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        metavar="SEC",
        help="start offset relative to bag start (seconds), default 0 (from start)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SEC",
        help="duration to extract (seconds), default until bag end",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=30,
        metavar="N",
        help="save 1 out of N+1 frames (skip N). Default 0 saves all, useful to speed up extraction",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # parameter validation
    if not os.path.isfile(args.bag_file):
        print(f"[ERROR] bag file does not exist: {args.bag_file}")
        sys.exit(1)

    if args.start is not None and args.start < 0:
        print("[ERROR] --start cannot be negative")
        sys.exit(1)

    if args.duration is not None and args.duration <= 0:
        print("[ERROR] --duration must be > 0")
        sys.exit(1)

    if args.skip < 0:
        print("[ERROR] --skip cannot be negative")
        sys.exit(1)
    if args.output_dir is None:
        args.output_dir = args.bag_file + "_extracted"
    os.makedirs(args.output_dir, exist_ok=True)

    extract_images(
        bag_file=args.bag_file,
        output_dir=args.output_dir,
        topics=args.topics,
        start_offset=args.start,
        duration=args.duration,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
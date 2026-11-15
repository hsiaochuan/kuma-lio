#!/usr/bin/env python3
"""Extract MARS-LVIG ground truth as TUM files.

The dataset has no ground-truth files: the reference is the on-board RTK
stream inside every bag.  This script takes /dji_osdk_ros/rtk_position
(NavSatFix, 5 Hz) as it is -- the RTK antenna position, with its recorded
stamps -- converts it to a local ENU frame anchored at the first valid fix,
and writes it in TUM format with an identity orientation.

Usage:
  source /opt/ros/noetic/setup.bash
  python3 extract_mars_lvig_gt.py /media/hsiaochuan/hsiaochuan-data/MARS-LVIG \
      -o /media/hsiaochuan/hsiaochuan-data/MARS-LVIG/ground-truth --overwrite
"""
import argparse
import glob
import math
import os
import sys

try:
    import rosbag
except ImportError:
    sys.exit("rosbag not importable -- source a ROS 1 setup.bash first "
             "(e.g. `source /opt/ros/noetic/setup.bash`)")

RTK_POS_TOPIC = "/dji_osdk_ros/rtk_position"

# WGS84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(x, y, z, ref_lat_deg, ref_lon_deg, ref_ecef):
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    dx = x - ref_ecef[0]
    dy = y - ref_ecef[1]
    dz = z - ref_ecef[2]
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return e, n, u


def stamp_of(msg, bag_time):
    """The header stamp as recorded, else the bag record time."""
    header = getattr(msg, "header", None)
    if header is not None and (header.stamp.secs or header.stamp.nsecs):
        return header.stamp.to_sec()
    return bag_time.to_sec()


def extract(bag_path, out_path):
    name = os.path.splitext(os.path.basename(bag_path))[0]
    with rosbag.Bag(bag_path, "r") as bag:
        if RTK_POS_TOPIC not in bag.get_type_and_topic_info().topics:
            print("  [skip] %s: no %s" % (name, RTK_POS_TOPIC))
            return False
        fixes = [(stamp_of(msg, t), msg) for _, msg, t in bag.read_messages(topics=[RTK_POS_TOPIC])]
    fixes.sort(key=lambda f: f[0])

    rows = []
    ref = None
    n_invalid, n_dup = 0, 0
    last_ts = None
    for ts, msg in fixes:
        if not all(map(math.isfinite, (msg.latitude, msg.longitude, msg.altitude))) or \
                (msg.latitude == 0.0 and msg.longitude == 0.0):
            n_invalid += 1
            continue
        if last_ts is not None and ts <= last_ts:
            n_dup += 1
            continue
        last_ts = ts
        if ref is None:
            ref = (msg.latitude, msg.longitude, msg.altitude, lla_to_ecef(msg.latitude, msg.longitude, msg.altitude))
        x, y, z = lla_to_ecef(msg.latitude, msg.longitude, msg.altitude)
        rows.append((ts,) + ecef_to_enu(x, y, z, ref[0], ref[1], ref[3]))

    if len(rows) < 2:
        print("  [skip] %s: only %d valid fixes" % (name, len(rows)))
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# MARS-LVIG ground truth extracted from %s\n" % os.path.basename(bag_path))
        f.write("# source topic: %s (RTK antenna position, identity orientation)\n" % RTK_POS_TOPIC)
        f.write("# local ENU frame anchored at lat=%.9f lon=%.9f alt=%.4f\n" % ref[:3])
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, e, n, u in rows:
            f.write("%.9f %.6f %.6f %.6f 0.0 0.0 0.0 1.0\n" % (ts, e, n, u))

    length = sum(math.dist(rows[i][1:4], rows[i - 1][1:4]) for i in range(1, len(rows)))
    print("  [ok]   %s -> %s" % (name, out_path))
    print("         %d poses, %.1f s, path %.1f m" % (len(rows), rows[-1][0] - rows[0][0], length))
    if n_invalid or n_dup:
        print("         dropped: %d invalid, %d non-monotonic" % (n_invalid, n_dup))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="bag files, or directories to scan for *.bag")
    ap.add_argument("-o", "--out-dir", required=True,
                    help="directory the TUM ground-truth files are written to")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract bags whose output file already exists")
    args = ap.parse_args()

    bags = []
    for item in args.inputs:
        if os.path.isdir(item):
            bags.extend(sorted(glob.glob(os.path.join(item, "*.bag"))))
        else:
            bags.append(item)
    if not bags:
        sys.exit("no bags found")

    n_ok = 0
    for bag_path in bags:
        name = os.path.splitext(os.path.basename(bag_path))[0]
        out_path = os.path.join(args.out_dir, name + ".txt")
        print("%s" % bag_path)
        if os.path.exists(out_path) and not args.overwrite:
            print("  [skip] %s already exists (use --overwrite)" % out_path)
            continue
        if extract(bag_path, out_path):
            n_ok += 1
    print("\nwrote %d ground-truth file(s) to %s" % (n_ok, args.out_dir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
GPS_POS_TOPIC = "/dji_osdk_ros/gps_position"
UBLOX_LLA_TOPIC = "/ublox_driver/receiver_lla"

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
    """Prefer the header stamp: in MARS-LVIG it is GPS-disciplined and shares a
    clock with /livox/lidar, while the bag record time carries write jitter."""
    header = getattr(msg, "header", None)
    if header is not None and (header.stamp.secs or header.stamp.nsecs):
        return header.stamp.to_sec()
    return bag_time.to_sec()


def extract(bag_path, out_path):
    name = os.path.splitext(os.path.basename(bag_path))[0]
    with rosbag.Bag(bag_path, "r") as bag:
        available = set(bag.get_type_and_topic_info().topics.keys())

        pos_topic = None
        for cand in (RTK_POS_TOPIC, GPS_POS_TOPIC, UBLOX_LLA_TOPIC):
            if cand in available:
                pos_topic = cand
                break
        if pos_topic is None:
            print("  [skip] %s: no NavSatFix ground-truth topic" % name)
            return False
        if pos_topic != RTK_POS_TOPIC:
            print("  [warn] %s: %s missing, falling back to %s (lower accuracy)"
                  % (name, RTK_POS_TOPIC, pos_topic))

        fixes = [(stamp_of(msg, t), msg)
                 for _, msg, t in bag.read_messages(topics=[pos_topic])]

    if not fixes:
        print("  [skip] %s: %s carries no messages" % (name, pos_topic))
        return False

    fixes.sort(key=lambda p: p[0])

    rows = []
    ref = None
    n_nan, n_dup = 0, 0
    last_ts = None
    for ts, msg in fixes:
        if not all(map(math.isfinite, (msg.latitude, msg.longitude, msg.altitude))):
            n_nan += 1
            continue
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            n_nan += 1
            continue
        if last_ts is not None and ts <= last_ts:
            n_dup += 1
            continue
        last_ts = ts

        if ref is None:
            ref = (msg.latitude, msg.longitude, msg.altitude,
                   lla_to_ecef(msg.latitude, msg.longitude, msg.altitude))
        x, y, z = lla_to_ecef(msg.latitude, msg.longitude, msg.altitude)
        rows.append((ts,) + ecef_to_enu(x, y, z, ref[0], ref[1], ref[3]))

    if len(rows) < 2:
        print("  [skip] %s: only %d valid fixes" % (name, len(rows)))
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# MARS-LVIG ground truth extracted from %s\n" % os.path.basename(bag_path))
        f.write("# source topic: %s (position only, identity orientation)\n" % pos_topic)
        f.write("# local ENU frame anchored at lat=%.9f lon=%.9f alt=%.4f\n"
                % (ref[0], ref[1], ref[2]))
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, e, n, u in rows:
            f.write("%.9f %.6f %.6f %.6f 0.0 0.0 0.0 1.0\n" % (ts, e, n, u))

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    zs = [r[3] for r in rows]
    length = sum(math.dist(rows[i][1:4], rows[i - 1][1:4]) for i in range(1, len(rows)))
    print("  [ok]   %s -> %s" % (name, out_path))
    print("         %d poses, %.1f s, path %.1f m, extent E %.1f m / N %.1f m / U %.1f m"
          % (len(rows), rows[-1][0] - rows[0][0], length,
             max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    if n_nan or n_dup:
        print("         dropped: %d invalid, %d non-monotonic" % (n_nan, n_dup))
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

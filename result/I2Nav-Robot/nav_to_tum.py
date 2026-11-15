#!/usr/bin/env python3
"""i2Nav-Robot *_groundtruth.nav -> TUM trajectory for slam_test.py.

The .nav rows are
    sow  north east down [m]  vn ve vd [m/s]  roll pitch yaw [deg]
with sow the GPS seconds of week, the position in a local NED frame and the pose that of the ADIS16465
(the IMU of the bags), so no lever arm is needed.  The output rows are
    unix_time x y z qx qy qz qw
with the same NED position and the quaternion of R_n_b = Rz(yaw) Ry(pitch) Rx(roll).  slam_test.py aligns
with SE(3) before the translation APE, so the NED world needs no rotation into the z-up world of the SLAM.

The GPS week is not in the file: pass --week, or --bag to read it from the first message of the bag.

    python3 nav_to_tum.py groundtruth/building00_groundtruth.nav groundtruth/building00.txt --bag building00.bag
"""
import argparse
import math

GPS_EPOCH_UNIX = 315964800.0  # 1980-01-06 00:00:00 UTC
LEAP_SECONDS = 18.0           # GPS - UTC since 2017
WEEK_SEC = 604800.0


def week_of_unix(unix_time: float) -> int:
    return int((unix_time - GPS_EPOCH_UNIX + LEAP_SECONDS) // WEEK_SEC)


def week_of_bag(bag_fname: str) -> int:
    import rosbag
    with rosbag.Bag(bag_fname) as bag:
        return week_of_unix(bag.get_start_time())


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """Quaternion (x, y, z, w) of Rz(yaw) Ry(pitch) Rx(roll), angles in radians."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def convert(nav_fname: str, tum_fname: str, week: int) -> int:
    offset = GPS_EPOCH_UNIX + week * WEEK_SEC - LEAP_SECONDS
    rows = 0
    with open(nav_fname) as fin, open(tum_fname, "w") as fout:
        for line in fin:
            v = line.split()
            if len(v) < 10:
                continue
            sow, n, e, d = (float(x) for x in v[0:4])
            roll, pitch, yaw = (math.radians(float(x)) for x in v[7:10])
            qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
            fout.write(f"{sow + offset:.6f} {n:.6f} {e:.6f} {d:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nav", help="*_groundtruth.nav")
    parser.add_argument("tum", help="output TUM file, <sequence>.txt for slam_test.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--week", type=int, help="GPS week")
    group.add_argument("--bag", help="the sequence's bag; its start time gives the GPS week")
    args = parser.parse_args()

    week = args.week if args.week is not None else week_of_bag(args.bag)
    rows = convert(args.nav, args.tum, week)
    print(f"{args.tum}: {rows} poses, GPS week {week}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert TartanGround data (downloaded by download.py) into a ROS1 bag.

For each trajectory the following is written:
  /<camera>/image_raw         sensor_msgs/Image        (image_<camera>/*.png @ 1/time_step)
  /lidar                      sensor_msgs/PointCloud2  (lidar/*.ply, if present)
  /imu                        sensor_msgs/Imu          (imu/acc.npy + gyro.npy @ imu_fps, if present)

Image timestamps are generated from the trajectory metadata time_step
(TartanGround captures all cameras synchronously); timestamps start
at --t0 (TartanGround times start at 0, which rosbag tools dislike).

Example:
  python3 to_rosbag.py --env GothicIsland --traj P1000
  python3 to_rosbag.py --env GothicIsland --traj P1000 --compressed -o gothic_P1000.bag
"""

import argparse
import heapq
import json
import os

import cv2
import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import Quaternion, Vector3
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import Header
from tqdm import tqdm

DEFAULT_ROOT = '/home/hsiaochuan/Downloads/tartanground'
DEFAULT_CAMERAS = ['lcam_front']


def to_time(t):
    return rospy.Time.from_sec(float(t))


def camera_times(traj_dir, traj, t0):
    with open(os.path.join(traj_dir, f'{traj}_metadata.json')) as f:
        meta = json.load(f)
    return np.arange(meta['num_poses']) * meta.get('time_step', 0.1) + t0


def image_messages(traj_dir, camera, time, compressed):
    img_dir = os.path.join(traj_dir, f'image_{camera}')
    files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
    assert len(files) == len(time), f'{len(files)} images vs {len(time)} timestamps for {camera}'

    topic = f'/{camera}/image_raw' + ('/compressed' if compressed else '')
    for i, fname in enumerate(files):
        path = os.path.join(img_dir, fname)
        if compressed:
            msg = CompressedImage()
            msg.format = 'png'
            with open(path, 'rb') as f:
                msg.data = f.read()
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            msg = Image()
            msg.height, msg.width = img.shape[:2]
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = img.tobytes()
        msg.header.stamp = to_time(time[i])
        msg.header.frame_id = camera
        msg.header.seq = i
        yield time[i], topic, msg


def lidar_messages(traj_dir, time):
    lidar_dir = os.path.join(traj_dir, 'lidar')
    files = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.ply'))
    assert len(files) == len(time), f'{len(files)} lidar scans vs {len(time)} timestamps'

    for i, fname in enumerate(files):
        ply = PlyData.read(os.path.join(lidar_dir, fname))['vertex']
        points = np.stack([ply['x'], ply['y'], ply['z']], axis=-1).astype(np.float32)
        header = Header(stamp=to_time(time[i]), frame_id='lidar', seq=i)
        msg = point_cloud2.create_cloud_xyz32(header, points)
        yield time[i], '/lidar', msg


def imu_messages(traj_dir, t0):
    imu_dir = os.path.join(traj_dir, 'imu')
    time = np.load(os.path.join(imu_dir, 'imu_time.npy')) + t0
    acc = np.load(os.path.join(imu_dir, 'acc.npy'))
    gyro = np.load(os.path.join(imu_dir, 'gyro.npy'))
    # ori_global stores euler angles (roll pitch yaw) of the body in NED
    quat = Rotation.from_euler('xyz', np.load(os.path.join(imu_dir, 'ori_global.npy'))).as_quat()

    for i in range(len(time)):
        msg = Imu()
        msg.header.stamp = to_time(time[i])
        msg.header.frame_id = 'imu'
        msg.header.seq = i
        msg.orientation = Quaternion(*quat[i])
        msg.angular_velocity = Vector3(*gyro[i])
        msg.linear_acceleration = Vector3(*acc[i])
        yield time[i], '/imu', msg


def convert(traj_dir, traj, cameras, out_bag, t0, compressed):
    time = camera_times(traj_dir, traj, t0)
    streams = [image_messages(traj_dir, cam, time, compressed) for cam in cameras]
    total = len(cameras) * len(time)
    if os.path.isdir(os.path.join(traj_dir, 'lidar')):
        streams.append(lidar_messages(traj_dir, time))
        total += len(time)
    else:
        print('no lidar/ directory, skipping lidar')
    if os.path.isdir(os.path.join(traj_dir, 'imu')):
        streams.append(imu_messages(traj_dir, t0))
        total += len(np.load(os.path.join(traj_dir, 'imu', 'imu_time.npy')))
    else:
        print('no imu/ directory, skipping imu')

    print(f'Writing {out_bag}')
    with rosbag.Bag(out_bag, 'w') as bag:
        merged = heapq.merge(*streams, key=lambda x: x[0])
        for t, topic, msg in tqdm(merged, total=total, unit='msg'):
            bag.write(topic, msg, to_time(t))

    with rosbag.Bag(out_bag, 'r') as bag:
        info = bag.get_type_and_topic_info()[1]
        for topic, ti in info.items():
            print(f'  {topic}: {ti.message_count} msgs ({ti.msg_type})')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_ROOT, help='tartanground download root')
    parser.add_argument('--env', required=True, help='environment name, e.g. GothicIsland')
    parser.add_argument('--traj', help='trajectory name')
    parser.add_argument('--version', help='dataset version (Data_<version>)')
    parser.add_argument('--camera', nargs='+', default=DEFAULT_CAMERAS,
                        help='camera names (default: %(default)s)')
    parser.add_argument('--compressed', action='store_true',
                        help='write sensor_msgs/CompressedImage instead of raw Image')
    parser.add_argument('--t0', type=float, default=1000.0,
                        help='time offset added to all timestamps')
    parser.add_argument('-o', '--output', default=None,
                        help='output bag path (default: <env>_<traj>.bag next to this script)')
    args = parser.parse_args()

    traj_dir = os.path.join(args.root, args.env, f'Data_{args.version}', args.traj)
    if not os.path.isdir(traj_dir):
        raise SystemExit(f'trajectory dir not found: {traj_dir}')

    out_bag = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          f'{args.env}_{args.traj}.bag')
    convert(traj_dir, args.traj, args.camera, out_bag, args.t0, args.compressed)


if __name__ == '__main__':
    main()

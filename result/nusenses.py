"""
nuScenes Lidar Data Extractor
==============================
Extract all LiDAR frames from a specified scene, including non-keyframes.
Output:
  1. <output_dir>/lidar/       — Each point cloud is saved as <timestamp>.txt (x y z intensity ring)
  2. <output_dir>/poses.txt    — LiDAR poses in the global coordinate system for each frame
                                  Format: timestamp,px,py,pz,qx,qy,qz,qw

Pose computation:
    T_global_lidar = T_global_ego ∘ T_ego_lidar
    where T_ego_lidar comes from calibrated_sensor (the extrinsic from lidar to ego),
    and T_global_ego comes from ego_pose (ego to global coordinate system).

Usage:
    python extract_lidar.py \
        --dataroot /data/sets/nuscenes \
        --version  v1.0-mini \
        --scene    scene-0061 \
        --output   ./output_scene0061
"""

import argparse
import os

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def load_lidar_bin(filepath: str) -> np.ndarray:
    """
    Read a nuScenes LiDAR .pcd.bin file.
    File format: float32 × 5 columns, in the order x, y, z, intensity, ring_index.
    Returns a numpy array of shape (N, 5).
    """
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return points


def compose_pose(ego_translation, ego_rotation_q: Quaternion,
                 cs_translation, cs_rotation_q: Quaternion):
    """
    Compose the lidar→ego extrinsic and ego→global pose to get the LiDAR pose
    in the global coordinate system.

    T_global_lidar = T_global_ego  ∘  T_ego_lidar

    Position: p = R_ge * t_el + t_ge
    Rotation: q = q_ge * q_el     (pyquaternion multiplication composes rotations)

    Args:
        ego_translation  : list[3]       ego translation in global coordinates
        ego_rotation_q   : Quaternion    ego rotation in global coordinates (w,x,y,z)
        cs_translation   : list[3]       lidar translation in ego coordinates
        cs_rotation_q    : Quaternion    lidar rotation in ego coordinates (w,x,y,z)

    Returns:
        (px, py, pz, qx, qy, qz, qw)
    """
    t_ge = np.array(ego_translation)
    t_el = np.array(cs_translation)

    # LiDAR position in the global coordinate system
    p = ego_rotation_q.rotate(t_el) + t_ge   # R_ge * t_el + t_ge

    # LiDAR rotation in the global coordinate system
    q = ego_rotation_q * cs_rotation_q        # rotation composition

    # pyquaternion stores quaternions as (w, x, y, z); convert output to (x, y, z, w)
    return p[0], p[1], p[2], q.x, q.y, q.z, q.w


# ──────────────────────────────────────────────────────────────────────────────
# Core logic: collect all LiDAR sample_data in the scene
# ──────────────────────────────────────────────────────────────────────────────

def get_all_lidar_sample_data(nusc: NuScenes, scene_token: str):
    """
    Starting from scene_token, collect all sample_data entries belonging to the
    LIDAR_TOP channel in the scene (including keyframes and non-keyframes),
    sorted by timestamp.

    Strategy:
      1. Traverse all samples in the scene via the sample.next chain and collect their tokens.
      2. Start from the first sample's LIDAR_TOP keyframe and walk backward via prev to the earliest frame.
      3. Then traverse forward via next and stop when the associated sample is no longer in this scene.
    """
    scene_record = nusc.get('scene', scene_token)

    # ── Step 1: collect all sample tokens in this scene ───────────────────────
    scene_sample_tokens = set()
    cur_sample_token = scene_record['first_sample_token']
    while cur_sample_token:
        scene_sample_tokens.add(cur_sample_token)
        sample_rec = nusc.get('sample', cur_sample_token)
        cur_sample_token = sample_rec['next']  # '' indicates the end of the chain

    # ── Step 2: locate the first LIDAR_TOP keyframe ──────────────────────────
    first_sample = nusc.get('sample', scene_record['first_sample_token'])
    if 'LIDAR_TOP' not in first_sample['data']:
        raise RuntimeError("Scene {} 中未找到 LIDAR_TOP 数据。".format(scene_record['name']))

    first_lidar_token = first_sample['data']['LIDAR_TOP']

    # ── Step 3: walk backward via prev to the earliest LiDAR frame ───────────
    earliest_token = first_lidar_token
    while True:
        sd = nusc.get('sample_data', earliest_token)
        if not sd['prev']:
            break
        prev_sd = nusc.get('sample_data', sd['prev'])
        # A non-keyframe sample_token points to the sample of the next keyframe;
        # as long as that sample is still in this scene, the frame belongs here.
        if prev_sd['sample_token'] not in scene_sample_tokens:
            break
        earliest_token = sd['prev']

    # ── Step 4: collect all frames by traversing forward via next ─────────────
    all_sd_tokens = []
    cur_token = earliest_token
    while cur_token:
        sd = nusc.get('sample_data', cur_token)
        # Stop once we leave the scene boundary
        if sd['sample_token'] not in scene_sample_tokens:
            break
        all_sd_tokens.append(cur_token)
        cur_token = sd['next']

    return all_sd_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def extract(dataroot: str, version: str, scene_name: str, output_dir: str):
    print(f"[1/5] Loading nuScenes ({version}) ...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    # ── Search for scene ─────────────────────────────────────────────────────
    print(f"[2/5] Searching for scene '{scene_name}' ...")
    scene_token = None
    for s in nusc.scene:
        if s['name'] == scene_name:
            scene_token = s['token']
            break
    if scene_token is None:
        available = [s['name'] for s in nusc.scene]
        raise ValueError(
            f"Scene '{scene_name}' was not found.\nAvailable scenes: {available}"
        )
    scene_token = str(scene_token)

    # ── Collect all LiDAR sample_data ───────────────────────────────────────
    print("[3/5] Collecting all LiDAR frames (including non-keyframes) ...")
    all_lidar_tokens = get_all_lidar_sample_data(nusc, scene_token)
    print(f"      Found {len(all_lidar_tokens)} frames.")

    # ── Create output directory ──────────────────────────────────────────────
    lidar_out_dir = os.path.join(output_dir, 'lidar')
    os.makedirs(lidar_out_dir, exist_ok=True)

    poses_path = os.path.join(output_dir, 'poses.txt')

    # ── Extract and save ─────────────────────────────────────────────────────
    print("[4/5] Extracting point clouds and poses ...")
    pose_lines = ["#timestamp,px,py,pz,qx,qy,qz,qw"]

    for idx, sd_token in enumerate(all_lidar_tokens):
        sd_record  = nusc.get('sample_data', sd_token)
        cs_record  = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ego_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        timestamp = sd_record['timestamp']

        # ── Point cloud ─────────────────────────────────────────────────────
        lidar_filepath = os.path.join(dataroot, sd_record['filename'])
        if not os.path.exists(lidar_filepath):
            print(f"  [Warning] Point cloud file not found: {lidar_filepath}, skipping.")
            continue

        points = load_lidar_bin(lidar_filepath)   # (N, 5): x y z intensity ring

        out_txt = os.path.join(lidar_out_dir, f"{timestamp}.txt")
        # Save as space-separated text, column order: x y z intensity ring_index
        np.savetxt(out_txt, points, fmt='%.6f %.6f %.6f %.6f %d',
                   header='#x y z intensity ring_index', comments='')

        # ── Pose: LiDAR in the global coordinate system ─────────────────────
        ego_q = Quaternion(ego_record['rotation'])   # w,x,y,z
        cs_q  = Quaternion(cs_record['rotation'])    # w,x,y,z

        px, py, pz, qx, qy, qz, qw = compose_pose(
            ego_record['translation'], ego_q,
            cs_record['translation'],  cs_q
        )

        pose_lines.append(
            f"{timestamp} {px:.8f} {py:.8f} {pz:.8f} "
            f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}"
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(all_lidar_tokens):
            print(f"  Progress: {idx + 1}/{len(all_lidar_tokens)}")

    # ── Write pose file ──────────────────────────────────────────────────────
    print("[5/5] Saving pose file ...")
    with open(poses_path, 'w') as f:
        f.write('\n'.join(pose_lines) + '\n')

    n_lidar  = len(os.listdir(lidar_out_dir))
    n_poses  = len(pose_lines) - 1   # subtract header
    print(f"\nDone!")
    print(f"  Point cloud frames: {n_lidar}  →  {lidar_out_dir}")
    print(f"  Pose entries: {n_poses}  →  {poses_path}")
    if n_lidar != n_poses:
        print(f"  [Warning] The number of point cloud files ({n_lidar}) does not match the number of pose entries ({n_poses}); "
              "some point cloud files may be missing.")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nuScenes LiDAR data extraction tool')
    parser.add_argument('--dataroot', required=True,
                        help='Root directory of the nuScenes dataset, e.g. /data/sets/nuscenes')
    parser.add_argument('--version',  default='v1.0-mini',
                        help='Dataset version, e.g. v1.0-mini / v1.0-trainval (default: v1.0-mini)')
    parser.add_argument('--scene',    required=True,
                        help='Scene name, e.g. scene-0061')
    parser.add_argument('--output',   required=True,
                        help='Output root directory, e.g. ./output_scene0061')
    args = parser.parse_args()

    extract(
        dataroot=args.dataroot,
        version=args.version,
        scene_name=args.scene,
        output_dir=args.output,
    )
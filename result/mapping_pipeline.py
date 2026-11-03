"""Scans + odometry poses -> loop-closed poses (run_pgo) -> scans without dynamic points (run_clean)."""
import os.path
import subprocess

ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "build")
CONFIG = os.path.join(ROOT, "config", "pgo.yaml")

datas_dir = [
    "/home/hsiaochuan/slam/openMVG/data/building00_at128",
]
for data_dir in datas_dir:
    scans_dir = os.path.join(data_dir, "pcd")
    pose_fname = os.path.join(data_dir, "result", "result.txt")
    out = os.path.join(data_dir, "mapping")
    pgo_dir = os.path.join(out, "pgo")
    clean_dir = os.path.join(out, "clean")
    traj_pgo = os.path.join(pgo_dir, "traj_pgo.txt")

    # run PGO
    subprocess.run([
        os.path.join(BIN, "run_pgo"),
        "--scans_dir", scans_dir,
        "--pose_fname", pose_fname,
        "--config", CONFIG,
        "--out", pgo_dir,
    ], check=True)

    # run dynamic remove, on the corrected poses
    subprocess.run([
        os.path.join(BIN, "run_clean"),
        "--scans_dir", scans_dir,
        "--pose_fname", traj_pgo,
        "--out", clean_dir,
        "--voxel", "0.4",
        "--margin", "0.8",
        "--max_range", "80",
        "--min_free", "2",
        "--ratio", "3",
        "--min_hits", "6",
        "--near_range", "8",
        "--ground_dist", "0.1",
        "--up_z", "-1",            # building00: the world's z axis points down
        "--sensor_height", "0.95",
    ], check=True)

    # merge the cleaned scans into one map
    subprocess.run([
        os.path.join(BIN, "points_merge"),
        "--trajectory", traj_pgo,
        "--pcd_dir", clean_dir,
        "--output", os.path.join(out, "map.pcd"),
        "--voxel_size", "0.05",
    ], check=True)

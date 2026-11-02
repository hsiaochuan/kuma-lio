"""Scans + odometry poses -> loop-closed poses (run_pgo) -> scans without dynamic points (run_clean)."""
import subprocess
from pathlib import Path

BIN = Path(__file__).resolve().parent.parent / "build"
CONFIG = Path(__file__).resolve().parent.parent / "config" / "pgo.yaml"

scans_dir = "/home/hsiaochuan/slam/openMVG/data/building00/pcd"
pose_fname = "/home/hsiaochuan/slam/openMVG/data/building00/result/result.txt"
out = Path("/home/hsiaochuan/slam/openMVG/data/building00/mapping")

# run PGO
subprocess.run([
    str(BIN / "run_pgo"),
    "--scans_dir", scans_dir,
    "--pose_fname", pose_fname,
    "--config", str(CONFIG),
    "--out", str(out / "pgo"),
], check=True)

# run dynamic remove, on the corrected poses
subprocess.run([
    str(BIN / "run_clean"),
    "--scans_dir", scans_dir,
    "--pose_fname", str(out / "pgo" / "traj_pgo.txt"),
    "--out", str(out / "clean"),
    "--voxel", "0.2",
    "--margin", "0.4",
    "--max_range", "80",
    "--min_free", "3",
    "--ratio", "1.0",
    "--ground_dist", "0.3",
], check=True)

# merge the cleaned scans into one map
subprocess.run([
    str(BIN / "points_merge"),
    "--trajectory", str(out / "pgo" / "traj_pgo.txt"),
    "--pcd_dir", str(out / "clean"),
    "--output", str(out / "map.pcd"),
    "--voxel_size", "0.1",
], check=True)

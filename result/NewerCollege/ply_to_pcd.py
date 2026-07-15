"""
Convert a PLY point cloud to PCD, estimating point normals along the way.

Typical use: converting prior maps (e.g. Newer College's
maths-institute.ply / new-college-combined-5cm-v2.ply) into the .pcd
format expected as --prior_map_fname elsewhere in this repo.
"""

import argparse

import open3d as o3d


def ply_to_pcd(
    input_ply: str,
    output_pcd: str,
    normal_radius: float = 0.5,
    normal_max_nn: int = 30,
):
    pcd = o3d.io.read_point_cloud(input_ply)
    if not pcd.has_points():
        raise ValueError(f"No points read from {input_ply}")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    o3d.io.write_point_cloud(output_pcd, pcd, write_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Convert PLY to PCD with normal estimation")
    parser.add_argument("--input", help="Input .ply file", default="/mnt/data/home/hsiaochuan/data/newer_college2/prior/new-college-combined-5cm-v2.ply")
    parser.add_argument("--output", help="Output .pcd file", default="/mnt/data/home/hsiaochuan/data/newer_college2/prior/new-college-combined-5cm-v2.pcd")
    args = parser.parse_args()

    ply_to_pcd(
        args.input,
        args.output,
        0.1,
        12,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

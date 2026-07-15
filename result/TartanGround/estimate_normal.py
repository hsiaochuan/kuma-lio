#!/usr/bin/env python3
"""Read a pcd file, downsample, estimate normals, save pcd with normals.

Usage: python3 estimate_normal.py in.pcd [-o out.pcd] [--knn 30] [--voxel 0.1]
"""

import argparse

import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input .pcd', default="/home/hsiaochuan/Downloads/tartanground/AbandonedCable/AbandonedCable_rgb.pcd")
parser.add_argument('-o', '--output', default="/home/hsiaochuan/Downloads/tartanground/AbandonedCable/AbandonedCable.pcd",
                    help='output .pcd (default: <input>_with_normals.pcd)')
parser.add_argument('--knn', type=int, default=12, help='neighbors for normal estimation')
parser.add_argument('--voxel', type=float, default=0.05,
                    help='voxel size for downsampling, 0 to disable')
args = parser.parse_args()

out = args.output or args.input.replace('.pcd', '_with_normals.pcd')

pcd = o3d.io.read_point_cloud(args.input)
print(f'{args.input}: {len(pcd.points)} points')

if args.voxel > 0:
    pcd = pcd.voxel_down_sample(args.voxel)
    print(f'downsampled (voxel {args.voxel}): {len(pcd.points)} points')

pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(args.knn))
o3d.io.write_point_cloud(out, pcd)
print(f'saved {out}')

import os
import glob
import open3d as o3d
from tqdm import tqdm


def merge_pcds(input_dir, output_pcd, voxel_size=0.05):
    """
    Merge all PCD files in a folder, perform voxel downsampling,
    and save the merged point cloud.

    Parameters
    ----------
    input_dir : str
        Directory containing PCD files.
    output_pcd : str
        Output PCD file path.
    voxel_size : float
        Voxel size used for downsampling.
    """

    # Find all PCD files
    pcd_files = sorted(glob.glob(os.path.join(input_dir, "*.pcd")))

    if len(pcd_files) == 0:
        raise RuntimeError(f"No PCD files found in {input_dir}")

    merged_cloud = o3d.geometry.PointCloud()

    print(f"Found {len(pcd_files)} PCD files.")

    # Read and merge all point clouds
    for file in tqdm(pcd_files, desc="Reading PCD files"):
        pcd = o3d.io.read_point_cloud(file)
        merged_cloud += pcd

    print(f"Number of points before downsampling : {len(merged_cloud.points):,}")

    # Apply voxel downsampling
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size)

    print(f"Number of points after downsampling  : {len(merged_cloud.points):,}")

    # Save merged point cloud
    success = o3d.io.write_point_cloud(output_pcd, merged_cloud)

    if not success:
        raise RuntimeError(f"Failed to write {output_pcd}")

    print(f"Merged point cloud saved to:\n{output_pcd}")


if __name__ == "__main__":

    input_dir = "/home/hsiaochuan/Downloads/TUHH/pointclouds"
    output_pcd = "/home/hsiaochuan/Downloads/TUHH/map.pcd"

    merge_pcds(
        input_dir=input_dir,
        output_pcd=output_pcd,
        voxel_size=0.1
    )
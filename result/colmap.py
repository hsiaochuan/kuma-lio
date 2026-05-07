import subprocess
import os
from typing import List
import yaml

seq_dirs = ["/mnt/data/home/hsiaochuan/data/FAST-LIVO2/CBD_Building_01_faster_lio_result", ]
COLMAP_EXE = "colmap"


def ReadCameraModel(yaml_file: str) -> str:
    """Extract camera model from YAML config and map to COLMAP model name.

    Based on COLMAP models.h:
    - RADIAL: f, cx, cy, k1, k2
    - OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    cam_config = config.get("cam", {})
    model_name = cam_config.get("camera_model", "pinhole")

    # Map custom model names to COLMAP model names
    model_mapping = {
        "pinhole": "PINHOLE",
        "pinhole_radial": "OPENCV",
        "pinhole_fisheye": "OPENCV_FISHEYE",
    }

    return model_mapping.get(model_name, "OPENCV")


def ReadCameraParams(yaml_file: str) -> str:
    model_name = ReadCameraModel(yaml_file)
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    cam_config = config.get("cam", {})
    pinhole_param = cam_config.get("pinhole_param", [])
    distortion_param = cam_config.get("distortion_param", [])
    # normalize pinhole params to fx, fy, cx, cy
    fx = fy = cx = cy = 0.0
    if len(pinhole_param) == 4:
        fx, fy, cx, cy = [float(x) for x in pinhole_param]
    else:
        print("Warning: Invalid pinhole_param length: ", len(pinhole_param))

    # map COLMAP model names to parameter ordering
    if model_name == 'PINHOLE':
        params = [fx, fy, cx, cy]
    elif model_name == 'OPENCV':
        k1, k2, k3, p1, p2 = [float(x) for x in distortion_param]
        params = [fx, fy, cx, cy, k1, k2, p1, p2]
    elif model_name == 'OPENCV_FISHEYE':
        k1, k2, k3, k4 = [float(x) for x in distortion_param]
        params = [fx, fy, cx, cy, k1, k2, k3, k4]
    else:
        print("Warning: Unknown camera model: ", model_name)
        params = [fx, fy, cx, cy]

    # join as comma-separated string without spaces
    return ','.join([str(x) for x in params])


def tum_to_image_list_file(tum_file: str, images_dir: str, output_path: str) -> None:
    with open(tum_file, "r") as f:
        lines = f.readlines()
    image_paths = []
    for line in lines:
        if line.strip() == "" or line.startswith("#"):
            continue
        timestamp = line.split()[0]
        img_path = os.path.join(images_dir, timestamp + ".jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"Warning: Image file {img_path} does not exist.")

    write_image_list(image_paths, output_path)

def write_image_list(image_paths: List[str], output_path: str) -> None:
    with open(output_path, "w") as f:
        for img in image_paths:
            f.write(img + "\n")


def run_colmap(data_dir: str,
               extract_match: bool = True,
               mapping: bool = True,
               triangulate: bool = True) -> None:
    img_dir = os.path.join(data_dir, "images")
    database_fname = os.path.join(data_dir, "database.db")
    config_fname = os.path.join(data_dir, "config.yaml")
    image_list_path = os.path.join(data_dir, "images.txt")

    if extract_match:
        subprocess.run([
            COLMAP_EXE, "feature_extractor",
            "--database_path", database_fname,
            "--image_path", img_dir,
            "--image_list_path", image_list_path,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", ReadCameraModel(config_fname),
            "--ImageReader.camera_params", ReadCameraParams(config_fname),
        ], check=True)

        subprocess.run([
            COLMAP_EXE, "exhaustive_matcher",
            "--database_path", database_fname,
            "--SiftMatching.guided_matching", "1",
        ], check=True)
    if triangulate:
        os.makedirs(os.path.join(data_dir, "colmap_final"), exist_ok=True)
        subprocess.run([
            COLMAP_EXE, "point_triangulator",
            '--database_path', database_fname,
            '--image_path', img_dir,
            '--input_path', os.path.join(data_dir, "colmap_result"),
            '--output_path', os.path.join(data_dir, "colmap_final"),
            '--Mapper.tri_ignore_two_view_tracks', '0',
        ],check=True)
        subprocess.run([
            COLMAP_EXE, "model_analyzer",
            "--path", os.path.join(data_dir, "colmap_final"),
        ], check=True)
        subprocess.run([
            COLMAP_EXE, "model_converter",
            "--output_type", "TXT",
            "--input_path", os.path.join(data_dir, "colmap_final"),
            "--output_path", os.path.join(data_dir, "colmap_final"),
        ], check=True)
    if mapping:
        sparse_dir = os.path.join(data_dir, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)
        subprocess.run([
            COLMAP_EXE, "mapper",
            "--database_path", database_fname,
            "--image_path", img_dir,
            "--output_path", sparse_dir,
        ], check=True)

        subprocess.run([
            COLMAP_EXE, "model_converter",
            "--output_type", "TXT",
            "--input_path", sparse_dir,
            "--output_path", sparse_dir,
        ], check=True)

        subprocess.run([
            COLMAP_EXE, "model_analyzer",
            "--path", sparse_dir,
        ], check=True)


def main() -> None:
    for seq in seq_dirs:
        run_colmap(seq)


if __name__ == "__main__":
    main()

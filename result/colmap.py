import subprocess
import os
from typing import List
import yaml

seq_dirs = ["/mnt/data/home/hsiaochuan/data/FAST-LIVO2/CBD_Building_01_faster_lio_result",]
COLMAP_EXE = "colmap"

def ReadCameraModel(yaml_file: str)->str:
    """Extract camera model from YAML config and map to COLMAP model name.

    Based on COLMAP models.h:
    - RADIAL: f, cx, cy, k1, k2
    - OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    - FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
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
        "spherical": "OPENCV"
    }

    return model_mapping.get(model_name, "OPENCV")

def ReadCameraParams(yaml_file: str)->str:
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
def sample_images(img_dir: str, sample_interval: int) -> List[str]:
    sample_result = []
    img_files = os.listdir(img_dir)
    img_files.sort()
    for i in range(0, len(img_files), sample_interval):
        sample_result.append(os.path.join(img_dir, img_files[i]))
    return sample_result
for seq in seq_dirs:
    img_dir = os.path.join(seq, "images")
    database_fname = os.path.join(seq, "database.db")
    config_fname = os.path.join(seq, "config.yaml")
    # sample images
    sub_images = sample_images(img_dir, 5)
    with open(os.path.join(seq, "images.txt"), "w") as f:
        for img in sub_images:
            f.write(img + "\n")
    subprocess.run([
        COLMAP_EXE, "feature_extractor",
        '--database_path', database_fname,
        '--image_path', img_dir,
        '--image_list_path', os.path.join(seq, "images.txt"),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', ReadCameraModel(config_fname),
        '--ImageReader.camera_params', ReadCameraParams(config_fname),
    ])

    subprocess.run([
        COLMAP_EXE, 'exhaustive_matcher',
        '--database_path', database_fname,
        '--SiftMatching.guided_matching', '1',
    ])

    os.makedirs(os.path.join(seq, "sparse"), exist_ok=True)
    subprocess.run([
        COLMAP_EXE, 'mapper',
        '--database_path', database_fname,
        '--image_path', img_dir,
        '--output_path', os.path.join(seq, "sparse"),
    ])

    subprocess.run([
        COLMAP_EXE, 'model_converter',
        '--output_type', 'TXT',
        '--input_path', os.path.join(seq, "sparse/0"),
        '--output_path', os.path.join(seq, "sparse/0"),
    ])

    subprocess.run([
        COLMAP_EXE, "model_analyzer",
        "--path", os.path.join(seq, "sparse/0"),
    ])

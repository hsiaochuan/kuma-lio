import subprocess
import os
pixel_lc_calibra_dir = "/mnt/data/home/hsiaochuan/data/pixel_lidar_camera_calibration"
pixel_lc_calibra_datas = [
    "0",
    # "1",
    # "2",
]

subprocess.run([
    'make',
    '-C', '../build',
    'lidar_camera_calibration',
    '-j', '4'
])
LIDAR_CAMERA_CALIBRATION_EXE = "../build/lidar_camera_calibration"
for data in pixel_lc_calibra_datas:
    os.makedirs(os.path.join(pixel_lc_calibra_dir, data + "_result"), exist_ok=True)
    subprocess.run([
        LIDAR_CAMERA_CALIBRATION_EXE,
        "--image_file", os.path.join(pixel_lc_calibra_dir, data + ".png"),
        "--pcd_file", os.path.join(pixel_lc_calibra_dir, data + ".pcd"),
        "--output_dir", os.path.join(pixel_lc_calibra_dir, data + "_result"),
        "--calib_config_file", "../config/calibration/config_outdoor.yaml",
        "--use_rough_calib", "1",
    ])
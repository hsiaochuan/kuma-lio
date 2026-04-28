import subprocess
import os

seq_dirs = ["/mnt/data/home/hsiaochuan/data/FAST-LIVO2/CBD_Building_01_faster_lio_result",]
COLMAP_EXE = "colmap"
for seq in seq_dirs:
    img_dir = os.path.join(seq, "images")
    dataset_fname = os.path.join(seq, "dataset.db")
    subprocess.run([
        COLMAP_EXE, "feature_extractor",
        '--database_path', dataset_fname,
        '--image_path', img_dir,
    ])

    subprocess.run([
        COLMAP_EXE, 'exhaustive_matcher',
        '--database_path', dataset_fname,
    ])
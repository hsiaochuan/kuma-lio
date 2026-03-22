import subprocess
import os
import datetime


def pcd_points_count(pcd_fname: str) -> int:
    try:
        with open(pcd_fname, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('POINTS'):
                    points_count = int(line.split()[1])
                    return points_count
                if line.startswith('DATA'):
                    break
        return 0
    except Exception as e:
        print(f"error in open: {e}")
        return -1


OFFLINE_APP = "../build/run_mapping_offline"
bag_files = [
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ntu_day_01.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ntu_day_02.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ntu_day_10.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ntu_night_04.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ntu_night_08.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/ntu_night_13.bag",

    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/kth_day_06.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/kth_day_09.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/kth_day_10.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/kth_night_01.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/kth_night_04.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/kth_night_05.bag",

    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/tuhh_day_02.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/tuhh_day_03.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/tuhh_day_04.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/tuhh_night_07.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/tuhh_night_08.bag",
    "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/bag/tuhh_night_09.bag",
]
atv_config = '../config/MCD_VIRAL_ATV.yaml'
handheld_config = '../config/MCD_VIRAL_HandHeld.yaml'
time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
f = open(os.path.join("/tmp/result_" + time_str + ".txt"), 'w')
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
f.write(f'GIT_BRANCH = "{branch}"\n')
f.write(f'GIT_COMMIT = "{commit_id}"\n')
f.flush()

subprocess.run([
    'make',
    '-C', '../build',
    '-j', '4',
], check=True)
for bag_fname in bag_files:
    dir = os.path.dirname(bag_fname)
    base_name = os.path.basename(bag_fname)
    name, ext = os.path.splitext(base_name)
    output_dir = os.path.join(dir, name + "_faster_lio_result")
    os.makedirs(output_dir, exist_ok=True)
    config = ''
    if 'ntu' in name:
        config = atv_config
    else:
        config = handheld_config
    subprocess.run([OFFLINE_APP,
                    '--config_file', config,
                    '--bag_file', bag_fname,
                    '--output_dir', output_dir,
                    ])
    points_count = 0
    for fname in os.listdir(os.path.join(output_dir, "maps")):
        points_count += pcd_points_count(os.path.join(output_dir, "maps", fname))
    f.write("{},{}\n".format(name, points_count))

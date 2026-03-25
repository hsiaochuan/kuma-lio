import subprocess
import os
import datetime
import time


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


if __name__ == "__main__":
    OFFLINE_APP = "../build/run_mapping_offline"
    ONLINE_APP = "../build/run_mapping_online"
    bag_files = [
        "/mnt/data/home/hsiaochuan/data/Botanic/1005_00_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1005_01_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1005_05_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1005_07_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1006_01_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1006_03_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1008_01_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1008_03_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1018_00_LIO.bag",
        "/mnt/data/home/hsiaochuan/data/Botanic/1018_13_LIO.bag",

    ]
    config = '../config/BotanicGarden.yaml'

    # write the header
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    f = open(os.path.join("./Botanic_Garden_" + time_str + ".txt"), 'w')
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    f.write(f'DATA_SET = Botanic Garden\n')
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
        offline = True
        if offline:
            subprocess.run([OFFLINE_APP,
                            '--config_file', config,
                            '--bag_file', bag_fname,
                            '--output_dir', output_dir,
                            ])
        else:
            roscore = subprocess.Popen([
                'roscore',
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rviz = subprocess.Popen(['rviz',
                                     '-d',
                                     '../rviz_cfg/loam_livox.rviz'
                                     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
            subprocess.run(['rosparam', 'load',config])
            online_app = subprocess.Popen([ONLINE_APP, '--output_dir', output_dir])
            subprocess.run(['rosbag', 'play', bag_fname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            online_app.terminate()
            rviz.terminate()
            roscore.terminate()
        points_count = 0
        for fname in os.listdir(os.path.join(output_dir, "maps")):
            points_count += pcd_points_count(os.path.join(output_dir, "maps", fname))
        f.write("{},{}\n".format(name, points_count))

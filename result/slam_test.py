"""
SLAM Test Framework
====================
General SLAM algorithm test framework, supporting multiple datasets, offline/online modes, automatic compilation and result aggregation.
"""

import subprocess
import os
import sys
import datetime
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from enum import Enum
import shutil
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evo.tools import plot
from evo.tools.settings import SETTINGS

# Use the evo checked out in third_party/evo (not any pip-installed evo) for ATE evaluation
_EVO_DIR = str(Path(__file__).resolve().parent.parent / "third_party" / "evo")
if _EVO_DIR not in sys.path:
    sys.path.insert(0, _EVO_DIR)
from evo.core import metrics, sync
from evo.tools import file_interface
# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

class RunMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass
class RunTask:
    """All parameters for a single run of one bag"""
    bag_file: str
    config: str
    output_dir: str
    run_mode: RunMode = RunMode.OFFLINE
    start: float = 0.0
    duration: float = -1.0
    prior_map_fname: str = ""  # prior map for localization mode
    prior_init_pose: str = ""  # init pose for localization mode
    name: str = ""  # automatically extracted from bag_file
    ground_truth_fname: str = ""
    is_localization: bool = False

    @property
    def localization_args(self) -> List[str]:
        if not self.is_localization:
            return []
        args = ["--prior_map_fname", self.prior_map_fname]
        args += ["--prior_init_pose", self.prior_init_pose]
        return args


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    bag_files: List[str]
    config: Dict[str,str] = field(default_factory=dict)
    start: float = 0.0
    duration: float = -1.0
    run_mode: RunMode = RunMode.OFFLINE
    prior_init_pose: Dict[str, str] = field(default_factory=dict)
    prior_map: Dict[str, str] = field(default_factory=dict)
    ground_truth_dir: str = ""
    is_localization: bool = False

@dataclass
class TestResult:
    bag_name: str
    bag_file: str
    points_count: int = 0
    duration_sec: float = 0.0
    ate: float = 0.0

@dataclass
class SuiteResult:
    """Overall test suite result"""
    dataset_name: str
    start_time: str = ""
    end_time: str = ""
    test_results: List[TestResult] = field(default_factory=list)


# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────


def pcd_points_count(pcd_fname: str) -> int:
    """Read PCD file header, return point count; return -1 on failure"""
    try:
        with open(pcd_fname, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('POINTS'):
                    return int(line.split()[1])
                if line.startswith('DATA'):
                    break
        return 0
    except Exception as e:
        return -1


def count_points_in_dir(maps_dir: str) -> int:
    """Count total points in all PCD files in the directory"""
    total = 0
    if not os.path.isdir(maps_dir):
        return 0
    for fname in os.listdir(maps_dir):
        if fname.endswith('.pcd'):
            total += max(0, pcd_points_count(os.path.join(maps_dir, fname)))
    return total


def get_git_info() -> Tuple[str, str]:
    """Return (branch, commit_id); return empty strings on failure"""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"


# ──────────────────────────────────────────────
# Core Class
# ──────────────────────────────────────────────

class SLAMTestRunner:
    """
    SLAM test executor
    """

    def __init__(
            self,
            offline_app: str = "../build/run_mapping_offline",
            online_app: str = "../build/run_mapping_online",
            points_post_app: str = "../build/points_jet",

            build_dir: str = "../build",
            output_root: str = "./test_results",

            if_delete_result_dir: bool = False,
            if_slam: bool = True,
            if_postprocess: bool = False,
    ):
        self.git_branch = get_git_info()[0]
        self.git_commit = get_git_info()[1]
        self.offline_app = offline_app
        self.online_app = online_app
        self.points_post_app = points_post_app

        self.build_dir = build_dir
        self.output_root = output_root

        self.run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.if_delete_result_dir = if_delete_result_dir
        self.if_slam = if_slam
        self.if_postprocess = if_postprocess

    # ── Build ──────────────────────────────────

    def build(self) -> bool:
        jobs = 4
        print(f"Building project (jobs={jobs}) ...")
        try:
            subprocess.run(
                ["make",
                 "-C", self.build_dir,
                 "run_mapping_offline",
                 # "run_mapping_online",
                 "-j", str(jobs)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Build succeeded")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build failed:\n{e.stderr.decode()}")
            return False

    # ── Single bag run ───────────────────────────

    def run_offline(self, task: RunTask) -> bool:
        rviz = subprocess.Popen(
            ["rviz", "-d", "../rviz_cfg/loam_livox.rviz"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            subprocess.run(
                [self.offline_app,
                 "--config_file", task.config,
                 "--bag_file", task.bag_file,
                 "--output_dir", task.output_dir,
                 "--start", str(task.start),
                 '--duration', str(task.duration),
                 ],
                check=True,
            )
            rviz.terminate()
            return True
        except subprocess.CalledProcessError as e:
            print(f"  run_mapping_offline exited with code {e.returncode}")
            rviz.terminate()
            return False

    def run_post_process(self, output_dir: str):
        subprocess.run([
            self.points_post_app,
            '--input', os.path.join(output_dir, "final.pcd"),
            '--output', os.path.join(output_dir, "final_post.pcd"),
        ], check=True)

    def run_online(self, task: RunTask) -> bool:
        roscore = rviz = online_proc = None
        try:
            roscore = subprocess.Popen(
                ["roscore"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            rviz = subprocess.Popen(
                ["rviz", "-d", "../rviz_cfg/loam_livox.rviz"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            time.sleep(5)

            online_proc = subprocess.Popen(
                [self.online_app,
                 "--output_dir", task.output_dir,
                 "--config_fname", task.config] + task.localization_args,
            )

            if task.is_localization:
                time.sleep(30)
            bag_play_command = [
                "rosbag", "play",
                "--start", str(task.start),
                task.bag_file,
            ]
            if task.duration > 0:
                bag_play_command += ["--duration", str(task.duration)]
            subprocess.run(
                bag_play_command,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception as e:
            print(f"  Online mode exception: {e}")
            return False
        finally:
            for proc in [online_proc, rviz, roscore]:
                if proc:
                    proc.terminate()

    def run_single(self, task: RunTask) -> TestResult:
        output_dir = task.output_dir
        result = TestResult(
            bag_name=task.name,
            bag_file=task.bag_file,
        )
        print(f"{task.name}  [{task.run_mode}]")

        # copy config
        shutil.copy(task.config, os.path.join(output_dir, "config.yaml"))

        # pipeline
        start_time = time.time()
        if self.if_slam:
            if task.run_mode == RunMode.OFFLINE:
                self.run_offline(task)
            else:
                self.run_online(task)

        if self.if_postprocess:
            self.run_post_process(output_dir)

        end_time = time.time()

        result.duration_sec = round(end_time - start_time, 1)
        # result.points_count = count_points_in_dir(os.path.join(output_dir, "maps"))
        if os.path.exists(task.ground_truth_fname):
            result.ate = self.evaluate_ate(task.ground_truth_fname, output_dir)
            print("ATE RMSE: {}".format(result.ate))
        else:
            print("Fail to evaluate the ATE RMSE")
        return result

    def evaluate_ate(self, ground_truth_fname: str, output_dir: str) -> float:
        est_fname = os.path.join(output_dir, "traj_log.txt")
        traj_ref = file_interface.read_tum_trajectory_file(ground_truth_fname)
        traj_est = file_interface.read_tum_trajectory_file(est_fname)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)
        traj_est.align(traj_ref, correct_scale=False)

        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_ref, traj_est))

        self.save_ate_plots(traj_ref, traj_est, ape_metric, output_dir)

        return ape_metric.get_statistic(metrics.StatisticsType.rmse)

    def save_ate_plots(self, traj_ref, traj_est, ape_metric: "metrics.APE", output_dir: str):
        """Save the same two plots `evo_ape --save_plot` produces: the raw APE error
        over the trajectory index, and the trajectory colored by APE error."""
        error_array = ape_metric.error
        stats = ape_metric.get_all_statistics()

        fig_error = plt.figure(figsize=SETTINGS.plot_figsize)
        plot.error_array(
            fig_error.gca(), error_array,
            statistics={s: stats[s] for s in SETTINGS.plot_statistics if s not in ("min", "max")},
            name="APE", title="APE w.r.t. translation part (m)", xlabel="index",
        )
        fig_error.savefig(os.path.join(output_dir, "ate_error.png"))
        plt.close(fig_error)

        plot_mode = plot.PlotMode.xy
        fig_traj = plt.figure(figsize=SETTINGS.plot_figsize)
        ax = plot.prepare_axis(fig_traj, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, style=SETTINGS.plot_reference_linestyle,
                  color=SETTINGS.plot_reference_color, label="reference",
                  alpha=SETTINGS.plot_reference_alpha)
        plot.traj_colormap(
            ax, traj_est, error_array, plot_mode,
            min_map=stats["min"], max_map=stats["max"],
            title="ATE mapped onto trajectory (aligned)",
        )
        fig_traj.savefig(os.path.join(output_dir, "ate_trajectory.png"))
        plt.close(fig_traj)


    # ── Dataset run ────────────────────────────

    def run_dataset(self, dataset: DatasetConfig) -> SuiteResult:
        branch, commit = get_git_info()
        suite = SuiteResult(
            dataset_name=dataset.name,
        )

        print(f"\n{'=' * 50}")
        print(f"   Dataset: {dataset.name}  ({len(dataset.bag_files)} bags)")
        print(f"   branch={branch}  commit={commit[:8]}")

        suite.start_time = datetime.datetime.now().isoformat()
        for bag_file in dataset.bag_files:
            name = Path(bag_file).stem
            output_dir = os.path.join(
                Path(bag_file).parent,
                name + "_faster_lio_result"
            )
            if os.path.exists(output_dir) and self.if_delete_result_dir:
                print(f"Result dir {output_dir} already exists, remove it")
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            config_fname = ""
            for pattern, config_fname in dataset.config.items():
                if re.search(pattern, name):
                    break
            prior_map_fname = ""
            for pattern, prior_map_fname in dataset.prior_map.items():
                if re.search(pattern, name):
                    break
            prior_init_pose = ""
            for pattern, prior_init_pose in dataset.prior_init_pose.items():
                if re.search(pattern, name):
                    break
            task = RunTask(
                bag_file=bag_file,
                config=config_fname,
                name=name,
                output_dir=output_dir,
                run_mode=dataset.run_mode,
                start=dataset.start,
                duration=dataset.duration,
                prior_map_fname=prior_map_fname,
                prior_init_pose=prior_init_pose,
                ground_truth_fname= os.path.join(dataset.ground_truth_dir, name + ".txt"),
                is_localization=dataset.is_localization,
            )
            test_result = self.run_single(task)
            suite.test_results.append(test_result)

        suite.end_time = datetime.datetime.now().isoformat()
        return suite

    # ── Write result files ────────────────────────────

    def write_txt_report(self, suite_results: List[SuiteResult], fname: str):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(f"GIT_BRANCH  = \"{self.git_branch}\"\n")
            f.write(f"GIT_COMMIT  = \"{self.git_commit}\"\n")
            f.write("-" * 65 + "\n")
            for suite_result in suite_results:
                for test_result in suite_result.test_results:
                    f.write("{},\t{},\t{}、n\n".format(test_result.bag_name, test_result.duration_sec, test_result.ate))
            f.write("-" * 65 + "\n")


    # ── Main entry ────────────────────────────────

    def run_all(self, datasets: List[DatasetConfig]) -> List[SuiteResult]:
        # Build
        build_ok = self.build()
        if not build_ok:
            print("Build failed, aborting tests")
            return []

        suite_results = []
        for dataset in datasets:
            suite_result = self.run_dataset(dataset)
            suite_results.append(suite_result)
        self.write_txt_report(suite_results, os.path.join(self.output_root, self.run_time + ".txt"))

        return suite_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def DatasetsList(name_list: List[str]) -> List[DatasetConfig]:

    mcd_viral = DatasetConfig(
        name="mcd_viral",
        config={
            "ntu": "../config/mcd_viral_atv.yaml",
            "kth|tuhh": "../config/mcd_viral_handheld.yaml",
        },
        is_localization=False,
        prior_map={
            "ntu": "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/map/NTU/map.pcd",
            "tuhh": "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/map/TUHH/map.pcd",
            "kth": "/mnt/data/home/hsiaochuan/data/MCD_VIRAL/map/KTH/map.pcd",
        },
        prior_init_pose={
            "ntu_day_01": "49.260631711144441, 107.371797989246588, 7.635809572392588, 0.936118452267473, -0.351663294301812, 0.003894594980225, -0.000053806028052",
            "ntu_day_02": "61.961311814185514, 119.593225444885661, 7.696161795293277, 0.393007783516859, -0.919096202232126, -0.005400025874432, -0.027890730686113",
            "ntu_day_10": "39.916532928019564, 23.052257138316396, 7.299254388426964, 0.886432211042536, 0.292027951287205, 0.156779604991748, 0.323075480889327",
            "tuhh_day_02": "45.690090862136245, 447.141417866168638, 14.609562334189976, -0.197423661689090, 0.979502111007773, 0.001873158034956, 0.039950013962013",
            "tuhh_day_03": "43.608801682558919, 446.534935432984810, 14.597698878700372, -0.154246716717734, 0.987560512077699, -0.009204072595886, 0.029111345112889",
            "tuhh_day_04": "35.139850852161452, 114.906595889361554, -1.274419348455590, -0.413172157260666, 0.910454794447510, -0.007934554354419, 0.017259159287378",
            "kth_day_06": "64.393253256515791, 66.483233094665749, 38.514334105006881, 0.289019760011319, 0.956543598228710, -0.018788172343168, -0.033748001284148",
            "kth_day_09": "70.392009196482377, 63.111919578407303, 38.304007735781937, 0.974443073722663, -0.223447643378903, -0.022907254727646, -0.002665412385693",
            "kth_day_10": "69.125536959169182, 63.567368550279582, 38.383523176445017, 0.305395593145653, 0.952159100742401, 0.002354607087536, -0.011001562893546",
        },
        bag_files=[
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
        ],
        run_mode=RunMode.OFFLINE,
        ground_truth_dir="/mnt/data/home/hsiaochuan/data/MCD_VIRAL/ground_truth",
    )

    mcd_viral_ouster = DatasetConfig(
        name="mcd_viral_ouster",
        config={".":"../config/mcd_viral_ouster.yaml"},
        bag_files=[
            "/media/hsiaochuan/hsiaochuan-data/MCD_VIRAL/raw/ouster/ntu_day_02_os1_128.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )


    hilti_2022 = DatasetConfig(
        name="hilti_2022",
        config={".":"../config/hilti_2022.yaml"},
        bag_files=[
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp01_construction_ground_level.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp03_construction_stairs.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp04_construction_upper_level.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp06_construction_upper_level_3.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp07_long_corridor.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp09_cupola.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp11_lower_gallery.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp14_basement_2.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp15_attic_to_upper_gallery.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp16_attic_to_upper_gallery_2.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp18_corridor_lower_gallery_2.bag",
            "/media/hsiaochuan/hsiaochuan-data/hilti2022/exp21_outside_building.bag",
        ],
        run_mode=RunMode.OFFLINE,
        ground_truth_dir="/media/hsiaochuan/hsiaochuan-data/hilti2022/groundtruth",
    )

    fast_livo2 = DatasetConfig(
        name="fast_livo2",
        config={
            "Retail_Street": "../config/fast_livo2.yaml",
            "CBD_Building_01": "../config/fast_livo2.yaml",
            "Bright_Screen_Wall": "../config/fast_livo2.yaml",
            "HKU_Landmark": "../config/fast_livo2.yaml",
            "HKU_Centennial_Garden": "../config/fast_livo2.yaml",
            "HKU_Main_Building": "../config/fast_livo2.yaml",
            "HKU_Lecture_Center_01": "../config/fast_livo2.yaml",
            "SYSU_01": "../config/fast_livo2.yaml",
        },
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/Retail_Street.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/CBD_Building_01.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/Bright_Screen_Wall.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/HKU_Landmark.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/HKU_Centennial_Garden_01.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/HKU_Main_Building.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/HKU_Lecture_Center_01.bag",
            "/mnt/data/home/hsiaochuan/data/FAST-LIVO2/SYSU_01.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )

    ntu_viral = DatasetConfig(
        name="ntu_viral",
        config={".":"../config/ntu_viral.yaml"},
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/ntu_viral/eee_03/eee_03.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )
    mars_lvig = DatasetConfig(
        name="mars_lvig",
        config={".":"../config/mars_lvig.yaml"},
        bag_files=[
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKairport_GNSS03.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKisland_GNSS03.bag",
        ],
        run_mode=RunMode.OFFLINE,
        ground_truth_dir="/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/ground-truth",
    )
    all_datasets = [mcd_viral, mcd_viral_ouster, hilti_2022, fast_livo2, ntu_viral, mars_lvig]
    run_datasets = []
    for dataset in all_datasets:
        if dataset.name in name_list:
            run_datasets.append(dataset)
    return run_datasets


def main():
    parser = argparse.ArgumentParser(description="SLAM Test Framework")
    parser.add_argument("--datasets", nargs="+",
                        default=["mcd_viral"],
                        help="Run only specified datasets (by name)")
    parser.add_argument("--if_delete_result_dir", action="store_true", default=True, help="Delete result dir if exists")
    parser.add_argument("--if_slam", action="store_true", default=True, help="Run SLAM")
    parser.add_argument("--if_lvba", action="store_true", default=True, help="Run LVBA")
    parser.add_argument("--if_postprocess", action="store_true", default=False, help="Run points post-processing")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (sec) for offline mode")
    parser.add_argument("--duration", type=float, default=-1.0,
                        help="Duration (sec) for offline mode, -1 for full length")
    args = parser.parse_args()

    # decide the datasets to run
    data_name_list = args.datasets
    datasets = DatasetsList(data_name_list)
    print(f"Selected datasets: {[d.name for d in datasets]}")

    # decide the start and duration
    for dataset in datasets:
        dataset.start = args.start
        dataset.duration = args.duration
    print(f"start from {args.start} and duration is {args.duration}")
    # run
    runner = SLAMTestRunner()
    runner.if_delete_result_dir = args.if_delete_result_dir
    runner.if_slam = args.if_slam
    runner.if_lvba = args.if_lvba
    runner.if_postprocess = args.if_postprocess
    runner.run_all(datasets)


if __name__ == "__main__":
    main()

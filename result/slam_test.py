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

# Use the evo checked out in third_party/evo (not any pip-installed evo) for ATE evaluation
_EVO_DIR = str(Path(__file__).resolve().parent.parent / "third_party" / "evo")
if _EVO_DIR not in sys.path:
    sys.path.insert(0, _EVO_DIR)
from evo.core import metrics, sync
from evo import main_ape
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
    name: str = ""  # automatically extracted from bag_file
    ground_truth_fname: str = ""
    gnss_obs_fname: str = ""
    gnss_nav_fname: str = ""
    clean_config: str = "../config/clean.yaml"
    pgo_config: str = "../config/pgo.yaml"
    ba_config: str = "../config/ba.yaml"


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    bag_files: List[str]
    config: Dict[str, str] = field(default_factory=dict)
    start: float = 0.0
    duration: float = -1.0
    run_mode: RunMode = RunMode.OFFLINE
    ground_truth_dir: str = ""
    gnss_dir: str = ""
    clean_config: str = "../config/clean.yaml"  # run_clean parameters
    pgo_config: str = "../config/pgo.yaml"  # run_pgo parameters, they depend on the sensor's field of view
    ba_config: str = "../config/ba.yaml"  # run_ba parameters


@dataclass
class TestResult:
    bag_name: str
    bag_file: str
    points_count: int = 0
    duration_sec: float = 0.0  # SLAM only
    ate: float = 0.0
    ate_pgo: float = 0.0
    ate_ba: float = 0.0


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
            mls_app: str = "../build/run_mls",
            pgo_app: str = "../build/run_pgo",
            clean_app: str = "../build/run_clean",
            ba_app: str = "../build/run_ba",
            merge_app: str = "../build/run_merge",

            build_dir: str = "../build",
            output_root: str = "./test_results",

            if_delete_result_dir: bool = False,
            if_slam: bool = True,
            if_pgo: bool = False,
            if_clean: bool = False,
            if_ba: bool = False,
            if_merge: bool = False,
            if_mls: bool = False,
    ):
        self.git_branch = get_git_info()[0]
        self.git_commit = get_git_info()[1]
        self.offline_app = offline_app
        self.mls_app = mls_app
        self.pgo_app = pgo_app
        self.clean_app = clean_app
        self.ba_app = ba_app
        self.merge_app = merge_app

        self.build_dir = build_dir
        self.output_root = output_root

        self.run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.if_delete_result_dir = if_delete_result_dir
        self.if_slam = if_slam
        self.if_pgo = if_pgo
        self.if_clean = if_clean
        self.if_ba = if_ba
        self.if_merge = if_merge
        self.if_mls = if_mls

    # ── Build ──────────────────────────────────

    def build(self) -> bool:
        jobs = 4
        print(f"Building project (jobs={jobs}) ...")
        try:
            subprocess.run(
                ["make",
                 "-C", self.build_dir,
                 "run_mapping_offline",
                 "run_pgo",
                 "run_clean",
                 "run_ba",
                 "run_merge",
                 "run_mls",
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
        try:
            subprocess.run(
                [self.offline_app,
                 "--config_file", task.config,
                 "--bag_file", task.bag_file,
                 "--output_dir", task.output_dir,
                 "--start", str(task.start),
                 '--duration', str(task.duration),
                 "--gnss_obs_fname", task.gnss_obs_fname,
                 "--nav_fname", task.gnss_nav_fname,
                 ],
                check=True,
            )

            return True
        except subprocess.CalledProcessError as e:
            print(f"  run_mapping_offline exited with code {e.returncode}")
            return False

    def run_step(self, name: str, command: List[str]) -> bool:
        print(f"  {name}")
        try:
            subprocess.run(command, check=True)
            return True
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"  {name} failed: {e}")
            return False

    def run_mapping(self, task: RunTask):
        """SLAM scans -> loop closure -> dynamic point removal -> bundle adjustment -> merged map -> smoothed map.

        Every step takes what the steps before it produced: without --if_pgo the scans are placed by
        traj_log.txt, without --if_clean the merged map has the dynamic points and run_ba adjusts with them,
        without --if_ba the merged map is placed by the poses of run_pgo.  Poses are picked by which files exist,
        even from an earlier run: run_ba starts from pgo/traj_pgo.txt, else traj_log.txt; run_merge takes
        ba/traj_ba.txt, else pgo/traj_pgo.txt, else traj_log.txt.
        """
        output_dir = task.output_dir
        scans_dir = os.path.join(output_dir, "scans")
        poses = os.path.join(output_dir, "traj_log.txt")  # its first column is the file stem of each scan
        map_fname = os.path.join(output_dir, "map.ply")
        if not (self.if_pgo or self.if_clean or self.if_ba or self.if_merge or self.if_mls):
            return
        if not os.path.isdir(scans_dir):
            print(f"  no {scans_dir}, mapping steps skipped: they need front_end: fastlio and pcd_save_en: true")
            return

        if self.if_pgo:
            pgo_dir = os.path.join(output_dir, "pgo")
            if not self.run_step("run_pgo", [
                self.pgo_app,
                "--scans_dir", scans_dir,
                "--pose_fname", poses,
                "--config", task.pgo_config,
                "--out", pgo_dir,
            ]):
                return
            poses = os.path.join(pgo_dir, "traj_pgo.txt")

        if self.if_clean:
            clean_dir = os.path.join(output_dir, "clean")
            if not self.run_step("run_clean", [
                self.clean_app,
                "--scans_dir", scans_dir,
                "--pose_fname", poses,
                "--config", task.clean_config,
                "--out", clean_dir,
            ]):
                return

        if self.if_ba:
            ba_dir = os.path.join(output_dir, "ba")
            traj_pgo = os.path.join(output_dir, "pgo", "traj_pgo.txt")
            poses = traj_pgo if os.path.exists(traj_pgo) else os.path.join(output_dir, "traj_log.txt")
            if not self.run_step("run_ba", [
                self.ba_app,
                "--scans_dir", scans_dir,
                "--pose_fname", poses,
                "--config", task.ba_config,
                "--out", ba_dir,
                "--images_dir", os.path.join(output_dir, "images"),
                "--extrin_fname", "/home/hsiaochuan/slam/faster-lio/config/ba_extrin_i2nav.yaml",
            ]):
                return

        if self.if_merge:
            traj_log = os.path.join(output_dir, "traj_log.txt")
            traj_pgo = os.path.join(output_dir, "pgo", "traj_pgo.txt")
            traj_ba = os.path.join(output_dir, "ba", "traj_ba.txt")
            # the best poses there are: run_ba's, else run_pgo's, else the SLAM trajectory
            poses = traj_ba if os.path.exists(traj_ba) else traj_pgo if os.path.exists(traj_pgo) else traj_log

            clean_scans_dir = os.path.join(output_dir, "clean")
            scans = clean_scans_dir if os.path.isdir(clean_scans_dir) else scans_dir
            if not self.run_step("run_merge", [
                self.merge_app,
                "--trajectory", poses,
                "--pcd_dir", scans,
                "--output", map_fname,
                "--voxel_size", "0.02",
            ]):
                return

        if self.if_mls:
            self.run_step("run_mls", [
                self.mls_app,
                "--input", map_fname,
                "--output", os.path.join(output_dir, "map_post.ply"),
            ])

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
            self.run_offline(task)
        end_time = time.time()

        self.run_mapping(task)

        result.duration_sec = round(end_time - start_time, 1)
        if os.path.exists(task.ground_truth_fname):
            traj_log = os.path.join(output_dir, "traj_log.txt")
            if os.path.exists(traj_log):
                result.ate = self.evaluate_ate(task.ground_truth_fname, traj_log)
                print("ATE RMSE LIO: {}".format(result.ate))
            traj_pgo = os.path.join(output_dir, "pgo", "traj_pgo.txt")
            if os.path.exists(traj_pgo):
                result.ate_pgo = self.evaluate_ate(task.ground_truth_fname, traj_pgo)
                print("ATE RMSE PGO: {}".format(result.ate_pgo))
            traj_ba = os.path.join(output_dir, "ba", "traj_ba.txt")
            if os.path.exists(traj_ba):
                result.ate_ba = self.evaluate_ate(task.ground_truth_fname, traj_ba)
                print("ATE RMSE BA: {}".format(result.ate_ba))
        else:
            print("Fail to evaluate the ATE RMSE")
        return result

    def evaluate_ate(self, ground_truth_fname: str, traj_fname: str) -> float:
        traj_ref = file_interface.read_tum_trajectory_file(ground_truth_fname)
        traj_est = file_interface.read_tum_trajectory_file(traj_fname)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)
        result = main_ape.ape(traj_ref, traj_est, metrics.PoseRelation.translation_part, align=True)
        return result.stats["rmse"]

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
            task = RunTask(
                bag_file=bag_file,
                config=config_fname,
                name=name,
                output_dir=output_dir,
                run_mode=dataset.run_mode,
                start=dataset.start,
                duration=dataset.duration,
                ground_truth_fname=os.path.join(dataset.ground_truth_dir, name + ".txt"),
                gnss_obs_fname=os.path.join(dataset.gnss_dir, name + ".o"),
                gnss_nav_fname=os.path.join(dataset.gnss_dir, name + ".rnx"),
                clean_config=dataset.clean_config,
                pgo_config=dataset.pgo_config,
                ba_config=dataset.ba_config,
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
                    f.write("{},\t{},\t{},\t{},\t{}\n".format(test_result.bag_name, test_result.duration_sec,
                                                              test_result.ate, test_result.ate_pgo, test_result.ate_ba))
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
        config={".": "../config/mcd_viral_ouster.yaml"},
        bag_files=[
            "/media/hsiaochuan/hsiaochuan-data/MCD_VIRAL/raw/ouster/ntu_day_02_os1_128.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )

    hilti_2022 = DatasetConfig(
        name="hilti_2022",
        config={".": "../config/hilti_2022.yaml"},
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
        config={".": "../config/ntu_viral.yaml"},
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/ntu_viral/eee_03/eee_03.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )
    mars_lvig = DatasetConfig(
        name="mars_lvig",
        config={".": "../config/mars_lvig.yaml"},
        bag_files=[
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/AMtown03.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/AMvalley03.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKisland_GNSS01.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKisland_GNSS02.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKisland_GNSS03.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKairport_GNSS01.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKairport_GNSS02.bag",
            "/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/HKairport_GNSS03.bag",
        ],
        gnss_dir="/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/GNSS",
        run_mode=RunMode.OFFLINE,
        ground_truth_dir="/media/hsiaochuan/hsiaochuan-data/MARS-LVIG/ground-truth",
    )
    m3dgr = DatasetConfig(
        name="m3dgr",
        config={".": "../config/m3dgr.yaml"},
        bag_files=[
            f"/media/hsiaochuan/hsiaochuan-data/M3DGR/{seq}.bag"
            for seq in [
                "Dark01",
                "Dark02",
                "Occlusion03",
                "Occlusion04",
                "Outdoor01",
                "Varying-illu03",
                "Varying-illu04",
                "Varying-illu05",
                "GNSS-denial01",
                "GNSS-denial02",
            ]
        ],
        run_mode=RunMode.OFFLINE,
        ground_truth_dir="/media/hsiaochuan/hsiaochuan-data/M3DGR/groundtruth",
    )
    i2nav_robot = DatasetConfig(
        name="i2nav_robot",
        config={".": "../config/i2nav_robot.yaml"},
        bag_files=[
            f"/media/hsiaochuan/hsiaochuan-data/inav-robot/{seq}.bag"
            for seq in [
                "building00",
                "building01",
                "parking00",
                "parking01",
            ]
        ],
        run_mode=RunMode.OFFLINE,
        ground_truth_dir="/media/hsiaochuan/hsiaochuan-data/inav-robot/groundtruth",
        pgo_config="../config/pgo.yaml",
    )
    all_datasets = [mcd_viral, mcd_viral_ouster, hilti_2022, fast_livo2, ntu_viral, mars_lvig, m3dgr, i2nav_robot]
    run_datasets = []
    for dataset in all_datasets:
        if dataset.name in name_list:
            run_datasets.append(dataset)
    return run_datasets


def main():
    parser = argparse.ArgumentParser(description="SLAM Test Framework")
    parser.add_argument("--datasets", nargs="+",
                        default=["i2nav_robot"],
                        help="Run only specified datasets (by name)")
    parser.add_argument("--if_delete_result_dir", action="store_true", default=True, help="Delete result dir if exists")
    parser.add_argument("--if_slam", action="store_true", default=True, help="Run SLAM")
    parser.add_argument("--if_pgo", action="store_true", default=False, help="Run run_pgo on the SLAM scans")
    parser.add_argument("--if_clean", action="store_true", default=False, help="Run run_clean, dynamic point removal")
    parser.add_argument("--if_merge", action="store_true", default=False, help="Run points_merge, writes map.pcd")
    parser.add_argument("--if_postprocess", action="store_true", default=False, help="Run points_jet on map.pcd")
    parser.add_argument("--if_rviz", action="store_true", default=True, help="Run rviz")
    parser.add_argument("--start", type=float, default=5, help="Start time (sec) for offline mode")
    parser.add_argument("--duration", type=float, default=-1.0,
                        help="Duration (sec) for offline mode, -1 for full length")
    args = parser.parse_args()
    if args.if_rviz:
        rviz = subprocess.Popen(
            ["rviz", "-d", "../rviz_cfg/loam_livox.rviz"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
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
    runner.if_pgo = args.if_pgo
    runner.if_clean = args.if_clean
    runner.if_ba = args.if_ba
    runner.if_merge = args.if_merge
    runner.if_mls = args.if_mls
    runner.run_all(datasets)


if __name__ == "__main__":
    main()

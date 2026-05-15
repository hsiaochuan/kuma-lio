"""
SLAM Test Framework
====================
General SLAM algorithm test framework, supporting multiple datasets, offline/online modes, automatic compilation and result aggregation.
"""

import subprocess
import os
import datetime
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from enum import Enum
import shutil
import colmap
import pandas as pd

# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

class RunMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass
class BagTestCase:
    """Single bag file test case"""
    bag_file: str
    config: str
    name: str = ""  # automatically extracted from bag_file
    run_mode: RunMode = RunMode.OFFLINE

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.bag_file).stem


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    bag_files: List[str]
    config: str  # single config (BotanicGarden)
    start: float = 0.0
    duration: float = -1.0
    config_map: Dict[str, str] = field(default_factory=dict)  # keyword → config (MCD_VIRAL)
    run_mode: RunMode = RunMode.OFFLINE
    def resolve_config(self, bag_name: str) -> str:
        """Select configuration file based on bag name"""
        for keyword, cfg in self.config_map.items():
            if keyword in bag_name:
                return cfg
        return self.config


@dataclass
class TestResult:
    bag_name: str
    bag_file: str
    points_count: int = 0
    duration_sec: float = 0.0


@dataclass
class SuiteResult:
    """Overall test suite result"""
    dataset_name: str
    git_branch: str
    git_commit: str
    start_time: str
    end_time: str = ""
    results: List[TestResult] = field(default_factory=list)


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
            lvba_app: str = "../build/lvba",
            points_color_app: str = "../build/points_color",

            build_dir: str = "../build",
            build_jobs: int = 4,
            output_root: str = "./test_results",
            rviz_config: str = "../rviz_cfg/loam_livox.rviz",
            online_wait: int = 5,

            if_delete_result_dir: bool = False,
            if_slam: bool = True,
            if_lvba: bool = True,
            if_postprocess: bool = False,
    ):
        self.offline_app = offline_app
        self.online_app = online_app
        self.points_post_app = points_post_app
        self.lvba_app = lvba_app
        self.points_color_app = points_color_app

        self.build_dir = build_dir
        self.build_jobs = build_jobs
        self.output_root = output_root
        self.rviz_config = rviz_config
        self.online_wait = online_wait

        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_ts = ts

        self.if_delete_result_dir = if_delete_result_dir
        self.if_slam = if_slam
        self.if_lvba = if_lvba
        self.if_postprocess = if_postprocess

    # ── Build ──────────────────────────────────

    def build(self) -> bool:
        print(f"Building project (jobs={self.build_jobs}) ...")
        try:
            subprocess.run(
                ["make", "-C", self.build_dir, "-j", str(self.build_jobs)],
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

    def _run_offline(self, bag_file: str, config: str, output_dir: str, start: float, duration: float) -> bool:
        try:
            subprocess.run(
                [self.offline_app,
                 "--config_file", config,
                 "--bag_file", bag_file,
                 "--output_dir", output_dir,
                 "--start", str(start),
                 '--duration', str(duration),
                 ],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  run_mapping_offline exited with code {e.returncode}")
            return False

    def run_post_process(self, output_dir: str):
        subprocess.run([
            self.points_post_app,
            '--input', os.path.join(output_dir, "final.pcd"),
            '--output', os.path.join(output_dir, "final_post.pcd"),
        ], check=True)
    def run_points_color(self, output_dir: str):
        subprocess.run([
            self.points_color_app,
            '--lidar_points_fname', os.path.join(output_dir, "final_post.pcd"),
            '--color_points_fname', os.path.join(output_dir, "final_post_color.pcd"),
            '--images_dir', os.path.join(output_dir, "images"),
            '--colmap_result', os.path.join(output_dir, "colmap_result"),
        ], check=True)
    def run_lvba(self, output_dir: str):
        subprocess.run([
            self.lvba_app,
            '--cam_trajectory', os.path.join(output_dir, "cam_final.txt"),
            '--lidar_points_fname', os.path.join(output_dir, "final.pcd"),
            '--database', os.path.join(output_dir, "database.db"),
            '--colmap_output', os.path.join(output_dir, "colmap_result"),
        ], check=True)
    def _run_online(self, bag_file: str, config: str, output_dir: str) -> bool:
        roscore = rviz = online_proc = None
        try:
            roscore = subprocess.Popen(
                ["roscore"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            rviz = subprocess.Popen(
                ["rviz", "-d", self.rviz_config],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(self.online_wait)
            online_proc = subprocess.Popen(
                [self.online_app,
                 "--output_dir", output_dir,
                 "--config_fname", config]
            )
            subprocess.run(
                ["rosbag", "play", bag_file],
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
    def run_time_analysis(self, time_log: str, time_cost_summ_f: str):
        result_f = open(time_cost_summ_f, "w")
        df = pd.read_csv(time_log, engine="python", skipinitialspace=True)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, [c for c in df.columns if c != ""]]
        for c in df:
            x = pd.to_numeric(df[c], errors="coerce")
            x = x[x.notna()]  # remove nan
            fmt = "%-35s: num=%d, ave=%f, std=%f, max=%f, min=%f\n"
            result_f.write(fmt % (c, len(x), x.mean(), x.std(), x.max(), x.min()))
        result_f.close()
    def run_single(self,
                   bag_file: str,
                   config: str,
                   output_dir: str, run_mode: RunMode, start: float, duration: float) -> TestResult:
        name = Path(bag_file).stem
        result = TestResult(
            bag_name=name,
            bag_file=bag_file,
        )
        print(f"{name}  [{run_mode}]")

        # copy config
        shutil.copy(config, os.path.join(output_dir, "config.yaml"))

        # pipeline
        start_time = time.time()
        if self.if_slam:
            if run_mode == RunMode.OFFLINE:
                self._run_offline(bag_file, config, output_dir, start, duration)
            else:
                self._run_online(bag_file, config, output_dir)
            self.run_time_analysis(os.path.join(output_dir, "time_log.txt"), os.path.join(output_dir, "time_cost_summ.txt"))
        if self.if_postprocess:
            self.run_post_process(output_dir)
        if os.path.exists(os.path.join(output_dir, "images")) and self.if_lvba:
            colmap.run_colmap(output_dir, extract_match=True, mapping=False, triangulate=True)
            # self.run_lvba(output_dir)
            # self.run_points_color(output_dir)
        end_time = time.time()


        result.duration_sec = round(end_time - start_time, 1)
        result.points_count = count_points_in_dir(os.path.join(output_dir, "maps"))

        return result

    # ── Dataset run ────────────────────────────

    def run_dataset(self, dataset: DatasetConfig) -> SuiteResult:
        branch, commit = get_git_info()
        suite = SuiteResult(
            dataset_name=dataset.name,
            git_branch=branch,
            git_commit=commit,
            start_time=datetime.datetime.now().isoformat(),
        )

        print(f"\n{'=' * 50}")
        print(f"   Dataset: {dataset.name}  ({len(dataset.bag_files)} bags)")
        print(f"   branch={branch}  commit={commit[:8]}")

        for bag_file in dataset.bag_files:
            name = Path(bag_file).stem
            config = dataset.resolve_config(name)
            output_dir = os.path.join(
                Path(bag_file).parent,
                name + "_faster_lio_result"
            )
            if os.path.exists(output_dir) and self.if_delete_result_dir:
                print(f"Result dir {output_dir} already exists, remove it")
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            result = self.run_single(bag_file, config, output_dir, dataset.run_mode, dataset.start, dataset.duration)
            result.dataset = dataset.name
            suite.results.append(result)

        suite.end_time = datetime.datetime.now().isoformat()
        return suite

    # ── Write result files ────────────────────────────

    def write_txt_report(self, suite: SuiteResult) -> str:
        fname = os.path.join(
            self.output_root,
            f"{suite.dataset_name.replace(' ', '_')}_{self.run_ts}.txt"
        )
        os.makedirs(self.output_root, exist_ok=True)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(f"DATA_SET    = {suite.dataset_name}\n")
            f.write(f"GIT_BRANCH  = \"{suite.git_branch}\"\n")
            f.write(f"GIT_COMMIT  = \"{suite.git_commit}\"\n")
            f.write(f"START_TIME  = {suite.start_time}\n")
            f.write(f"END_TIME    = {suite.end_time}\n")
            f.write("-" * 65 + "\n")
            for r in suite.results:
                f.write(
                    "{},\t{}\n".format(r.bag_name, r.points_count)
                )
            f.write("-" * 65 + "\n")
        return fname

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
            self.write_txt_report(suite_result)
            suite_results.append(suite_result)

        return suite_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def DatasetsList(name_list: List[str]) -> List[DatasetConfig]:
    """Default datasets equivalent to original two test scripts"""
    botanic_garden = DatasetConfig(
        name="botanic_garden",
        config="../config/botanic_garden.yaml",
        bag_files=[
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
        ],
        run_mode=RunMode.OFFLINE,
    )

    mcd_viral = DatasetConfig(
        name="mcd_viral",
        config="../config/mcd_viral_handheld.yaml",
        config_map={
            "ntu": "../config/mcd_viral_atv.yaml",
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
    )

    new_college = DatasetConfig(
        name="new_college",
        config="../config/new_college.yaml",
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/New_College/rooster_2020-07-10-09-23-18_0.bag"
        ],
        run_mode=RunMode.OFFLINE,
    )

    hilti_2022 = DatasetConfig(
        name="hilti_2022",
        config="../config/hilti_2022.yaml",
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/Hilti2022/exp04_construction_upper_level.bag",
            "/mnt/data/home/hsiaochuan/data/Hilti2022/exp07_long_corridor.bag",
            "/mnt/data/home/hsiaochuan/data/Hilti2022/exp11_lower_gallery.bag",
            "/mnt/data/home/hsiaochuan/data/Hilti2022/exp14_basement_2.bag",
            "/mnt/data/home/hsiaochuan/data/Hilti2022/exp18_corridor_lower_gallery_2.bag"
            "/mnt/data/home/hsiaochuan/data/Hilti2022/exp21_outside_building.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )

    fast_livo2 = DatasetConfig(
        name="fast_livo2",
        config="../config/fast_livo2.yaml",
        config_map={
            "Retail_Street": "../config/fast_livo2.yaml",
            "CBD_Building_01": "../config/fast_livo2.yaml",
            "Bright_Screen_Wall": "../config/fast_livo2.yaml",

            "HKU_Landmark": "../config/fast_livo2_1.yaml",
            "HKU_Centennial_Garden": "../config/fast_livo2_1.yaml",
            "HKU_Main_Building": "../config/fast_livo2_1.yaml",
            "HKU_Lecture_Center_01":"../config/fast_livo2_1.yaml",

            "SYSU_01":"../config/fast_livo2_2.yaml",
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

    urban_loco = DatasetConfig(
        name="urban_loco",
        config="../config/urban_loco.yaml",
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/urban_loco/test2.bag",
        ],
        run_mode=RunMode.OFFLINE,
    )
    all_datasets = [botanic_garden, mcd_viral, new_college, hilti_2022, fast_livo2, urban_loco]
    run_datasets = []
    for dataset in all_datasets:
        if dataset.name in name_list:
            run_datasets.append(dataset)
    return run_datasets
def main():
    parser = argparse.ArgumentParser(description="SLAM Test Framework")
    parser.add_argument("--datasets", nargs="+",
                        default=["mcd_viral", "botanic_garden", "new_college", "fast_livo2", "hilti_2022", "urban_loco"],
                        help="Run only specified datasets (by name)")
    parser.add_argument("--if_delete_result_dir", action="store_true", default=True, help="Delete result dir if exists")
    parser.add_argument("--if_slam", action="store_true", default=True, help="Run SLAM")
    parser.add_argument("--if_lvba", action="store_true", default=True, help="Run LVBA")
    parser.add_argument("--if_postprocess", action="store_true", default=False, help="Run points post-processing")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (sec) for offline mode")
    parser.add_argument("--duration", type=float, default=-1.0, help="Duration (sec) for offline mode, -1 for full length")
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

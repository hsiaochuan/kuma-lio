"""
SLAM Test Framework
====================
General SLAM algorithm test framework, supporting multiple datasets, offline/online modes, automatic compilation and result aggregation.
"""

import subprocess
import os
import datetime
import time
import yaml
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from enum import Enum


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
    config_map: Dict[str, str] = field(default_factory=dict)  # keyword → config (MCD_VIRAL)
    run_mode: RunMode = RunMode.OFFLINE
    enabled: bool = True

    def resolve_config(self, bag_name: str) -> str:
        """Select configuration file based on bag name"""
        for keyword, cfg in self.config_map.items():
            if keyword in bag_name:
                return cfg
        return self.config


@dataclass
class TestResult:
    """Single bag test result"""
    dataset: str
    bag_name: str
    bag_file: str
    config_file: str
    points_count: int = 0
    success: bool = True
    duration_sec: float = 0.0
    output_dir: str = ""


@dataclass
class SuiteResult:
    """Overall test suite result"""
    dataset_name: str
    git_branch: str
    git_commit: str
    start_time: str
    end_time: str = ""
    results: List[TestResult] = field(default_factory=list)
    build_success: bool = True


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
            build_dir: str = "../build",
            build_jobs: int = 4,
            output_root: str = "./test_results",
            rviz_config: str = "../rviz_cfg/loam_livox.rviz",
            online_wait: int = 5,
    ):
        self.offline_app = offline_app
        self.online_app = online_app
        self.build_dir = build_dir
        self.build_jobs = build_jobs
        self.output_root = output_root
        self.rviz_config = rviz_config
        self.online_wait = online_wait

        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_ts = ts

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

    def _run_offline(self, bag_file: str, config: str, output_dir: str) -> bool:
        try:
            subprocess.run(
                [self.offline_app,
                 "--config_file", config,
                 "--bag_file", bag_file,
                 "--output_dir", output_dir,
                 "--start", '0',
                 '--duration', '-1',
                 ],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  run_mapping_offline exited with code {e.returncode}")
            return False

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

    def run_single(self, bag_file: str, config: str,
                   output_dir: str, run_mode: RunMode) -> TestResult:
        name = Path(bag_file).stem
        os.makedirs(os.path.join(output_dir, "maps"), exist_ok=True)

        result = TestResult(
            dataset="",
            bag_name=name,
            bag_file=bag_file,
            config_file=config,
            output_dir=output_dir,
        )

        print(f"{name}  [{run_mode}]")
        start_time = time.time()

        if run_mode == RunMode.OFFLINE:
            ok = self._run_offline(bag_file, config, output_dir)
        else:
            ok = self._run_online(bag_file, config, output_dir)

        result.duration_sec = round(time.time() - start_time, 1)
        result.success = ok

        maps_dir = os.path.join(output_dir, "maps")
        result.points_count = count_points_in_dir(maps_dir)

        status = "OK" if ok else "Fail"
        print(
            f"{status} points={result.points_count:,}  time={result.duration_sec}s"
        )
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
            result = self.run_single(bag_file, config, output_dir, dataset.run_mode)
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
            f.write(f"BUILD_OK    = {suite.build_success}\n\n")
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

        suites = []
        for ds in datasets:
            if not ds.enabled:
                print(f"  Skipping dataset: {ds.name}")
                continue
            suite_result = self.run_dataset(ds)
            suite_result.build_success = build_ok
            self.write_txt_report(suite_result)
            suites.append(suite_result)

        return suites


# ──────────────────────────────────────────────
# YAML Configuration Loading
# ──────────────────────────────────────────────

def load_config(config_path: str) -> Tuple[SLAMTestRunner, List[DatasetConfig]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    runner_cfg = cfg.get("runner", {})
    runner = SLAMTestRunner(
        offline_app=runner_cfg.get("offline_app", "../build/run_mapping_offline"),
        online_app=runner_cfg.get("online_app", "../build/run_mapping_online"),
        build_dir=runner_cfg.get("build_dir", "../build"),
        build_jobs=runner_cfg.get("build_jobs", 4),
        output_root=runner_cfg.get("output_root", "./test_results"),
        rviz_config=runner_cfg.get("rviz_config", "../rviz_cfg/loam_livox.rviz"),
        online_wait=runner_cfg.get("online_wait", 5),
    )

    datasets = []
    for ds_cfg in cfg.get("datasets", []):
        datasets.append(DatasetConfig(
            name=ds_cfg["name"],
            bag_files=ds_cfg["bag_files"],
            config=ds_cfg.get("config", ""),
            config_map=ds_cfg.get("config_map", {}),
            run_mode=RunMode(ds_cfg.get("run_mode", "offline")),
            enabled=ds_cfg.get("enabled", True),
        ))

    return runner, datasets


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLAM Test Framework")
    parser.add_argument("--datasets", nargs="+", default=["mcd_viral", "botanic", "new_college"],
                        help="Run only specified datasets (by name)")
    args = parser.parse_args()

    data_name_list = args.datasets
    runner = SLAMTestRunner()
    datasets = DatasetsList(data_name_list)

    runner.run_all(datasets)


def DatasetsList(name_list: List[str]) -> List[DatasetConfig]:
    """Default datasets equivalent to original two test scripts"""
    botanic = DatasetConfig(
        name="Botanic Garden",
        config="../config/BotanicGarden.yaml",
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
        config="../config/MCD_VIRAL_HandHeld.yaml",
        config_map={
            "ntu": "../config/MCD_VIRAL_ATV.yaml",
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
        config="../config/New_College.yaml",
        bag_files=[
            "/mnt/data/home/hsiaochuan/data/New_College/rooster_2020-07-10-09-23-18_0.bag"
        ],
        run_mode=RunMode.OFFLINE,
    )

    all_datasets = [botanic, mcd_viral, new_college]
    run_datasets = []
    for dataset in all_datasets:
        if dataset.name in name_list:
            run_datasets.append(dataset)
    return run_datasets


if __name__ == "__main__":
    main()

//
// Created by hsiaochuan on 2026/03/23.
//
#include "bareg.h"
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <point_cluster.h>
#include <boost/filesystem.hpp>
#include "lidar_simulator.h"
#include "pose3.h"
#include "voxel_map.h"
using namespace faster_lio;
TEST(Bareg, first) {
    // lidar config
    LidarConfig cfg;
    cfg.num_beams = 16;
    cfg.vfov_min_deg = -15.0;
    cfg.vfov_max_deg = +15.0;
    cfg.h_res_deg = 0.5;
    cfg.max_range = 20.0;
    cfg.range_noise_std = 0.01;
    cfg.rotation_hz = 10.0;

    // generate traj
    Trajectory traj = TrajectoryGenerator::circular(Eigen::Vector2d(0, 0),  // center
                                                    3.0,                    // radius 8m
                                                    2.0,                    // linear speed 2 m/s
                                                    0.5,                    // motion height z=0.5m
                                                    20.0,                    // total duration 5s
                                                    0.005);
    LidarSimulator sim(cfg, traj);
    sim.scene() = scene_factory::make_demo_room();
    sim.set_seed(42);

    // generate the scan
    double start_time = 0., end_time = 10.;
    auto lidar_scans = sim.scan_sequence(start_time, end_time, true);

    // get the stamped poses
    Trajectory lidar_stamped_poses;
    for (double scan_time = start_time; scan_time < end_time; scan_time += 1. / cfg.rotation_hz) {
        auto lidar_pose = sim.query_lidar_pose(scan_time);
        lidar_stamped_poses.emplace_back(scan_time, lidar_pose);
    }
}

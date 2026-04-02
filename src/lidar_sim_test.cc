
/**
 * @file  main.cpp
 * @brief Motion multi-line LiDAR simulator usage example
 *
 * Compile:
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 main.cpp -o lidar_sim
 */

#include <gtest/gtest.h>
#include <iostream>
#include "lidar_simulator.h"
using namespace lidar_sim;

TEST(LidarSim, LinearMotion) {
    LidarConfig cfg;
    cfg.num_beams = 16;
    cfg.vfov_min_deg = -15.0;
    cfg.vfov_max_deg = +15.0;
    cfg.h_res_deg = 0.5;
    cfg.max_range = 20.0;
    cfg.range_noise_std = 0.02;
    cfg.rotation_hz = 10.0;

    // Start from (0, -18, 0), travel at 1 m/s along +Y for 3 seconds
    Trajectory traj =
        TrajectoryGenerator::linear(
            Eigen::Vector3d(0, -5, 0),
            Eigen::Vector3d(0, 1, 0),
            10,
            0.005);
    LidarSimulator sim(cfg, traj);
    sim.scene() = scene_factory::make_demo_room();
    sim.set_seed(42);

    auto full_clouds = sim.scan_sequence(0.0, 10);  // 3 sec × 10 Hz = 30 frames
}
TEST(LidarSim, CircularMotion) {
    LidarConfig cfg;
    cfg.num_beams = 32;
    cfg.vfov_min_deg = -15.0;
    cfg.vfov_max_deg = +15.0;
    cfg.h_res_deg = 0.5;
    cfg.max_range = 25.0;
    cfg.range_noise_std = 0.02;
    cfg.rotation_hz = 10.0;

    // Circular: radius 8m, linear speed 2 m/s, one rotation ≈ 25s, simulate 5s
    Trajectory traj = TrajectoryGenerator::circular(Eigen::Vector2d(0, 0),  // center
                                                    8.0,                    // radius 8m
                                                    2.0,                    // linear speed 2 m/s
                                                    0.5,                    // motion height z=0.5m
                                                    10.0,                    // total duration 5s
                                                    0.005);

    LidarSimulator sim(cfg, traj);
    sim.scene() = scene_factory::make_demo_room();
    sim.set_seed(123);
    auto clouds = sim.scan_sequence(0.0, 10);
}
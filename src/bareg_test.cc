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

    // add noise in lidar traj
    for (int i = 0; i < lidar_stamped_poses.size(); ++i) {
        Pose3 pose = lidar_stamped_poses[i].pose;
        pose.AddNoise(0.01, 0.001);
        lidar_stamped_poses[i].pose = pose.GetIsometry3d();
    }

    // convert the scan
    std::vector<PointCloud> scans;
    scans.resize(lidar_scans.size());
    for (int scan_i = 0; scan_i < lidar_scans.size(); ++scan_i) {
        LidarScan &lidar_scan = lidar_scans[scan_i];
        PointCloud &scan = scans[scan_i];
        scan.resize(lidar_scan.size());
        for (int p_j = 0; p_j < lidar_scan.size(); ++p_j) {
            scan[p_j] = lidar_scan[p_j].Point();
            scan[p_j].timestamp = p_j;
            scan[p_j].intensity = scan_i;
        }
    }

    // add to map
    faster_lio::VoxelMap::Config config;
    faster_lio::VoxelMap map(config);
    for (int i = 0; i < scans.size(); ++i) {
        PointCloud::Ptr scan_world(new PointCloud);
        pcl::transformPointCloud(scans[i], *scan_world, lidar_stamped_poses[i].pose.matrix());
        map.AddCloud(scan_world);
    }

    // add to problem
    std::vector<Pose3> poses;
    poses.resize(lidar_stamped_poses.size());
    for (int i = 0; i < lidar_stamped_poses.size(); ++i) {
        poses[i].SetTrans(lidar_stamped_poses[i].pose.translation());
        poses[i].SetQuat(Eigen::Quaterniond(lidar_stamped_poses[i].pose.linear()));
    }
    ceres::Problem problem;
    for (int i = 0; i < poses.size(); ++i) {
        problem.AddParameterBlock(poses[i].QuatData(), 4, new ceres::QuaternionParameterization);
        problem.AddParameterBlock(poses[i].PosData(), 3);
    }

    // add constrain
    int plane_voxel_count = 0;
    for (const auto &iter: map.voxel_map_) {
        if (iter.second->normal != Eigen::Vector3d::Zero()) {
            plane_voxel_count++;
            auto grid = iter.second;
            std::unordered_map<int, PointCluster> clusters;
            for (Point &point: grid->points) {
                int scan_i = point.intensity;
                int p_j = point.timestamp;
                Eigen::Vector3d point_lidar = scans[scan_i].points[p_j].getVector3fMap().cast<double>();
                clusters[scan_i].Push(point_lidar);
            }

            for (auto &[scan_k, cluster_k]: clusters) {
                int nk = cluster_k.N;
                Eigen::Matrix3d cov_k = cluster_k.Cov();
                Eigen::Vector3d mean_k = cluster_k.Mean();
                Eigen::Vector3d normal = grid->normal;
                Eigen::Vector3d center = grid->cluster.Mean();
                auto costs = BaregCostFunctionCreate(nk, cov_k, mean_k, normal, center, 1.0);
                if (costs[0])
                    problem.AddResidualBlock(costs[0], nullptr, poses[scan_k].QuatData(), poses[scan_k].PosData());
                if (costs[1])
                    problem.AddResidualBlock(costs[1], nullptr, poses[scan_k].QuatData());
                if (costs[2])
                    problem.AddResidualBlock(costs[2], nullptr, poses[scan_k].QuatData());
            }
        }
    }

    std::cout << "voxel count: " << map.voxel_map_.size() << std::endl;
    std::cout << "plane voxel count: " << plane_voxel_count << std::endl;

    // merge before points
    PointCloud::Ptr merged(new PointCloud);
    for (int i = 0; i < poses.size(); ++i) {
        PointCloud::Ptr scan_world(new PointCloud);
        pcl::transformPointCloud(scans[i], *scan_world, poses[i].GetMat4d());
        *merged += *scan_world;
    }
    pcl::VoxelGrid<PointType> sampler;
    sampler.setLeafSize(0.05, 0.05, 0.05);
    sampler.setInputCloud(merged);
    sampler.filter(*merged);
    pcl::io::savePCDFileBinary("./before.pcd", *merged);
    int before_opt_points_size = merged->points.size();
    std::cout << "before points count: " << merged->points.size() << std::endl;

    // solve
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // merge after points
    merged->clear();
    for (int i = 0; i < poses.size(); ++i) {
        PointCloud::Ptr scan_world(new PointCloud);
        pcl::transformPointCloud(scans[i], *scan_world, poses[i].GetMat4d());
        *merged += *scan_world;
    }
    sampler.setInputCloud(merged);
    sampler.filter(*merged);
    pcl::io::savePCDFileBinary("./after.pcd", *merged);
    int after_opt_points_size = merged->points.size();
    std::cout << "after points count: " << merged->points.size() << std::endl;
    EXPECT_GT(before_opt_points_size, after_opt_points_size);
}

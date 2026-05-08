#pragma once
// clang-format off
#include <pcl/registration/gicp.h>
#include <pcl/kdtree/kdtree_flann.h>
// clang-format on

#include <ceres/ceres.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl_conversions/pcl_conversions.h>
#include <point_cluster.h>
#include <yaml-cpp/yaml.h>
#include <boost/container/flat_map.hpp>
#include <boost/progress.hpp>
#include <cmath>
#include <future>
#include <unordered_set>
#include "bareg.h"
#include "common_lib.h"
#include "cost_functions.h"
#include "pose3.h"
#include "stamp_pose.h"
#include "thread_pool.h"
#include "voxel_map.h"
namespace fs = boost::filesystem;
namespace faster_lio {
using scan_pair_t = uint64_t;
struct ScanPair {
    scan_t scan_id1_;
    scan_t scan_id2_;
    ScanPair(scan_t scan_id1, scan_t scan_id2) : scan_id1_(scan_id1), scan_id2_(scan_id2) {
        CHECK(scan_id1 != scan_id2);
        if (scan_id1 > scan_id2) {
            std::swap(scan_id1_, scan_id2_);
        }
    }

    bool operator==(const ScanPair& other) const {
        return (scan_id1_ == other.scan_id1_ && scan_id2_ == other.scan_id2_);
    }
};
scan_pair_t ScanPairToId(const ScanPair& pair) {
    return (static_cast<scan_pair_t>(pair.scan_id1_) << 32) | static_cast<scan_pair_t>(pair.scan_id2_);
}
ScanPair IdToScanPair(scan_pair_t id) {
    ScanPair pair(static_cast<scan_t>(id >> 32), static_cast<scan_t>(id & 0xffffffff));
    return pair;
}
}  // namespace faster_lio
namespace std {
template <>
struct hash<faster_lio::ScanPair> {
    size_t operator()(const faster_lio::ScanPair& scan_pair) const { return faster_lio::ScanPairToId(scan_pair); }
};
}  // namespace std

namespace faster_lio {
struct PairData {
    int source_points_count = 0;
    int target_points_count = 0;
    Pose3 ab_rel_pose;
    double average_error;
    int corres_count = 0;
};

class GlobalOptimizor {
   public:
    struct Options {
        // loop weight
        bool lc_enable = true;
        double loop_weight = 100.0;

        // ba
        bool ba_enable = false;
        int ba_iters = 3;

        // keyframe selection thresholds
        double keyframe_dist_threshold = 1.0;  // meters
        double keyframe_angle_threshold = 10;
        double keyframe_time_threshold = 1.0;  // seconds

        // lc
        bool output_all_loop_reigster_result = true;
        double lc_detect_dist_thr = 5.0;
        double lc_detect_temporal_dist_thr = 60.0;
        int sub_map_interval = 5;
        void LoadFromYaml(const std::string& config_fname) {
            try {
                YAML::Node config = YAML::LoadFile(config_fname);
                lc_enable = config["global"]["lc_enable"].as<bool>();
                loop_weight = config["global"]["loop_weight"].as<double>();

                ba_enable = config["global"]["ba_enable"].as<bool>();
                ba_iters = config["global"]["ba_iters"].as<int>();

                keyframe_dist_threshold = config["global"]["keyframe_dist_threshold"].as<double>();
                keyframe_angle_threshold = config["global"]["keyframe_angle_threshold"].as<double>();
                keyframe_time_threshold = config["global"]["keyframe_time_threshold"].as<double>();

                output_all_loop_reigster_result = config["global"]["output_all_loop_reigster_result"].as<bool>();
                lc_detect_dist_thr = config["global"]["lc_detect_dist_thr"].as<double>();
                lc_detect_temporal_dist_thr = config["global"]["lc_detect_temporal_dist_thr"].as<double>();
                sub_map_interval = config["global"]["sub_map_interval"].as<int>();
            } catch (std::exception& e) {
                throw std::runtime_error("Fail to load the parameter: " + std::string(e.what()));
            }
        }
    };

    GlobalOptimizor::Options options_;
    boost::container::flat_map<scan_t, ScanFrame::Ptr> scans_;
    boost::container::flat_map<scan_t, ScanFrame::Ptr> keyscans_;
    std::unordered_map<ScanPair, PairData> loops_buf;
    std::string output_dir;

    void AddScan(ScanFrame::Ptr scan) { scans_[scan->scan_id] = scan; }

    void ScanFilter() {
        keyscans_.clear();
        if (scans_.empty()) {
            std::cerr << "No scans to filter." << std::endl;
            return;
        }

        auto it = scans_.begin();
        ScanFrame::Ptr last_keyframe = it->second;
        Pose3 last_pose = last_keyframe->GetWorldFromBody();
        double last_time = it->second->GetTimestamp();
        keyscans_[it->first] = last_keyframe;

        for (++it; it != scans_.end(); ++it) {
            const ScanFrame::Ptr& scan = it->second;
            const Pose3 pose = scan->GetWorldFromBody();
            const double dt = it->second->GetTimestamp() - last_time;
            const double trans = (pose.Trans() - last_pose.Trans()).norm();
            const double angle = LogMat(last_pose.Mat3d().transpose() * pose.Mat3d()).norm() * 180.0 / M_PI;

            if (trans >= options_.keyframe_dist_threshold || angle >= options_.keyframe_angle_threshold ||
                dt >= options_.keyframe_time_threshold) {
                keyscans_[it->first] = scan;
                last_pose = pose;
                last_time = it->second->GetTimestamp();
            }
        }

        // filter key scans
        pcl::PointCloud<pcl::PointXYZI>::Ptr keyscan_pos_points(new pcl::PointCloud<pcl::PointXYZI>);
        for (auto& [scan_id, scan] : keyscans_) {
            pcl::PointXYZI pt;
            pt.getVector3fMap() = scan->GetWorldFromBody().Trans().cast<float>();
            pt.intensity = scan_id;
            keyscan_pos_points->push_back(pt);
        }
        pcl::UniformSampling<pcl::PointXYZI> sampler(true);
        sampler.setInputCloud(keyscan_pos_points);
        sampler.setRadiusSearch(0.05);
        pcl::PointCloud<pcl::PointXYZI>::Ptr keyscan_pos_points_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        sampler.filter(*keyscan_pos_points_filtered);
        auto remove_ids = sampler.getRemovedIndices();
        for (const auto& id : *remove_ids) {
            pcl::PointXYZI pt = keyscan_pos_points->at(id);
            auto scan_id = static_cast<scan_t>(pt.intensity);
            keyscans_.erase(scan_id);
        }

        std::cout << "Total scans: " << scans_.size() << ", keyscans: " << keyscans_.size() << std::endl;
    }
    void BundleAdjustment() {
        // convert the points
        for (const auto& [scan_id, scan] : keyscans_) {
            PointCloud::Ptr points = scan->GetScan();
            for (int j = 0; j < points->size(); ++j) {
                points->points[j].timestamp = scan_id;
                points->points[j].intensity = j;
            }
        }

        // add to map
        faster_lio::VoxelMap::Config config;
        faster_lio::VoxelMap map(config);
        for (const auto& [scan_id, scan] : keyscans_) {
            PointCloud::Ptr scan_world(new PointCloud);
            pcl::transformPointCloud(*scan->GetScan(), *scan_world, scan->GetWorldFromBody().Mat4d());
            map.AddPoints(scan_world);
        }

        map.Finish(10,0.025);
        // add to problem
        ceres::Problem problem;
        for (const auto& [scan_id, scan] : keyscans_) {
            problem.AddParameterBlock(scan->world_from_body.QuatData(), 4, new ceres::EigenQuaternionParameterization);
            problem.AddParameterBlock(scan->world_from_body.PosData(), 3);
        }

        problem.SetParameterBlockConstant(keyscans_.begin()->second->world_from_body.QuatData());
        problem.SetParameterBlockConstant(keyscans_.begin()->second->world_from_body.PosData());

        // add constrain
        int plane_voxel_count = 0;
        for (const auto& iter : map.voxel_map_) {
            if (iter.second->plane_coeff != Vec4::Zero()) {
                plane_voxel_count++;
                auto grid = iter.second;
                std::unordered_map<scan_t, PointCluster> clusters;
                for (int inlier_idx : iter.second->inliers_) {
                    Point& point = grid->points->at(inlier_idx);
                    int point_id = static_cast<int>(point.intensity);
                    scan_t scan_id = static_cast<scan_t>(point.timestamp);
                    CHECK(keyscans_.count(scan_id));
                    PointCloud::Ptr scan_points = keyscans_[scan_id]->GetScan();
                    CHECK(scan_points && point_id >= 0 && point_id < scan_points->size());
                    Eigen::Vector3d p = scan_points->points[point_id].getVector3fMap().cast<double>();
                    clusters[scan_id].Push(p);
                }

                for (auto& [scan_k, cluster_k] : clusters) {
                    int nk = cluster_k.N;
                    Eigen::Matrix3d cov_k = cluster_k.Cov();
                    Eigen::Vector3d mean_k = cluster_k.Mean();
                    Vec4 plane_coeff = grid->plane_coeff;
                    auto costs = BaregCostFunctionCreate(nk, cov_k, mean_k, plane_coeff, 1.0);
                    if (costs[0])
                        problem.AddResidualBlock(costs[0], nullptr, keyscans_[scan_k]->world_from_body.QuatData(),
                                                 keyscans_[scan_k]->world_from_body.PosData());
                    if (costs[1])
                        problem.AddResidualBlock(costs[1], nullptr, keyscans_[scan_k]->world_from_body.QuatData());
                    if (costs[2])
                        problem.AddResidualBlock(costs[2], nullptr, keyscans_[scan_k]->world_from_body.QuatData());
                }
            }
        }

        std::cout << "voxel count: " << map.voxel_map_.size() << std::endl;
        std::cout << "plane voxel count: " << plane_voxel_count << std::endl;

        // solve
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (summary.IsSolutionUsable()) {
            std::cout << "Bundle adjustment success." << std::endl;
        } else {
            std::cerr << "Bundle adjustment failed: " << std::endl;
        }
        std::cout << summary.BriefReport() << std::endl;
    }
    void PoseGraphOptimize() {
        if (keyscans_.empty()) {
            std::cerr << "No keyscans to optimize." << std::endl;
            return;
        }
        if (loops_buf.empty()) {
            std::cout << "No loop-closures to optimize." << std::endl;
            return;
        }
        ceres::Problem problem;
        for (const auto& [scan_id, scan] : keyscans_) {
            problem.AddParameterBlock(scan->world_from_body.QuatData(), 4, new ceres::EigenQuaternionParameterization);
            problem.AddParameterBlock(scan->world_from_body.PosData(), 3);
        }

        // Fix gauge freedom by anchoring the first keyscan.
        auto first_it = keyscans_.begin();
        problem.SetParameterBlockConstant(first_it->second->world_from_body.QuatData());
        problem.SetParameterBlockConstant(first_it->second->world_from_body.PosData());

        const double odom_pos_weight = 1.0;
        const double odom_rot_weight = 1.0;
        const double loop_pos_weight = options_.loop_weight;
        const double loop_rot_weight = options_.loop_weight;

        // Add sequential constraints between neighboring keyscans.
        for (auto it = keyscans_.begin(); it != keyscans_.end();) {
            auto it_next = std::next(it);
            if (it_next == keyscans_.end()) break;
            ScanFrame::Ptr scan_i = it->second;
            ScanFrame::Ptr scan_j = it_next->second;
            Pose3 rel_ij = scan_i->world_from_body.GetInverse() * scan_j->world_from_body;
            ceres::CostFunction* cost = RelativePoseCostFunctor::Create(rel_ij, odom_pos_weight, odom_rot_weight);
            problem.AddResidualBlock(cost, nullptr, scan_i->world_from_body.QuatData(),
                                     scan_i->world_from_body.PosData(), scan_j->world_from_body.QuatData(),
                                     scan_j->world_from_body.PosData());
            it = it_next;
        }

        // Add loop-closure constraints.
        if (options_.lc_enable) {
            for (const auto& [pair, pair_data] : loops_buf) {
                if (keyscans_.count(pair.scan_id1_) == 0 || keyscans_.count(pair.scan_id2_) == 0) continue;
                ScanFrame::Ptr scan_a = keyscans_[pair.scan_id1_];
                ScanFrame::Ptr scan_b = keyscans_[pair.scan_id2_];
                ceres::CostFunction* cost =
                    RelativePoseCostFunctor::Create(pair_data.ab_rel_pose, loop_pos_weight, loop_rot_weight);
                problem.AddResidualBlock(cost, nullptr, scan_a->world_from_body.QuatData(),
                                         scan_a->world_from_body.PosData(), scan_b->world_from_body.QuatData(),
                                         scan_b->world_from_body.PosData());
            }
        }

        // solve
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    std::unordered_map<ScanPair, PairData> DetectLoopClosure() {
        std::unordered_map<ScanPair, PairData> candidates;
        std::unordered_set<scan_t> scans_in_loop;
        for (const auto& [pair, pair_data] : loops_buf) {
            scans_in_loop.insert(pair.scan_id1_);
            scans_in_loop.insert(pair.scan_id2_);
        }
        // construct the Kdtree
        pcl::PointCloud<pcl::PointXYZI>::Ptr keyscan_pos_points(new pcl::PointCloud<pcl::PointXYZI>);
        for (auto& [scan_id, scan] : keyscans_) {
            pcl::PointXYZI pt;
            pt.getVector3fMap() = scan->GetWorldFromBody().Trans().cast<float>();
            pt.intensity = scan_id;
            keyscan_pos_points->push_back(pt);
        }
        pcl::KdTreeFLANN<pcl::PointXYZI> position_kdtree;
        position_kdtree.setInputCloud(keyscan_pos_points);

        // search for the close points
        for (auto& [scan_a_id, scan_a] : keyscans_) {
            if (scans_in_loop.count(scan_a_id)) continue;
            pcl::PointXYZI point_a;
            point_a.getVector3fMap() = scan_a->GetWorldFromBody().Trans().cast<float>();

            // search the nh points
            std::vector<int> knn_ids;
            std::vector<float> knn_dists;
            position_kdtree.radiusSearch(point_a, options_.lc_detect_dist_thr, knn_ids, knn_dists);
            for (const auto& id : knn_ids) {
                pcl::PointXYZI point_b = keyscan_pos_points->at(id);
                scan_t scan_b_id = static_cast<scan_t>(point_b.intensity);
                if (scan_b_id == scan_a_id) continue;
                if (scans_in_loop.count(scan_b_id)) continue;
                ScanFrame::Ptr scan_b = keyscans_[scan_b_id];
                ScanPair pair(scan_a->scan_id, scan_b->scan_id);

                // not in buffer check
                if (loops_buf.find(pair) != loops_buf.end()) continue;

                // check the time difference
                if (std::abs(scan_a->GetTimestamp() - scan_b->GetTimestamp()) < options_.lc_detect_temporal_dist_thr)
                    continue;
                candidates[pair] = PairData();
                scans_in_loop.insert(pair.scan_id1_);
                scans_in_loop.insert(pair.scan_id2_);
                break;
            }
        }
        std::cout << "Loop closure candidates count: " << candidates.size() << std::endl;
        std::unordered_map<scan_t, PointCloud::Ptr> submaps;
        for (auto& [pair_id, pair_data] : candidates) {
            const ScanPair& pair = pair_id;
            ScanFrame::Ptr scan_a = keyscans_[pair.scan_id1_];
            ScanFrame::Ptr scan_b = keyscans_[pair.scan_id2_];
            if (submaps.count(pair.scan_id1_) == 0) {
                PointCloud::Ptr submap_a = GetSubMap(pair.scan_id1_);
                submaps[pair.scan_id1_] = submap_a;
            }
            if (submaps.count(pair.scan_id2_) == 0) {
                PointCloud::Ptr submap_b = GetSubMap(pair.scan_id2_);
                submaps[pair.scan_id2_] = submap_b;
            }
        }
        auto register_task = [&](const ScanPair& reg_pair) {
            ScanFrame::Ptr scan_a = keyscans_[reg_pair.scan_id1_];
            ScanFrame::Ptr scan_b = keyscans_[reg_pair.scan_id2_];

            CHECK(submaps.count(reg_pair.scan_id1_));
            CHECK(submaps.count(reg_pair.scan_id2_));
            PointCloud::Ptr points_a = submaps[reg_pair.scan_id1_];
            PointCloud::Ptr points_b = submaps[reg_pair.scan_id2_];

            pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
            gicp.setMaxCorrespondenceDistance(2.0);
            gicp.setInputSource(points_a);
            gicp.setInputTarget(points_b);

            Pose3 init_ab = scan_a->GetWorldFromBody().GetInverse() * scan_b->GetWorldFromBody();

            PointCloud::Ptr aligned_source(new PointCloud);
            gicp.align(*aligned_source, init_ab.Mat4d().cast<float>());
            Eigen::Matrix4d reg_result_mat = gicp.getFinalTransformation().cast<double>();
            Pose3 reg_result(reg_result_mat);

            // calc the corres
            int corres_count = 0;
            double error_sum = 0.0;
            auto kdtree = gicp.getSearchMethodTarget();
            double corres_cal_thr = 0.5;
            std::vector<int> nn_indices(1);
            std::vector<float> nn_dists(1);
            for (auto& point : aligned_source->points) {
                kdtree->nearestKSearch(point, 1, nn_indices, nn_dists);
                if (nn_dists[0] <= corres_cal_thr) {
                    error_sum += nn_dists[0];
                    corres_count++;
                }
            }
            PairData pair_data;
            pair_data.target_points_count = points_a->size();
            pair_data.source_points_count = points_b->size();
            if (corres_count != 0) pair_data.average_error = error_sum / corres_count;
            else
                pair_data.average_error = 0.0;
            pair_data.corres_count = corres_count;
            pair_data.ab_rel_pose = reg_result;

            // debug
            if (options_.output_all_loop_reigster_result) {
                std::string pair_name = std::to_string(reg_pair.scan_id1_) + "_" + std::to_string(reg_pair.scan_id2_);
                fs::path save_dir(fs::path(output_dir) / fs::path("loop") / pair_name);
                fs::create_directories(save_dir);
                // target
                pcl::io::savePCDFile((save_dir / "target.pcd").string(), *points_a);
                // before register
                PointCloud points_b_tra;
                pcl::transformPointCloud(*points_b, points_b_tra, init_ab.Mat4d());
                pcl::io::savePCDFile((save_dir / "source.pcd").string(), points_b_tra);
                // after register
                pcl::io::savePCDFile((save_dir / "align_source.pcd").string(), *aligned_source);
            }
            return pair_data;
        };

        std::cout << "Start loop closure registration with " << candidates.size() << " candidates." << std::endl;
        std::unordered_map<ScanPair, std::future<PairData>> candidate_futures;
        ThreadPool thread_pool(4);
        for (auto& candidate : candidates) {
            candidate_futures[candidate.first] = thread_pool.enqueue(register_task, candidate.first);
        }

        boost::progress_display progress(candidates.size(), std::cout);
        for (auto& candidate : candidates) {
            PairData result = candidate_futures[candidate.first].get();
            candidate.second = result;
            ++progress;
        }

        // remove the bad candidate
        for (auto iter = candidates.begin(); iter != candidates.end();) {
            PairData& pair_data = iter->second;
            double source_overlap = double(pair_data.corres_count) / double(pair_data.source_points_count);
            double target_overlap = double(pair_data.corres_count) / double(pair_data.target_points_count);
            if (source_overlap < 0.2 || target_overlap < 0.2 || pair_data.average_error > 0.1)
                iter = candidates.erase(iter);
            else
                iter++;
        }
        printf("Loop closure detect candidates count: %lu\n", candidates.size());

        // add to buffer
        for (auto& [pair, pair_data] : candidates) loops_buf[pair] = pair_data;

        return candidates;
    }

    void SaveLoopToPcd(const std::string& save_fname) {
        pcl::PointCloud<pcl::PointXYZRGB> key_points;
        for (auto& [scan_id, scan] : keyscans_) {
            pcl::PointXYZRGB point;
            point.x = scan->GetWorldFromBody().Trans().x();
            point.y = scan->GetWorldFromBody().Trans().y();
            point.z = scan->GetWorldFromBody().Trans().z();
            point.r = 255;
            point.g = 255;
            point.b = 255;
            key_points.push_back(point);
        }

        for (auto& [pair, pair_data] : loops_buf) {
            Eigen::Vector3d p_a = keyscans_[pair.scan_id1_]->GetWorldFromBody().Trans();
            Eigen::Vector3d p_b = keyscans_[pair.scan_id2_]->GetWorldFromBody().Trans();
            int point_count = 20;
            double dist_interval = (p_b - p_a).norm() / point_count;
            Eigen::Vector3d dir = (p_b - p_a).normalized();
            for (int i = 0; i < point_count; ++i) {
                Eigen::Vector3d p = p_a + dist_interval * i * dir;
                pcl::PointXYZRGB point;
                point.x = p.x();
                point.y = p.y();
                point.z = p.z();
                point.r = 255;
                point.g = 0;
                point.b = 0;
                key_points.push_back(point);
            }
        }
        pcl::io::savePCDFile(save_fname, key_points);
    }
    void ExportMap(const std::string& save_fname) {
        PointCloud::Ptr whole_map(new PointCloud);
        for (auto& [scan_id, scan] : keyscans_) {
            PointCloud scan_world;
            pcl::transformPointCloud(*scan->GetScan(), scan_world, scan->GetWorldFromBody().Mat4d());
            *whole_map += scan_world;
        }
        pcl::io::savePCDFile(save_fname, *whole_map);
        std::cout << "Export map to " << save_fname << std::endl;
    }
    Trajectory ExportStampedPoses() {
        Trajectory stamped_poses;
        for (auto& [scan_id, scan] : keyscans_) {
            stamped_poses.emplace_back(scan->GetTimestamp(), scan->GetWorldFromBody().Isometry3d());
        }
        return stamped_poses;
    }
    PointCloud::Ptr GetSubMap(const scan_t& scan_id) {
        CHECK(keyscans_.count(scan_id));
        ScanFrame::Ptr scan_i = keyscans_[scan_id];

        PointCloud::Ptr submap(new PointCloud);

        Pose3 w_bi = scan_i->GetWorldFromBody();
        auto center_it = keyscans_.find(scan_id);
        auto begin_it = center_it;
        for (int i = 0; i < options_.sub_map_interval && begin_it != keyscans_.begin(); ++i) {
            --begin_it;
        }
        auto end_it = center_it;
        for (int i = 0; i <= options_.sub_map_interval && end_it != keyscans_.end(); ++i) {
            ++end_it;
        }
        for (auto it = begin_it; it != end_it; ++it) {
            PointCloud scan_j_tra;
            ScanFrame::Ptr& scan_j = it->second;
            Pose3 w_bj = scan_j->GetWorldFromBody();
            Pose3 bi_bj = w_bi.GetInverse() * w_bj;
            pcl::transformPointCloud(*scan_j->GetScan(), scan_j_tra, bi_bj.Mat4d());
            *submap += scan_j_tra;
        }

        pcl::UniformSampling<Point> sampler;
        sampler.setRadiusSearch(0.1);
        sampler.setInputCloud(submap);
        sampler.filter(*submap);
        return submap;
    }
};

}  // namespace faster_lio

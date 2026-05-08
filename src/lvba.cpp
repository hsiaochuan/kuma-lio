#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <boost/program_options.hpp>

#include <float.h>
#include <algorithm>
#include <boost/histogram.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <unordered_set>
#include "ba_cost_functions.h"
#include "common_lib.h"
#include "cost_functions.h"
#include "lidar_simulator.h"
#include "sfm_data.h"
#include "stamp_pose.h"
#include "union_find.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string cam_trajectory_fname;
    std::string colmap_output_dir;
    std::string databse_fname;
    std::string lidar_points_fname;
    po::options_description desc("Jet smoothing options");
    desc.add_options()("help,h", "Show help message")
        // clang-format off
    ("cam_trajectory,c",po::value<std::string>(&cam_trajectory_fname)->required(),"Camera trajectory file (TUM format)")
    ("lidar_points_fname,l", po::value<std::string>(&lidar_points_fname)->required(), "LiDAR points map")
    ("database,d", po::value<std::string>(&databse_fname)->required(), "Colmap database file")
    ("colmap_output,o", po::value<std::string>(&colmap_output_dir),"Colmap output directory (if specified, the program will load the reconstruction from the colmap output "
        "instead of the database)");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // load trajectory
    auto cam_stamped_poses = faster_lio::TrajectoryGenerator::load_from_tumtxt(cam_trajectory_fname);
    faster_lio::TrajectoryInterpolator cam_interpolator(cam_stamped_poses);

    // load from COLMAP
    sfm_data recon;
    recon.LoadFromDatabase(databse_fname);

    // load camera poses
    for (auto& [image_id, image] : recon.images_) {
        Eigen::Isometry3d cam_pose = cam_interpolator.query(image->TryReadTimeFromName());
        image->cam_from_world_ = Pose3(cam_pose).GetInverse();
    }

    // get obs set and obs vector and obs to id map
    // set joint
    UnionFind uf;
    std::unordered_set<size_t> obs_set;
    for (const auto& [image_pair_id, two_view_geometry] : recon.two_view_geometries_) {
        if (two_view_geometry.config == TwoViewGeometry::WATERMARK || two_view_geometry.inlier_matches.size() < 15)
            continue;
        FeatureMatches pair_match = two_view_geometry.inlier_matches;
        auto [image_id1, image_id2] = PairIdToImagePair(image_pair_id);
        for (auto match : pair_match) {
            Observation obs1(image_id1, match.point2D_idx1);
            Observation obs2(image_id2, match.point2D_idx2);
            size_t obs_id1 = ObservationToId(obs1);
            size_t obs_id2 = ObservationToId(obs2);
            uf.Union(obs_id1, obs_id2);
            obs_set.insert(obs_id1);
            obs_set.insert(obs_id2);
        }
    }

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> track_map;
    for (auto obs_id : obs_set) {
        landmark_t track_id = uf.Find(obs_id);
        track_map[track_id].insert(obs_id);
    }

    std::unordered_set<size_t> problematic_track_id;
    for (auto& [track_id, obs_ids] : track_map) {
        std::unordered_set<image_t> image_set;
        for (auto obs_id : obs_ids) {
            Observation obs = IdToObservation(obs_id);
            image_t image_id = obs.image_id;

            if (image_set.count(image_id) > 0) {
                problematic_track_id.insert(track_id);
                break;
            } else
                image_set.insert(image_id);
        }
    }

    std::unordered_set<size_t> remove_track_id;
    for (auto& track_id : problematic_track_id) {
        std::unordered_set<size_t>& obs_ids = track_map[track_id];
        std::unordered_map<image_t, Eigen::Vector2d> image_obs_map;
        for (auto obs_id : obs_ids) {
            Observation obs = IdToObservation(obs_id);
            image_t image_id = obs.image_id;
            if (image_obs_map.count(image_id) > 0) {
                // found duplicate, valid the distance
                Eigen::Vector2d kp1 = image_obs_map[image_id];
                Eigen::Vector2d kp2 = recon.images_[image_id]->points_[obs.point2d_id];
                if ((kp1 - kp2).norm() > 4.0) remove_track_id.insert(track_id);
            } else
                image_obs_map[image_id] = recon.images_[image_id]->points_[obs.point2d_id];
        }
    }

    std::cout << "Remove track id count: " << remove_track_id.size() << std::endl;
    for (auto& [track_id, obs_ids] : track_map) {
        if (remove_track_id.count(track_id) > 0) continue;
        if (obs_ids.size() < 2) continue;
        recon.landmarks_[track_id] = Landmark();
        for (auto obs_id : obs_ids) {
            recon.landmarks_[track_id].track.emplace_back(IdToObservation(obs_id));
        }
    }
    std::cout << "Landmark count: " << recon.landmarks_.size() << std::endl;
    std::cout << "Triangulate points..." << std::endl;
    // RobustTriangulate(recon);
    for (auto it = recon.landmarks_.begin(); it != recon.landmarks_.end();) {
        if (it->second.xyz.hasNaN()) {
            it = recon.landmarks_.erase(it);
        } else {
            ++it;
        }
    }
    std::vector<int> track_lens;
    for (const auto& [id, lm] : recon.landmarks_) {
        track_lens.push_back(lm.track.size());
    }
    if (!track_lens.empty()) {
        namespace bh = boost::histogram;
        const auto min_len_it = std::min_element(track_lens.begin(), track_lens.end());
        const auto max_len_it = std::max_element(track_lens.begin(), track_lens.end());
        const int min_len = *min_len_it;
        const int max_len = *max_len_it;
        auto hist = bh::make_histogram(bh::axis::integer<>(min_len, max_len + 1));
        for (int len : track_lens) {
            hist(len);
        }
        std::cout << "Track length histogram (len: count)" << std::endl;
        for (auto&& bin : bh::indexed(hist)) {
            std::cout << bin.bin() << ": " << *bin << std::endl;
        }
    }
    std::cout << "Landmark count: " << recon.landmarks_.size() << std::endl;
    std::cout << "Average track length: " << recon.MeanTrackLength() << std::endl;

    // load lidar points
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(lidar_points_fname, *points_xyz);

    if (points_xyz->empty()) {
        std::cerr << "Failed to load LiDAR points from " << lidar_points_fname << std::endl;
        return -1;
    } else
        std::cout << "Loaded " << points_xyz->size() << " LiDAR points from " << lidar_points_fname << std::endl;

    // estimate normal
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(points_xyz);
    ne.setKSearch(20);  // you can tune this parameter
    ne.compute(*normals);
    // kdtree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(points_xyz);

    // filter the fail 3d point
    for (auto iter = recon.landmarks_.begin(); iter != recon.landmarks_.end();) {
        if (iter->second.xyz.hasNaN()) {
            iter = recon.landmarks_.erase(iter);
        } else {
            ++iter;
        }
    }

    std::cout << "----------------------" << "Fusion" << "----------------------" << std::endl;
    std::cout << "Total track count: " << recon.landmarks_.size() << std::endl;
    std::cout << "Average track length: " << recon.MeanTrackLength() << std::endl;
    std::cout << "Mean reprojected error: " << recon.CalcMeanError() << std::endl;

    recon.FilterOutlier(8.0);
    std::cout << "----------------------" << "FIlter" << "----------------------" << std::endl;
    std::cout << "Total track count: " << recon.landmarks_.size() << std::endl;
    std::cout << "Average track length: " << recon.MeanTrackLength() << std::endl;
    std::cout << "Mean reprojected error: " << recon.CalcMeanError() << std::endl;
    // bundle adjustment
    std::cout << "Start bundle adjustment" << std::endl;
    ceres::Problem problem;
    std::unordered_map<camera_t, std::vector<double>> camera_params;
    for (auto& [camera_id, camera] : recon.cameras_) {
        camera_params[camera_id] = camera->get_params();
        problem.AddParameterBlock(camera_params[camera_id].data(), camera_params[camera_id].size(), nullptr);
        problem.SetParameterBlockConstant(camera_params[camera_id].data());
    }
    for (auto& [image_id, image] : recon.images_) {
        problem.AddParameterBlock(image->cam_from_world_.QuatData(), 4, new ceres::EigenQuaternionParameterization);
    }

    double max_search_radius = 0.1;
    int max_ba_iter = 1;
    for (int it = 0; it < max_ba_iter; ++it) {
        int search_count = 0;
        for (auto& [lm_id, landmark] : recon.landmarks_) {
            // search
            std::vector<int> nn_idx;
            std::vector<float> nn_dist;
            pcl::PointXYZ search_point;
            search_point.getVector3fMap() = landmark.xyz.cast<float>();
            tree->radiusSearch(search_point, max_search_radius, nn_idx, nn_dist);
            if (nn_idx.size() > 0) {
                pcl::PointXYZ nearest_point = points_xyz->points[nn_idx[0]];
                pcl::Normal nearest_normal = normals->points[nn_idx[0]];

                // ppl
                ceres::CostFunction* ppl_cost =
                    PointOnPlaneCostFunctor::Create(nearest_normal.getNormalVector3fMap().cast<double>(),
                                                    nearest_point.getVector3fMap().cast<double>(), 1.0);
                problem.AddResidualBlock(ppl_cost, nullptr, landmark.xyz.data());
                search_count++;
            }

            for (int j = 0; j < landmark.track.size(); ++j) {
                Observation& track_elem = landmark.track[j];
                Image::Ptr img = recon.images_[track_elem.image_id];
                Eigen::Vector2d& point2d = img->points_[track_elem.point2d_id];
                std::shared_ptr<CamModel> camera = recon.cameras_[img->CameraId()];
                ceres::CostFunction* cost_function;
                if (camera->get_type() == CAMERA_MODEL::PINHOLE) {
                    cost_function = PinholeIntrinsicCostFunctor::Create(point2d, 1.0);
                } else if (camera->get_type() == CAMERA_MODEL::PINHOLE_RADIAL) {
                    cost_function = PinholeRadialIntrinsicCostFunctor::Create(point2d, 1.0);
                } else {
                    std::cerr << "Unsupported camera model" << std::endl;
                    return -1;
                }

                // reproj
                problem.AddResidualBlock(cost_function, nullptr, camera_params[img->CameraId()].data(),
                                         img->cam_from_world_.QuatData(), img->cam_from_world_.PosData(),
                                         landmark.xyz.data());

            }  // for track element
        }  // for point

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        int filter_count = recon.FilterOutlier(4.0);
        std::cout << "Iteration " << it << " search count: " << search_count << std::endl;
        std::cout << "Iteration " << it << " filter outlier count: " << filter_count << std::endl;
        std::cout << "Iteration " << it << " reprojected error: " << recon.CalcMeanError() << std::endl;
    }  // for ba iter

    recon.WriteCOLMAPText(colmap_output_dir);
    return 0;
}
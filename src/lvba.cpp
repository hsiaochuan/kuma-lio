#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <boost/program_options.hpp>

#include <float.h>
#include <boost/pending/disjoint_sets.hpp>
#include <unordered_set>
#include "ba_cost_functions.h"
#include "common_lib.h"
#include "cost_functions.h"
#include "lidar_simulator.h"
#include "reconstruction.h"
#include "stamp_pose.h"
#include "union_find.h"
namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string cam_trajectory_fname;
    std::string colmap_output_dir;
    std::string config_fname;
    std::string databse_fname;
    std::string lidar_points_fname;
    po::options_description desc("Jet smoothing options");
    desc.add_options()("help,h", "Show help message")
        // clang-format off
    ("cam_trajectory,c",po::value<std::string>(&cam_trajectory_fname)->required(),"Camera trajectory file (TUM format)")
    ("lidar_points_fname,l", po::value<std::string>(&lidar_points_fname)->required(), "LiDAR points map")
    ("config,f", po::value<std::string>(&config_fname), "Configuration file (YAML format)")
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
    Reconstruction recon;
    recon.LoadFromDatabase(databse_fname);

    // undistort the points
    for (auto& [image_id, image] : recon.images_) {
        CamModel::Ptr camera = recon.camera(image->CameraId());
        for (int i = 0; i < image->points2D_.size(); ++i) {
            Eigen::Vector2d& point2d = image->points2D_[i].xy;
            image->points2D_[i].xy_undistort = camera->get_ud_pixel(point2d);
        }
    }
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
        track_map[uf.Find(obs_id)].insert(obs_id);
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

    for (auto& [track_id, obs_ids] : track_map) {
        if (problematic_track_id.count(track_id) > 0) continue;
        if (obs_ids.size() < 2) continue;
        recon.points3D_[track_id] = Point3D();
        for (auto obs_id : obs_ids) {
            recon.points3D_[track_id].track.emplace_back(IdToObservation(obs_id));
        }
    }

    std::cout << "Total track count: " << recon.points3D_.size() << std::endl;
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

    // get the depth prior of 2d points
    for (auto& [image_id, image] : recon.images_) {
        CamModel::Ptr camera = recon.cameras_[image->CameraId()];
        Eigen::MatrixXd z_img;
        z_img.resize(camera->h(), camera->w());
        z_img.fill(std::numeric_limits<double>::infinity());
        for (int i = 0; i < points_xyz->size(); ++i) {
            Eigen::Vector3d point_cam = image->cam_from_world_ * points_xyz->at(i).getVector3fMap().cast<double>();
            if (!camera->positive_z(point_cam)) continue;
            Eigen::Vector2d point_undistort = camera->cam2ima(point_cam.hnormalized());
            Eigen::Vector2i uv = point_undistort.cast<int>();
            std::vector<Eigen::Vector2i> uv_list;
            uv_list.emplace_back(uv.x() + 1, uv.y());
            uv_list.emplace_back(uv.x(), uv.y() + 1);
            uv_list.emplace_back(uv.x() - 1, uv.y());
            uv_list.emplace_back(uv.x(), uv.y() - 1);
            for (auto& uv_i : uv_list) {
                if (!camera->valid(uv_i)) continue;
                if (z_img(uv_i.y(), uv_i.x()) > point_cam.z()) {
                    z_img(uv_i.y(), uv_i.x()) = point_cam.z();
                }
            }
        }  // for points

        for (int i = 0; i < image->points2D_.size(); ++i) {
            Eigen::Vector2i uv = image->points2D_[i].xy.cast<int>();
            if (std::isfinite(z_img(uv.y(), uv.x()))) {
                image->points2D_[i].z_prior = z_img(uv.y(), uv.x());
            }
        }
    }  // for img

    // fusion to get the points of 3d
    for (auto& [point_id, point3d] : recon.points3D_) {
        std::vector<Observation>& track_elems = point3d.track;

        // unproject points
        std::vector<Eigen::Vector3d> unprj_points;
        for (auto& track_elem : track_elems) {
            Image::Ptr image = recon.image(track_elem.image_id);
            CamModel::Ptr camera = recon.camera(image->CameraId());
            Point2D& point2d = image->points2D_[track_elem.point2D_idx];

            if (std::isfinite(point2d.z_prior)) {
                Eigen::Vector3d point_cam = camera->ima2cam(point2d.xy_undistort).homogeneous();
                point_cam *= point2d.z_prior;
                Eigen::Vector3d point_world = image->cam_from_world_.GetInverse() * point_cam;
                unprj_points.push_back(point_world);
            }
        }

        // remove the outlier
        std::vector<double> mean_distances;
        mean_distances.resize(unprj_points.size());
        for (int i = 0; i < unprj_points.size(); ++i) {
            double mean_distance = 0;
            for (int j = 0; j < unprj_points.size(); ++j) {
                if (i == j) continue;
                double distance = (unprj_points[i] - unprj_points[j]).norm();
                mean_distance += distance;
            }
            mean_distances[i] = mean_distance / (unprj_points.size() - 1);
        }
        double mean_of_mean_dists =
            std::accumulate(mean_distances.begin(), mean_distances.end(), 0.0) / mean_distances.size();
        double variance = 0.0;
        for (double mean_distance : mean_distances) {
            variance += (mean_distance - mean_of_mean_dists) * (mean_distance - mean_of_mean_dists);
        }
        variance /= mean_distances.size();
        double stddev = std::sqrt(variance);
        double thr = mean_of_mean_dists + stddev;

        for (int i = 0; i < unprj_points.size(); ++i) {
            if (mean_distances[i] > thr) {
                unprj_points.erase(unprj_points.begin() + i);
                i--;
            }
        }

        Eigen::Vector3d mean_point = Eigen::Vector3d::Zero();
        for (auto& point : unprj_points) {
            mean_point += point;
        }
        mean_point /= unprj_points.size();
        if (unprj_points.size() > 0) {
            point3d.xyz = mean_point;
        }
    }

    // filter the fail 3d point
    for (auto iter = recon.points3D_.begin(); iter != recon.points3D_.end();) {
        if (iter->second.xyz.hasNaN()) {
            iter = recon.points3D_.erase(iter);
        } else {
            ++iter;
        }
    }

    std::cout << "----------------------" << "Fusion" << "----------------------" << std::endl;
    std::cout << "Total track count: " << recon.points3D_.size() << std::endl;
    std::cout << "Average track length: " << recon.MeanTrackLength() << std::endl;
    std::cout << "Mean reprojected error: " << recon.CalcMeanError() << std::endl;

    recon.FilterOutlier(8.0);
    std::cout << "----------------------" << "FIlter" << "----------------------" << std::endl;
    std::cout << "Total track count: " << recon.points3D_.size() << std::endl;
    std::cout << "Average track length: " << recon.MeanTrackLength() << std::endl;
    std::cout << "Mean reprojected error: " << recon.CalcMeanError() << std::endl;
    // bundle adjustment
    std::cout << "Start bundle adjustment" << std::endl;
    ceres::Problem problem;
    std::unordered_map<camera_t, std::vector<double>> camera_params;
    for (auto& [camera_id, camera] : recon.cameras_) {
        camera_params[camera_id] = camera->getParams();
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
        for (auto& [point_id, point3d] : recon.points3D_) {
            // search
            std::vector<int> nn_idx;
            std::vector<float> nn_dist;
            pcl::PointXYZ search_point;
            search_point.getVector3fMap() = point3d.xyz.cast<float>();
            tree->radiusSearch(search_point, max_search_radius, nn_idx, nn_dist);
            if (nn_idx.size() > 0) {
                pcl::PointXYZ nearest_point = points_xyz->points[nn_idx[0]];
                pcl::Normal nearest_normal = normals->points[nn_idx[0]];

                // ppl
                ceres::CostFunction* ppl_cost =
                    PointOnPlaneCostFunctor::Create(nearest_normal.getNormalVector3fMap().cast<double>(),
                                                    nearest_point.getVector3fMap().cast<double>(), 1.0);
                problem.AddResidualBlock(ppl_cost, nullptr, point3d.xyz.data());
                search_count++;
            }

            for (int j = 0; j < point3d.track.size(); ++j) {
                Observation& track_elem = point3d.track[j];
                Image::Ptr img = recon.images_[track_elem.image_id];
                Eigen::Vector2d& point2d = img->points2D_[track_elem.point2D_idx].xy;
                std::shared_ptr<CamModel> camera = recon.cameras_[img->CameraId()];
                ceres::CostFunction* cost_function;
                if (camera->getType() == CAMERA_MODEL::PINHOLE) {
                    cost_function = PinholeIntrinsicCostFunctor::Create(point2d, 1.0);
                } else if (camera->getType() == CAMERA_MODEL::PINHOLE_RADIAL) {
                    cost_function = PinholeRadialIntrinsicCostFunctor::Create(point2d, 1.0);
                } else {
                    std::cerr << "Unsupported camera model" << std::endl;
                    return -1;
                }

                // reproj
                problem.AddResidualBlock(cost_function, nullptr, camera_params[img->CameraId()].data(),
                                         img->cam_from_world_.QuatData(), img->cam_from_world_.PosData(),
                                         point3d.xyz.data());

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

    return 0;
}
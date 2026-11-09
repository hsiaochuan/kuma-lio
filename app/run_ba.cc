// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <glog/logging.h>
#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>

#include "ba/ba_data.h"
#include "ba/balm3.h"
#include "cameras/camera_calib.h"
#include "pgo/plane_cloud.h"
#include "pose_io.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace kuma_lio;

namespace {

struct BaParam {
    double downsample = 0.1;      // [m] voxel size used to thin every scan (0: off)
    double min_range = 1.0;       // [m] drop the points closer than this (the carrier itself)
    double min_constraint = 0.0;  // scans whose PlaneConstraint is below it are removed (0: off)
    ba::VoxelMapOptions voxel;
    ba::BAOptions opt;
};

void LoadParam(const std::string& file, BaParam* p) {
    const YAML::Node y = YAML::LoadFile(file);
    auto get = [&](const char* key, auto* v) {
        if (y[key]) *v = y[key].as<std::decay_t<decltype(*v)>>();
    };
    get("downsample", &p->downsample);
    get("min_range", &p->min_range);
    get("min_constraint", &p->min_constraint);
    get("voxel_size", &p->voxel.voxel_size);
    get("plane_eigen_ratio", &p->voxel.plane_eigen_ratio);
    get("min_plane_size", &p->voxel.min_plane_size);
    get("min_points", &p->voxel.min_points);
    get("max_outer", &p->opt.max_outer);
    get("max_inner", &p->opt.max_inner);
    get("eps", &p->opt.eps);
    get("verbose", &p->opt.verbose);
}

// The points of a .ply file; false if it cannot be read.
bool LoadPly(const fs::path& file, std::vector<Vec3>* points) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPLYFile(file.string(), cloud) < 0) return false;
    points->clear();
    points->reserve(cloud.size());
    for (const auto& q : cloud.points) points->emplace_back(q.x, q.y, q.z);
    return true;
}

// ba_data.extrin (T_body_cam: the poses are IMU poses) and ba_data.cameras from a camera calibration yaml, see
// LoadCameraCalib in cameras/camera_calib.h for its two extrinsic forms.
void LoadCalib(const std::string& file, ba::BaData* ba_data) {
    const CameraCalib calib = LoadCameraCalib(file);
    ba_data->extrin = calib.T_imu_cam;
    for (const auto& camera_it : calib.cameras) ba_data->cameras[camera_it.first] = camera_it.second;
    if (calib.from_lidar) LOG(INFO) << "extrin from T_cam_lidar and T_imu_lidar";
}

// Horizontal and vertical field of view [deg] of a camera: the angles between the rays through the middles of
// opposite image borders, so it holds for any camera model.  The SavePairList defaults if a ray is missing.
std::pair<double, double> CameraFov(const CamModel& cam) {
    const double w = cam.w(), h = cam.h();
    Vec3 left, right, top, bottom;
    if (!cam.CamRayFromImg(Vec2(0, h / 2), &left) || !cam.CamRayFromImg(Vec2(w, h / 2), &right) ||
        !cam.CamRayFromImg(Vec2(w / 2, 0), &top) || !cam.CamRayFromImg(Vec2(w / 2, h), &bottom))
        return {73.8, 58.8};
    const double kDeg = 180.0 / M_PI;
    return {std::acos(std::clamp(left.dot(right), -1.0, 1.0)) * kDeg,
            std::acos(std::clamp(top.dot(bottom), -1.0, 1.0)) * kDeg};
}

double MedianRange(const ba::BaData& ba_data) {
    std::vector<double> ranges;
    const size_t step = std::max<size_t>(1, ba_data.scans.size() / 200);
    size_t k = 0;
    for (const auto& scan_it : ba_data.scans) {
        if (k++ % step != 0) continue;
        const std::vector<Vec3>& pts = scan_it.second->points;
        for (size_t i = 0; i < pts.size(); i += 10) ranges.push_back(pts[i].norm());
    }
    if (ranges.empty()) return 0.0;
    const auto mid = ranges.begin() + ranges.size() / 2;
    std::nth_element(ranges.begin(), mid, ranges.end());
    return *mid;
}

// A file whose extension (any case) is one of the common image formats.
bool IsImageFile(const fs::path& file) {
    std::string ext = file.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    static const std::set<std::string> kImageExts = {".png",  ".jpg",  ".jpeg", ".bmp", ".tif",
                                                     ".tiff", ".webp", ".pgm",  ".ppm"};
    return kImageExts.count(ext) > 0;
}

// The points of a scan that go into the BA: the far enough ones, thinned.
std::vector<Vec3> PreprocessScan(const ba::Scan& scan, const BaParam& prm) {
    std::vector<Vec3> kept;
    kept.reserve(scan.points.size());
    for (const Vec3& p : scan.points)
        if (p.allFinite() && p.norm() >= prm.min_range) kept.push_back(p);
    return prm.downsample > 0 ? pgo::UniformSample(kept, prm.downsample) : kept;
}

// How firmly the planes hold each scan in place, by pose id: the smallest eigenvalue of sum N_ij u u^T over
// the planes the scan shares with another scan, N_ij its points on plane i and u the plane normal at the input
// poses.  It counts the plane points that push back along the scan's weakest direction, so it is 0 when every
// plane it sees is parallel (a drone sitting on the apron sees the ground only) or it shares no plane, and such
// a scan slides freely in the BA.  A scan in no shared plane is absent, i.e. 0.
std::map<ba::IndexT, double> PlaneConstraint(const std::vector<ba::PlaneFeature>& planes, const ba::Poses& poses) {
    std::map<ba::IndexT, Mat3> info;
    PointCluster world, tr;
    for (const ba::PlaneFeature& plane : planes) {
        if (plane.clus.size() < 2) continue;  // the BA drops it too
        world.SetZero();
        for (const ba::FrameCluster& o : plane.clus) {
            const Eigen::Isometry3d& pose = poses.at(o.xid);
            tr.Transform(o.clu, pose.linear(), pose.translation());
            world += tr;
        }
        const Vec3 u = Eigen::SelfAdjointEigenSolver<Mat3>(world.Cov()).eigenvectors().col(0);
        for (const ba::FrameCluster& o : plane.clus) {
            auto it = info.try_emplace(o.xid, Mat3::Zero()).first;
            it->second += o.clu.N * u * u.transpose();
        }
    }
    std::map<ba::IndexT, double> constraint;
    for (const auto& it : info)
        constraint[it.first] = Eigen::SelfAdjointEigenSolver<Mat3>(it.second, Eigen::EigenvaluesOnly).eigenvalues()[0];
    return constraint;
}

// Remove the frames that the planes do not hold, PlaneConstraint below min_constraint: the scan, its plane
// observations, its pose and the images at that pose all leave sfm_data.  Left in, the BA moves them along their
// free directions, tens of meters for a drone taking off.
// <out>/scan_constraint.txt gets "<name> <PlaneConstraint>" for every scan, to choose the threshold.
void RemoveWeakScans(ba::BaData& sfm_data, std::vector<ba::PlaneFeature>* planes, double min_constraint,
                     const fs::path& out) {
    const std::map<ba::IndexT, double> constraint = PlaneConstraint(*planes, sfm_data.poses);
    std::ofstream dump(out / "scan_constraint.txt");
    std::set<ba::IndexT> weak;
    std::ostringstream names;
    for (auto it = sfm_data.scans.begin(); it != sfm_data.scans.end();) {
        const auto c_it = constraint.find(it->second->id_pose);
        const double c = c_it == constraint.end() ? 0.0 : c_it->second;
        const std::string name = fs::path(it->second->path).stem().string();
        dump << name << ' ' << c << '\n';
        if (min_constraint > 0.0 && c < min_constraint) {
            weak.insert(it->second->id_pose);
            if (weak.size() <= 20) names << ' ' << name << '(' << std::lround(c) << ')';
            it = sfm_data.scans.erase(it);
        } else {
            ++it;
        }
    }
    if (weak.empty()) return;
    for (ba::PlaneFeature& plane : *planes)
        plane.clus.erase(std::remove_if(plane.clus.begin(), plane.clus.end(),
                                        [&](const ba::FrameCluster& o) { return weak.count(o.xid) > 0; }),
                         plane.clus.end());
    for (const ba::IndexT id : weak) sfm_data.poses.erase(id);
    size_t removed_images = 0;
    for (auto it = sfm_data.rgb_images.begin(); it != sfm_data.rgb_images.end();) {
        if (weak.count(it->second->id_pose)) {
            it = sfm_data.rgb_images.erase(it);
            ++removed_images;
        } else {
            ++it;
        }
    }
    LOG(INFO) << "removed " << weak.size() << " frames (" << removed_images << " images) with a plane constraint below "
              << min_constraint << ":" << names.str() << (weak.size() > 20 ? " ..." : "");
}

// Optimize the poses associated to a laser frame (Scan::id_pose) with the point-to-plane bundle adjustment of
// BALM3.  Only these poses move; false, with the poses untouched, if no plane ties two scans together.
bool BAOptimize(ba::BaData& sfm_data, const BaParam& prm, const fs::path& out) {
    // Plane association.  The map is only needed to find the planes, and is the largest thing in memory, so it
    // is gone before the optimization.
    ba::BundleAdjuster adjuster;
    adjuster.opt = prm.opt;
    {
        std::vector<const ba::Scan*> scans;
        for (const auto& scan_it : sfm_data.scans) scans.push_back(scan_it.second.get());
        std::vector<std::vector<Vec3>> pts(scans.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long i = 0; i < (long)scans.size(); ++i) pts[i] = PreprocessScan(*scans[i], prm);

        ba::PlaneMap plane_map(prm.voxel);
        size_t total_pts = 0;
        for (size_t i = 0; i < scans.size(); ++i) {
            const Eigen::Isometry3d& pose = sfm_data.poses.at(scans[i]->id_pose);
            plane_map.AddScan(scans[i]->id_pose, pose.linear(), pose.translation(), pts[i]);
            total_pts += pts[i].size();
        }
        LOG(INFO) << "inserted " << total_pts << " points from " << scans.size() << " laser frames into "
                  << plane_map.NumVoxels() << " voxels";
        std::vector<ba::PlaneFeature> planes = plane_map.ExtractPlanes(sfm_data.poses);
        RemoveWeakScans(sfm_data, &planes, prm.min_constraint, out);
        adjuster.Load(std::move(planes));
    }
    LOG(INFO) << "plane features: " << adjuster.NumFeatures() << ", observations: " << adjuster.NumObservations()
              << ", poses in the BA: " << adjuster.Nx << "/" << sfm_data.poses.size();
    if (adjuster.NumFeatures() == 0) {
        LOG(WARNING) << "no plane feature found (try a larger voxel_size or a looser plane_eigen_ratio)";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    adjuster.Optimize(sfm_data.poses);
    LOG(INFO) << "BALM3 took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    std::string scans_dir, images_dir, pose_fname, config_fname, calib_fname, out_dir;
    po::options_description desc("run_ba --scans_dir <dir> --pose_fname <file> [options]");
    // clang-format off
    desc.add_options()
        ("help,h", "show this help")
        ("scans_dir", po::value<std::string>(&scans_dir),"folder with the <name>.ply scans")
        ("images_dir", po::value<std::string>(&images_dir), "folder with the <name>.<png|jpg|...> images (optional)")
        ("pose_fname", po::value<std::string>(&pose_fname), "pose file, one \"<name> x y z qx qy qz qw\" per line")
        ("calib_fname", po::value<std::string>(&calib_fname), "camera yaml like config/i2nav_robot_camera.yaml: extrin_R, extrin_t (T_body_cam) or extrin_Rcl, extrin_tcl, extrin_Ril, extrin_til, and camera (cameras.txt lines); identity and no camera if empty")
        ("config,c", po::value<std::string>(&config_fname),"yaml with the BA parameters (see config/ba.yaml); defaults if absent")
        ("out,o", po::value<std::string>(&out_dir), "output folder; default <scans_dir>/ba");
    // clang-format on
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << e.what() << "\n" << desc << std::endl;
        return 1;
    }
    if (vm.count("help") || !vm.count("scans_dir") || !vm.count("pose_fname")) {
        std::cerr << desc << std::endl;
        return vm.count("help") ? 0 : 1;
    }
    const fs::path dir = scans_dir;
    if (!fs::is_directory(dir)) LOG(FATAL) << "not a folder: " << dir;
    if (!images_dir.empty() && !fs::is_directory(images_dir)) LOG(FATAL) << "not a folder: " << images_dir;
    const fs::path out = out_dir.empty() ? dir / "ba" : fs::path(out_dir);
    fs::create_directories(out);

    // Parameters of the lidar BA
    BaParam prm;
    if (!config_fname.empty()) LoadParam(config_fname, &prm);

    // Load the poses
    const std::vector<PoseLine> named_poses = LoadPoses(pose_fname);
    if (named_poses.empty()) LOG(FATAL) << "no poses in " << pose_fname;
    LOG(INFO) << "loaded " << named_poses.size() << " poses from " << pose_fname;

    // Poses are indexed by their line rank in the pose file.
    std::map<std::string, ba::IndexT> name_id_map;
    for (size_t i = 0; i < named_poses.size(); ++i)
        if (!name_id_map.emplace(named_poses[i].name, static_cast<ba::IndexT>(i)).second)
            LOG(FATAL) << "duplicated pose name in the pose file: " << named_poses[i].name;
    ba::BaData ba_data;
    if (!calib_fname.empty()) LoadCalib(calib_fname, &ba_data);
    for (const auto& camera_it : ba_data.cameras)
        LOG(INFO) << "camera " << camera_it.first << ": " << camera_it.second->model_name() << " "
                  << camera_it.second->w() << "x" << camera_it.second->h();
    // The id of the pose with this name (kUndefinedIndex if none), registered in sfm_data.poses.
    auto FindPoseId = [&](const std::string& name) -> ba::IndexT {
        const auto it = name_id_map.find(name);
        if (it == name_id_map.end()) return ba::kUndefinedIndex;
        ba_data.poses[it->second] = named_poses[it->second].pose;
        return it->second;
    };

    // Load the laser frames and associate them to the poses
    std::vector<fs::path> ply_files;
    for (const auto& entry : fs::directory_iterator(dir))
        if (entry.is_regular_file() && entry.path().extension() == ".ply") ply_files.push_back(entry.path());
    std::sort(ply_files.begin(), ply_files.end());
    std::vector<std::shared_ptr<ba::Scan>> loaded;
    size_t scan_without_pose = 0;
    for (const fs::path& points_fname : ply_files) {
        // A laser frame without pose is dropped
        const ba::IndexT id_pose = FindPoseId(points_fname.stem().string());
        if (id_pose == ba::kUndefinedIndex) {
            LOG(WARNING) << "no pose for point cloud, it is removed: " << points_fname;
            ++scan_without_pose;
            continue;
        }
        auto scan = std::make_shared<ba::Scan>();
        scan->path = points_fname.string();
        scan->id_pose = id_pose;
        loaded.push_back(std::move(scan));
    }
    std::vector<char> readable(loaded.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (long i = 0; i < (long)loaded.size(); ++i) readable[i] = LoadPly(loaded[i]->path, &loaded[i]->points);
    ba::IndexT scan_id = 0;
    for (size_t i = 0; i < loaded.size(); ++i) {
        if (!readable[i]) {
            LOG(WARNING) << "cannot read point cloud: " << loaded[i]->path;
            continue;
        }
        ba_data.scans[scan_id++] = std::move(loaded[i]);
    }
    LOG(INFO) << "laser frames loaded: " << ba_data.scans.size() << " (removed without pose: " << scan_without_pose
              << ")";

    // Associate the images to the poses; an image without pose is dropped.  Only the file is recorded.
    if (!images_dir.empty()) {
        std::vector<fs::path> image_files;
        for (const auto& entry : fs::directory_iterator(images_dir))
            if (entry.is_regular_file() && IsImageFile(entry.path())) image_files.push_back(entry.path());
        std::sort(image_files.begin(), image_files.end());
        std::set<ba::IndexT> pose_with_image;
        ba::IndexT image_id = 0;
        for (const fs::path& image_fname : image_files) {
            const ba::IndexT id_pose = FindPoseId(image_fname.stem().string());
            if (id_pose == ba::kUndefinedIndex) {
                LOG(WARNING) << "no pose for image, it is removed: " << image_fname;
                continue;
            }
            // the same name in two formats (<name>.png and <name>.jpg): the first one in file name order is kept
            if (!pose_with_image.insert(id_pose).second) {
                LOG(WARNING) << "another image already has this name, it is removed: " << image_fname;
                continue;
            }
            auto image = std::make_shared<ba::RgbImage>();
            image->path = image_fname.string();
            image->id_pose = id_pose;
            ba_data.rgb_images[image_id++] = std::move(image);
        }
        LOG(INFO) << "images associated to a pose: " << ba_data.rgb_images.size() << "/" << image_files.size();
    }

    // Optimize the poses associated to a laser frame
    if (!BAOptimize(ba_data, prm, out)) LOG(WARNING) << "poses unchanged";

    const fs::path traj = out / "traj_ba.txt";
    if (!ba_data.SaveTumPoses(traj.string())) LOG(FATAL) << "cannot write " << traj;
    LOG(INFO) << "wrote " << traj << ": " << ba_data.scans.size() << " poses";

    // The images at the refined poses, for colmap point_triangulator: <images_dir>/../sparse
    if (!ba_data.rgb_images.empty()) {
        fs::path images = fs::absolute(images_dir).lexically_normal();
        if (!images.has_filename()) images = images.parent_path();  // "images/" ends with a separator
        const fs::path sparse = images.parent_path() / "sparse";
        if (!ba_data.SaveCOLMAP(sparse.string())) LOG(FATAL) << "cannot write " << sparse;
        double hfov = 73.8, vfov = 58.8;
        if (ba_data.cameras.count(1)) std::tie(hfov, vfov) = CameraFov(*ba_data.cameras.at(1));
        const double scene_depth = MedianRange(ba_data);
        const double radius = scene_depth > 0.0 ? 1.5 * scene_depth : 10.0;
        const fs::path pair_list = images.parent_path() / "pair_list.txt";
        if (!ba_data.SavePairList(pair_list.string(), hfov, vfov, radius)) LOG(FATAL) << "cannot write " << pair_list;
        std::ifstream pairs_in(pair_list);
        const long num_pairs =
            std::count(std::istreambuf_iterator<char>(pairs_in), std::istreambuf_iterator<char>(), '\n');
        LOG(INFO) << "wrote " << pair_list << ": " << num_pairs << " pairs (fov " << hfov << " x " << vfov
                  << " deg, median scene depth " << scene_depth << " m, radius " << radius << " m)";
        LOG(INFO) << "wrote " << sparse << ": cameras.txt (" << ba_data.cameras.size() << " cameras), images.txt ("
                  << ba_data.rgb_images.size() << " images) and an empty points3D.txt";
    }
    return 0;
}

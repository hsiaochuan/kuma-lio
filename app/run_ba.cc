//
// Offline LiDAR bundle adjustment of the scan poses, BALM3.0 (Li et al., 2025): the poses are refined so that
// the planes the scans see together are as thin as possible.  The lidar part of openMVG's main_BALVM, laid out
// the same way: the poses and the laser frames are loaded into a BaData, associated by name, and
// BAOptimize refines the poses that hold a laser frame.
//
//   run_ba --scans_dir <dir> --pose_fname <file> [--config ba.yaml] [--out dir]  (-c/-o, -h for help)
//
// The inputs are those of run_pgo and run_clean: <scans_dir>/<name>.ply are the scans in the sensor's own
// frame and <pose_fname> holds a line "<name> x y z qx qy qz qw" (T_world_body) per scan, e.g. run_pgo's
// traj_pgo.txt.  The output <out>/traj_ba.txt is in TUM format, one line "<name> x y z qx qy qz qw" per laser
// frame, in the order of the file names; a pose without a readable scan is left out.  A scan that no plane ties
// to another scan keeps its pose, so do all of them if no plane is found.  The refined poses are written as
// the optimizer left them.
//

#include <glog/logging.h>
#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

#include "ba/ba_data.h"
#include "ba/balm3.h"
#include "pgo/plane_cloud.h"
#include "pose_io.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace faster_lio;

namespace {

struct BaParam {
    double downsample = 0.1;  // [m] voxel size used to thin every scan (0: off)
    double min_range = 1.0;   // [m] drop the points closer than this (the carrier itself)
    int threads = 0;          // 0: OpenMP default
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
    get("threads", &p->threads);
    get("voxel_size", &p->voxel.voxel_size);
    get("plane_eigen_ratio", &p->voxel.plane_eigen_ratio);
    get("min_plane_size", &p->voxel.min_plane_size);
    get("min_points", &p->voxel.min_points);
    get("max_outer", &p->opt.max_outer);
    get("max_inner", &p->opt.max_inner);
    get("eps", &p->opt.eps);
    get("verbose", &p->opt.verbose);
}

/// The points of a .ply file; false if it cannot be read.
bool LoadPly(const fs::path& file, std::vector<Vec3>* points) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPLYFile(file.string(), cloud) < 0) return false;
    points->clear();
    points->reserve(cloud.size());
    for (const auto& q : cloud.points) points->emplace_back(q.x, q.y, q.z);
    return true;
}

/// A file whose extension (any case) is one of the common image formats.
bool IsImageFile(const fs::path& file) {
    std::string ext = file.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    static const std::set<std::string> kImageExts = {".png", ".jpg", ".jpeg", ".bmp", ".tif",
                                                     ".tiff", ".webp", ".pgm", ".ppm"};
    return kImageExts.count(ext) > 0;
}

/// The points of a scan that go into the BA: the far enough ones, thinned.
std::vector<Vec3> PreprocessScan(const ba::Scan& scan, const BaParam& prm) {
    std::vector<Vec3> kept;
    kept.reserve(scan.points.size());
    for (const Vec3& p : scan.points)
        if (p.allFinite() && p.norm() >= prm.min_range) kept.push_back(p);
    return prm.downsample > 0 ? pgo::UniformSample(kept, prm.downsample) : kept;
}

/// Optimize the poses associated to a laser frame (Scan::id_pose) with the point-to-plane bundle adjustment of
/// BALM3.  Only these poses move; false, with the poses untouched, if no plane ties two scans together.
bool BAOptimize(ba::BaData& sfm_data, const BaParam& prm) {
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
        adjuster.Load(plane_map.ExtractPlanes(sfm_data.poses));
    }
    LOG(INFO) << "plane features: " << adjuster.NumFeatures() << ", observations: " << adjuster.NumObservations()
              << ", poses in the BA: " << adjuster.Nx << "/" << sfm_data.poses.size();
    if (adjuster.NumFeatures() == 0) {
        LOG(WARNING) << "no plane feature found (try a larger voxel_size or a looser plane_eigen_ratio)";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    adjuster.Optimize(sfm_data.poses);
    LOG(INFO) << "BALM3 took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << " s";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    std::string scans_dir, images_dir, pose_fname, config_fname, out_dir;
    po::options_description desc("run_ba --scans_dir <dir> --pose_fname <file> [options]");
    // clang-format off
    desc.add_options()
        ("help,h", "show this help")
        ("scans_dir", po::value<std::string>(&scans_dir),"folder with the <name>.ply scans")
        ("images_dir", po::value<std::string>(&images_dir), "folder with the <name>.<png|jpg|...> images (optional)")
        ("pose_fname", po::value<std::string>(&pose_fname), "pose file, one \"<name> x y z qx qy qz qw\" per line")
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
    if (prm.threads > 0) omp_set_num_threads(prm.threads);  // the scans are read in parallel too

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
    if (!BAOptimize(ba_data, prm)) LOG(WARNING) << "poses unchanged";

    const fs::path traj = out / "traj_ba.txt";
    if (!ba_data.SaveTumPoses(traj.string())) LOG(FATAL) << "cannot write " << traj;
    LOG(INFO) << "wrote " << traj << ": " << ba_data.scans.size() << " poses";
    return 0;
}

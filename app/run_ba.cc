//
// Offline LiDAR bundle adjustment of the scan poses, BALM3.0 (Li et al., 2025): the poses are refined so that
// the planes the scans see together are as thin as possible.
//
//   run_ba --scans_dir <dir> --pose_fname <file> [--config ba.yaml] [--out dir]  (-c/-o, -h for help)
//
// The inputs are those of run_pgo and run_clean: <scans_dir>/<name>.ply are the scans in the sensor's own
// frame and <pose_fname> holds a line "<name> x y z qx qy qz qw" (T_world_body) per scan, e.g. run_pgo's
// traj_pgo.txt.  The output <out>/traj_ba.txt is the same file with the refined poses: the same lines, in the
// same order, with the same names.  A pose without a readable scan, or whose scan no plane ties to another
// scan, comes out unchanged.  The refined poses are written as the optimizer left them.
//

#include <glog/logging.h>
#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

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

/// The scan's points in its own frame, thinned; empty if it cannot be read.
std::vector<Vec3> LoadScan(const fs::path& file, const BaParam& prm) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (!fs::exists(file) || pcl::io::loadPLYFile(file.string(), cloud) < 0) return {};
    std::vector<Vec3> kept;
    kept.reserve(cloud.size());
    for (const auto& q : cloud.points) {
        const Vec3 p(q.x, q.y, q.z);
        if (p.norm() >= prm.min_range) kept.push_back(p);  // false for a NaN
    }
    return prm.downsample > 0 ? pgo::UniformSample(kept, prm.downsample) : kept;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    std::string scans_dir, pose_fname, config_fname, out_arg;
    po::options_description desc("run_ba --scans_dir <dir> --pose_fname <file> [options]");
    // clang-format off
    desc.add_options()
        ("help,h", "show this help")
        ("scans_dir", po::value<std::string>(&scans_dir),"folder with the <name>.ply scans")
        ("pose_fname", po::value<std::string>(&pose_fname), "pose file, one \"<name> x y z qx qy qz qw\" per line")
        ("config,c", po::value<std::string>(&config_fname),"yaml with the BA parameters (see config/ba.yaml); defaults if absent")
        ("out,o", po::value<std::string>(&out_arg), "output folder; default <scans_dir>/ba")
        ("verify", "check the derivatives against finite differences before optimizing");
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
    const fs::path out = out_arg.empty() ? dir / "ba" : fs::path(out_arg);
    fs::create_directories(out);

    BaParam prm;
    if (!config_fname.empty()) LoadParam(config_fname, &prm);
    if (prm.threads > 0) omp_set_num_threads(prm.threads);

    const std::vector<PoseLine> lines = LoadPoses(pose_fname);
    if (lines.empty()) LOG(FATAL) << "no poses in " << pose_fname;
    LOG(INFO) << "poses: " << lines.size() << " from " << pose_fname;

    // Scans go into the map as they are read: a chunk is read in parallel, then added in order.  The map is only
    // needed to find the planes, and is the largest thing in memory, so it is gone before the optimization.
    ba::Poses poses;              // of the scans that could be read, by dense scan index
    std::vector<size_t> line_of;  // dense scan index -> index in `lines`
    ba::BundleAdjuster adjuster;
    adjuster.opt = prm.opt;
    {
        ba::PlaneMap plane_map(prm.voxel);
        size_t total_pts = 0;
        const size_t kChunk = 64;
        for (size_t begin = 0; begin < lines.size(); begin += kChunk) {
            const size_t end = std::min(lines.size(), begin + kChunk);
            std::vector<std::vector<Vec3>> chunk(end - begin);
#pragma omp parallel for schedule(dynamic, 1)
            for (long i = 0; i < (long)(end - begin); ++i)
                chunk[i] = LoadScan(dir / (lines[begin + i].name + ".ply"), prm);
            for (size_t i = begin; i < end; ++i) {
                const std::vector<Vec3>& pts = chunk[i - begin];
                if (pts.empty()) {
                    LOG(WARNING) << "no readable scan " << (dir / (lines[i].name + ".ply")) << ", pose left as it is";
                    continue;
                }
                plane_map.AddScan(static_cast<uint32_t>(poses.size()), lines[i].pose.linear(),
                                  lines[i].pose.translation(), pts);
                poses.push_back(lines[i].pose);
                line_of.push_back(i);
                total_pts += pts.size();
            }
        }
        LOG(INFO) << "inserted " << total_pts << " points from " << poses.size() << " scans into "
                  << plane_map.NumVoxels() << " voxels";
        adjuster.Load(plane_map.ExtractPlanes(poses));
    }
    ba::Poses refined = poses;  // the planes only move the scans they connect
    LOG(INFO) << "plane features: " << adjuster.NumFeatures() << ", observations: " << adjuster.NumObservations()
              << ", poses in the BA: " << adjuster.Nx << "/" << poses.size();
    if (adjuster.NumFeatures() == 0) {
        LOG(WARNING)
            << "no plane feature found (try a larger voxel_size or a looser plane_eigen_ratio): poses unchanged";
    } else {
        if (vm.count("verify"))
            LOG(INFO) << "derivatives: worst relative error " << adjuster.VerifyDerivatives(refined);
        const auto t0 = std::chrono::steady_clock::now();
        adjuster.Optimize(refined);
        LOG(INFO) << "BALM3 took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s";
    }

    // the same file as the input, with the refined poses
    std::vector<Eigen::Isometry3d> result;
    for (const PoseLine& line : lines) result.push_back(line.pose);
    double max_dt = 0.0;
    for (size_t k = 0; k < refined.size(); ++k) {
        max_dt = std::max(max_dt, (refined[k].translation() - poses[k].translation()).norm());
        result[line_of[k]] = refined[k];
    }
    std::ofstream out_file((out / "traj_ba.txt").string());
    out_file << std::setprecision(9);
    for (size_t i = 0; i < lines.size(); ++i) {
        const Eigen::Vector3d t = result[i].translation();
        const Eigen::Quaterniond q(result[i].rotation());
        out_file << lines[i].name << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' ' << q.x() << ' ' << q.y() << ' '
                 << q.z() << ' ' << q.w() << '\n';
    }
    LOG(INFO) << "wrote " << (out / "traj_ba.txt") << ": " << lines.size() << " poses, " << refined.size()
              << " with a scan, largest change " << max_dt << " m";
    return 0;
}

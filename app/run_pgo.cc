//
// Offline pose graph optimization: loop closure over the output of any odometry.
//
//   run_pgo <input_dir> [--config pgo.yaml] [--poses file.txt] [--out dir]
//
// <input_dir> holds the scans as PCD files, each in the sensor's own frame and named by its time in
// seconds (e.g. 1634567890.123456.pcd; a nanosecond stamp is accepted too), and one TUM-format
// pose file (t x y z qx qy qz qw, T_world_body).  Each scan is placed with the pose interpolated at
// its stamp.  The corrected trajectory comes out as <out>/traj_pgo.txt with one pose per input
// pose, next to the loop diagnostics.
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "pgo/pgo_backend.h"
#include "stamp_pose.h"

namespace fs = std::filesystem;
using namespace faster_lio;

DEFINE_string(config, "", "yaml with the pgo parameters (flat keys, see config/pgo.yaml); defaults if empty");
DEFINE_string(poses, "", "TUM pose file; default is the only .txt/.tum file in the input folder");
DEFINE_string(out, "", "output folder; default <input_dir>/pgo");

namespace {

void LoadParam(const std::string& file, pgo::PgoParam* p) {
    const YAML::Node y = YAML::LoadFile(file);
    auto get = [&](const char* key, auto* v) {
        if (y[key]) *v = y[key].as<std::decay_t<decltype(*v)>>();
    };
    get("kf_scans", &p->kf_scans);
    get("kf_min_angle", &p->kf_min_angle);
    get("kf_min_dist", &p->kf_min_dist);
    get("kf_ds_voxel", &p->kf_ds_voxel);
    get("kf_max_range", &p->kf_max_range);
    get("plane_voxel", &p->plane_voxel);
    get("plane_min_points", &p->plane_min_points);
    get("plane_min_eigen", &p->plane_min_eigen);
    get("loop_radius", &p->loop_radius);
    get("loop_max_dz", &p->loop_max_dz);
    get("loop_min_kf_gap", &p->loop_min_kf_gap);
    get("loop_min_travel", &p->loop_min_travel);
    get("loop_max_candidates", &p->loop_max_candidates);
    get("loop_max_yaw_diff", &p->loop_max_yaw_diff);
    get("icp_max_iter", &p->icp_max_iter);
    get("icp_eigval", &p->icp_eigval);
    get("icp_min_match", &p->icp_min_match);
    get("ratio_drift", &p->ratio_drift);
    get("opt_halt_kf", &p->opt_halt_kf);
    get("opt_min_drift", &p->opt_min_drift);
    get("isam_extra_iter", &p->isam_extra_iter);
    get("loop_var", &p->loop_var);
    get("prior_var", &p->prior_var);
    get("odom_var_rot", &p->odom_var_rot);
    get("odom_var_trans", &p->odom_var_trans);
}

struct ScanFile {
    double time;
    fs::path path;
};

/// Seconds from a file name; a stamp too large to be seconds is read as nanoseconds.
bool StampOf(const fs::path& p, double* t) {
    try {
        size_t used = 0;
        const std::string stem = p.stem().string();
        *t = std::stod(stem, &used);
        if (used != stem.size()) return false;
    } catch (const std::exception&) {
        return false;
    }
    if (*t > 1e11) *t *= 1e-9;
    return true;
}

fs::path FindPoseFile(const fs::path& dir) {
    std::vector<fs::path> found;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string ext = e.path().extension().string();
        if (e.is_regular_file() && (ext == ".txt" || ext == ".tum")) found.push_back(e.path());
    }
    if (found.size() != 1) {
        LOG(FATAL) << "expected exactly one .txt/.tum pose file in " << dir << ", found " << found.size()
                   << "; pass --poses";
    }
    return found.front();
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    gflags::SetUsageMessage("run_pgo <input_dir> [--config pgo.yaml] [--poses file.txt] [--out dir]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc != 2) {
        std::cerr << gflags::ProgramUsage() << std::endl;
        return 1;
    }
    const fs::path dir = argv[1];
    if (!fs::is_directory(dir)) LOG(FATAL) << "not a folder: " << dir;
    const fs::path out = FLAGS_out.empty() ? dir / "pgo" : fs::path(FLAGS_out);
    fs::create_directories(out);

    pgo::PgoParam prm;
    if (!FLAGS_config.empty()) LoadParam(FLAGS_config, &prm);

    const fs::path pose_file = FLAGS_poses.empty() ? FindPoseFile(dir) : fs::path(FLAGS_poses);
    Trajectory traj = TrajectoryGenerator::load_from_tumtxt(pose_file.string());
    if (traj.size() < 2) LOG(FATAL) << "need at least two poses in " << pose_file;
    std::sort(traj.begin(), traj.end(), [](const StampedPose& a, const StampedPose& b) { return a.time < b.time; });
    const TrajectoryInterpolator interp(traj);
    LOG(INFO) << "poses: " << traj.size() << " from " << pose_file << ", " << interp.t_start() << " .. "
              << interp.t_end() << " s";

    std::vector<ScanFile> scans;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".pcd") continue;
        ScanFile s;
        if (!StampOf(e.path(), &s.time)) {
            LOG(WARNING) << "skip " << e.path().filename() << ": the name is not a time stamp";
            continue;
        }
        s.path = e.path();
        scans.push_back(s);
    }
    std::sort(scans.begin(), scans.end(), [](const ScanFile& a, const ScanFile& b) { return a.time < b.time; });
    if (scans.empty()) LOG(FATAL) << "no <time>.pcd scans in " << dir;

    pgo::PgoBackend backend(prm);
    int used = 0, outside = 0;
    for (const ScanFile& s : scans) {
        if (s.time < interp.t_start() || s.time > interp.t_end()) {
            ++outside;
            continue;
        }
        pcl::PointCloud<pcl::PointXYZ> cloud;
        if (pcl::io::loadPCDFile(s.path.string(), cloud) < 0) {
            LOG(WARNING) << "cannot read " << s.path;
            continue;
        }
        std::vector<Vec3> pts;
        pts.reserve(cloud.size());
        for (const auto& p : cloud.points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) pts.emplace_back(p.x, p.y, p.z);
        }
        backend.AddScan(s.time, interp.query(s.time), pts);
        ++used;
    }
    if (outside) LOG(WARNING) << outside << " scans lie outside the pose file's time span and were skipped";
    if (used == 0) LOG(FATAL) << "no scan overlaps the pose file";
    backend.Finish();
    LOG(INFO) << "scans used: " << used;

    // The correction is known at the nodes only: between two of them take the interpolated one,
    // beyond the ends the nearest one.
    const pgo::ScanNodes& nodes = backend.nodes();
    pgo::Isometries delta(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) delta[i] = nodes[i].pose * backend.raw_poses()[i].inverse();
    Trajectory corrected;
    corrected.reserve(traj.size());
    size_t k = 0;
    for (const StampedPose& sp : traj) {
        while (k + 1 < nodes.size() && nodes[k + 1].time <= sp.time) ++k;
        Eigen::Isometry3d d;
        if (sp.time <= nodes.front().time) {
            d = delta.front();
        } else if (k + 1 >= nodes.size()) {
            d = delta.back();
        } else {
            const double a = (sp.time - nodes[k].time) / std::max(1e-9, nodes[k + 1].time - nodes[k].time);
            d = interpolate_pose(delta[k], delta[k + 1], a);
        }
        corrected.emplace_back(sp.time, d * sp.pose);
    }
    TrajectoryGenerator::save_to_tumtxt(corrected, (out / "traj_pgo.txt").string());
    backend.SaveDebug(out.string());
    LOG(INFO) << "wrote " << (out / "traj_pgo.txt") << ": " << corrected.size() << " poses, "
              << backend.loops().size() << " loops, " << backend.num_solves() << " solves";
    return 0;
}

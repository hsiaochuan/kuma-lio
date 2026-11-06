//
// Offline pose graph optimization: loop closure over the output of any odometry.
//
//   run_pgo --scans_dir <dir> --pose_fname <file> [--config pgo.yaml] [--out dir]  (-c/-o, -h for help)
//
// <scans_dir> holds the scans as PLY files in the sensor's own frame, <name>.ply; <pose_fname> is the
// pose file with a line "<name> x y z qx qy qz qw" (T_world_body) per scan.  <name> is the same string in
// both and is also the scan's time, in seconds.  The corrected poses come out as <out>/traj_pgo.txt,
// in the same format, next to the loop diagnostics and <out>/pgo_graph.ply (nodes white, loops red).
//

#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

#include "pgo/pgo_backend.h"
#include "pose_io.h"

namespace fs = std::filesystem;
using namespace faster_lio;

namespace po = boost::program_options;

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
    if (y["loop_search"]) {
        const std::string s = y["loop_search"].as<std::string>();
        if (s == "btc") {
            p->loop_search = pgo::LoopSearch::kBtc;
        } else if (s == "radius") {
            p->loop_search = pgo::LoopSearch::kRadius;
        } else {
            LOG(FATAL) << "loop_search must be btc or radius, got " << s;
        }
    }
    get("loop_min_kf_gap", &p->loop_min_kf_gap);
    get("loop_min_travel", &p->loop_min_travel);
    get("loop_radius", &p->loop_radius);
    get("loop_max_dz", &p->loop_max_dz);
    get("loop_max_candidates", &p->loop_max_candidates);
    get("loop_max_yaw_diff", &p->loop_max_yaw_diff);
    get("btc_merge_normal", &p->btc_merge_normal);
    get("btc_merge_dist", &p->btc_merge_dist);
    get("btc_ref_planes", &p->btc_ref_planes);
    get("btc_pixel", &p->btc_pixel);
    get("btc_layer_height", &p->btc_layer_height);
    get("btc_height_min", &p->btc_height_min);
    get("btc_height_max", &p->btc_height_max);
    get("btc_min_bits", &p->btc_min_bits);
    get("btc_line_filter", &p->btc_line_filter);
    get("btc_nms_radius", &p->btc_nms_radius);
    get("btc_max_keypoints", &p->btc_max_keypoints);
    get("btc_near_num", &p->btc_near_num);
    get("btc_side_min", &p->btc_side_min);
    get("btc_side_max", &p->btc_side_max);
    get("btc_side_res", &p->btc_side_res);
    get("btc_side_tol", &p->btc_side_tol);
    get("btc_similarity", &p->btc_similarity);
    get("btc_candidates", &p->btc_candidates);
    get("btc_min_votes", &p->btc_min_votes);
    get("btc_ransac_dist", &p->btc_ransac_dist);
    get("btc_min_inliers", &p->btc_min_inliers);
    get("btc_overlap_normal", &p->btc_overlap_normal);
    get("btc_overlap_dist", &p->btc_overlap_dist);
    get("btc_min_overlap", &p->btc_min_overlap);
    get("icp_max_iter", &p->icp_max_iter);
    get("icp_eigval", &p->icp_eigval);
    get("icp_min_match", &p->icp_min_match);
    get("ratio_drift", &p->ratio_drift);
    get("opt_halt_kf", &p->opt_halt_kf);
    get("opt_min_drift", &p->opt_min_drift);
    get("isam_extra_iter", &p->isam_extra_iter);
    get("opt_final_all", &p->opt_final_all);
    get("loop_var", &p->loop_var);
    get("prior_var", &p->prior_var);
    get("odom_var_rot", &p->odom_var_rot);
    get("odom_var_trans", &p->odom_var_trans);
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    std::string scans_dir, pose_fname, config_fname, out_arg;
    po::options_description desc("run_pgo --scans_dir <dir> --pose_fname <file> [options]");
    desc.add_options()("help,h", "show this help")
        ("scans_dir", po::value<std::string>(&scans_dir), "folder with the <name>.ply scans")
        ("pose_fname", po::value<std::string>(&pose_fname), "pose file, one \"<name> x y z qx qy qz qw\" per line")
        ("config,c", po::value<std::string>(&config_fname), "yaml with the pgo parameters (see config/pgo.yaml); defaults if absent")
        ("out,o", po::value<std::string>(&out_arg), "output folder; default <scans_dir>/pgo");
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
    const fs::path out = out_arg.empty() ? dir / "pgo" : fs::path(out_arg);
    fs::create_directories(out);

    pgo::PgoParam pgo_prm;
    if (!config_fname.empty())
        LoadParam(config_fname, &pgo_prm);

    const fs::path pose_file = pose_fname;
    std::vector<PoseLine> pose_lines = LoadPoses(pose_file);
    if (pose_lines.empty()) LOG(FATAL) << "no poses in " << pose_file;
    std::sort(pose_lines.begin(), pose_lines.end(), [](const PoseLine& a, const PoseLine& b) { return a.time < b.time; });
    LOG(INFO) << "poses: " << pose_lines.size() << " from " << pose_file;

    pgo::PgoBackend pgo(pgo_prm);
    std::vector<const PoseLine*> matched_lines;
    for (const PoseLine& pose_line : pose_lines) {
        const fs::path scan_file = dir / (pose_line.name + ".ply");
        pcl::PointCloud<pcl::PointXYZ> cloud;
        if (!fs::exists(scan_file) || pcl::io::loadPLYFile(scan_file.string(), cloud) < 0) {
            LOG(WARNING) << "no readable scan " << scan_file << ", pose skipped";
            continue;
        }
        std::vector<Vec3> pts;
        pts.reserve(cloud.size());
        for (const auto& q : cloud.points) {
            if (std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z)) pts.emplace_back(q.x, q.y, q.z);
        }
        pgo.AddScan(pose_line.time, pose_line.pose, pts);
        matched_lines.push_back(&pose_line);
    }
    if (matched_lines.empty()) LOG(FATAL) << "no pose has a scan";
    LOG(INFO) << matched_lines.size() << " poses match scans";
    pgo.Finish();

    // node i is used[i]; its corrected pose is what the graph ended with
    const pgo::ScanNodes& nodes = pgo.nodes();
    std::ofstream out_fname((out / "traj_pgo.txt").string());
    out_fname << std::setprecision(9);
    for (size_t i = 0; i < nodes.size(); ++i) {
        const Eigen::Vector3d t = nodes[i].pose.translation();
        const Eigen::Quaterniond q(nodes[i].pose.rotation());
        out_fname << matched_lines[i]->name << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' ' << q.x() << ' ' << q.y()
             << ' ' << q.z() << ' ' << q.w() << '\n';
    }
    pgo.SaveDebug(out.string());
    LOG(INFO) << "wrote " << (out / "traj_pgo.txt") << ": " << nodes.size() << " poses, "
              << pgo.loops().size() << " loops, " << pgo.num_solves() << " solves";
    return 0;
}

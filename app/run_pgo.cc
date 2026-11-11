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
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
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
using namespace kuma_lio;

namespace po = boost::program_options;

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
        pgo::LoadPgoParam(config_fname, &pgo_prm);

    const fs::path pose_file = pose_fname;
    std::vector<PoseLine> pose_lines = LoadPoses(pose_file);
    if (pose_lines.empty()) LOG(FATAL) << "no poses in " << pose_file;
    std::sort(pose_lines.begin(), pose_lines.end(), [](const PoseLine& a, const PoseLine& b) { return a.time < b.time; });
    LOG(INFO) << "poses: " << pose_lines.size() << " from " << pose_file;

    pgo::PgoBackend pgo(pgo_prm);
    std::vector<const PoseLine*> matched_lines;
    for (const PoseLine& pose_line : pose_lines) {
        const fs::path scan_file = dir / (pose_line.name + ".ply");
        pcl::PointCloud<pcl::PointXYZ> scan_points;
        if (!fs::exists(scan_file) || pcl::io::loadPLYFile(scan_file.string(), scan_points) < 0) {
            LOG(WARNING) << "no readable scan " << scan_file << ", pose skipped";
            continue;
        }
        std::vector<Vec3> pts;
        pts.reserve(scan_points.size());
        for (const auto& q : scan_points.points) {
            if (std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z))
                pts.emplace_back(q.x, q.y, q.z);
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

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
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigenvalues>
#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "pose_io.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZI>;
using Key = uint64_t;

// The parameters of config/clean.yaml, which documents each of them.
struct Param {
    double voxel = 0.2;     // [m]
    double margin = 0.4;    // [m] the last stretch of a ray is not called empty (surface thickness, pose error)
    double max_range = 80;  // [m] rays are not cast beyond it
    int min_free = 3;       // scans that must have seen the voxel empty
    double ratio = 1.0;     // ... and at least ratio * (scans that hit it)
    double ground_dist = 0.3;  // [m] a point this close to its patch's plane is ground and always kept; 0 = off
    int min_hits = 0;          // a point within near_range of the sensor needs its voxel hit by more scans than this
    double near_range = 8;     // [m]
    double up_z = 1;           // +1 if the world's z axis points up, -1 if it points down
    double sensor_height = 1;  // [m] above the ground
    double min_range = 2.0;    // [m] points nearer the sensor are dropped from the output (the carrier itself)
};

// Keys absent from the yaml keep their defaults; an unknown key is fatal, it is most likely a typo.
void LoadParam(const std::string& file, Param* p) {
    YAML::Node y;
    try {
        y = YAML::LoadFile(file);
    } catch (const YAML::Exception& e) {
        LOG(FATAL) << "cannot read " << file << ": " << e.what();
    }
    std::unordered_set<std::string> known;
    auto get = [&](const char* key, auto* v) {
        known.insert(key);
        if (y[key]) *v = y[key].as<std::decay_t<decltype(*v)>>();
    };
    get("voxel", &p->voxel);
    get("margin", &p->margin);
    get("max_range", &p->max_range);
    get("min_free", &p->min_free);
    get("ratio", &p->ratio);
    get("ground_dist", &p->ground_dist);
    get("min_hits", &p->min_hits);
    get("near_range", &p->near_range);
    get("up_z", &p->up_z);
    get("sensor_height", &p->sensor_height);
    get("min_range", &p->min_range);
    for (const auto& kv : y)
        if (!known.count(kv.first.as<std::string>())) LOG(FATAL) << "unknown key \"" << kv.first << "\" in " << file;
    if (!(p->voxel > 0)) LOG(FATAL) << "voxel must be > 0 in " << file;
}

struct Count {
    uint16_t free = 0, hit = 0;
};

inline Key MakeKey(int x, int y, int z) {
    constexpr int kOff = 1 << 20;
    return (Key(x + kOff) << 42) | (Key(y + kOff) << 21) | Key(z + kOff);
}

inline Key KeyOf(const Eigen::Vector3d& p, double inv) {
    return MakeKey(int(std::floor(p.x() * inv)), int(std::floor(p.y() * inv)), int(std::floor(p.z() * inv)));
}

bool LoadScan(const fs::path& file, Cloud* cloud) {
    return fs::exists(file) && pcl::io::loadPLYFile(file.string(), *cloud) >= 0;
}

// World-frame points of a scan; invalid ones get NaN so indices stay aligned with the cloud.
std::vector<Eigen::Vector3d> ToWorld(const Cloud& c, const Eigen::Isometry3d& T) {
    std::vector<Eigen::Vector3d> w(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        const auto& p = c[i];
        w[i] = std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
                   ? T * Eigen::Vector3d(p.x, p.y, p.z)
                   : Eigen::Vector3d::Constant(std::nan(""));
    }
    return w;
}

// Voxels the ray o->e crosses before the last `margin` metres (Amanatides & Woo).
void Traverse(const Eigen::Vector3d& o, const Eigen::Vector3d& e, const Param& prm, std::vector<Key>* out) {
    const double inv = 1.0 / prm.voxel;
    Eigen::Vector3d d = e - o;
    const double len = d.norm();
    if (len < 1e-6) return;
    d /= len;
    const double reach = std::min(len, prm.max_range) - prm.margin;
    if (reach <= 0) return;
    int idx[3], step[3];
    double t_max[3], t_delta[3];
    for (int k = 0; k < 3; ++k) {
        idx[k] = int(std::floor(o[k] * inv));
        step[k] = d[k] > 0 ? 1 : -1;
        if (std::abs(d[k]) < 1e-12) {
            t_max[k] = t_delta[k] = 1e30;
        } else {
            const double next = (idx[k] + (d[k] > 0 ? 1 : 0)) * prm.voxel;
            t_max[k] = (next - o[k]) / d[k];
            t_delta[k] = prm.voxel / std::abs(d[k]);
        }
    }
    while (true) {
        const int k = t_max[0] < t_max[1] ? (t_max[0] < t_max[2] ? 0 : 2) : (t_max[1] < t_max[2] ? 1 : 2);
        if (t_max[k] > reach) break;
        out->push_back(MakeKey(idx[0], idx[1], idx[2]));
        idx[k] += step[k];
        t_max[k] += t_delta[k];
    }
}

// Ground points of one scan, after Patchwork (Lim et al., RA-L 2021) without its likelihood estimation.
// `rel` are the points around the sensor with z up.  Space is cut into concentric zones, rings and
// sectors; in each patch a plane is fitted by iterating from the lowest points, and what lies on a
// plane that is flat and low enough is ground.  A scan alone decides, so moving objects cannot
// disturb it.  The height test matters near the sensor: a LiDAR that cannot see the ground there
// would otherwise take the lowest points of whoever walks by for it.
std::vector<char> SegmentGround(const std::vector<Eigen::Vector3d>& rel, const Param& prm) {
    constexpr double kBound[5] = {2.7, 12.3625, 22.025, 41.35, 80.0};
    constexpr int kRings[4] = {2, 4, 4, 4}, kSectors[4] = {16, 32, 54, 32};
    constexpr int kNumLpr = 20, kMinPts = 10, kIters = 3;
    constexpr double kSeedHeight = 0.4, kUpright = 0.707;
    constexpr double kHeightTol = 0.15, kMaxSlope = 0.08;  // [m], [m/m]: the terrain may rise or fall this much
    const double th_dist = prm.ground_dist;

    int offset[5] = {0, 0, 0, 0, 0};
    for (int z = 0; z < 4; ++z) offset[z + 1] = offset[z] + kRings[z] * kSectors[z];
    std::vector<std::vector<int>> patches(offset[4]);
    for (size_t i = 0; i < rel.size(); ++i) {
        const double r = rel[i].head<2>().norm();
        if (!(r >= kBound[0] && r < kBound[4])) continue;
        int z = 0;
        while (r >= kBound[z + 1]) ++z;
        const int ring = std::min(kRings[z] - 1, int((r - kBound[z]) / (kBound[z + 1] - kBound[z]) * kRings[z]));
        const double a = std::atan2(rel[i].y(), rel[i].x()) + M_PI;
        const int sector = std::min(kSectors[z] - 1, int(a / (2 * M_PI) * kSectors[z]));
        patches[offset[z] + ring * kSectors[z] + sector].push_back(int(i));
    }

    std::vector<char> ground(rel.size(), 0);
    for (std::vector<int>& ids : patches) {
        if (int(ids.size()) < kMinPts) continue;
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return rel[a].z() < rel[b].z(); });
        const int n_lpr = std::min<int>(kNumLpr, ids.size());
        double lpr = 0;
        for (int k = 0; k < n_lpr; ++k) lpr += rel[ids[k]].z();
        lpr /= n_lpr;
        std::vector<int> seeds;
        for (int i : ids)
            if (rel[i].z() < lpr + kSeedHeight) seeds.push_back(i);

        Eigen::Vector3d n = Eigen::Vector3d::UnitZ(), mean = Eigen::Vector3d::Zero();
        double d = 0;
        for (int it = 0; it < kIters; ++it) {
            if (int(seeds.size()) < 3) break;
            mean.setZero();
            for (int i : seeds) mean += rel[i];
            mean /= seeds.size();
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (int i : seeds) cov += (rel[i] - mean) * (rel[i] - mean).transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
            n = es.eigenvectors().col(0);  // smallest eigenvalue: the plane's normal
            if (n.z() < 0) n = -n;
            d = -n.dot(mean);
            seeds.clear();
            for (int i : ids)
                if (std::abs(n.dot(rel[i]) + d) < th_dist) seeds.push_back(i);
        }
        if (n.z() < kUpright) continue;
        const double tol = kHeightTol + kMaxSlope * mean.head<2>().norm();
        if (std::abs(mean.z() + prm.sensor_height) > tol) continue;
        for (int i : seeds) ground[i] = 1;
    }
    return ground;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

    std::string scans_dir, pose_fname, config_fname, out_arg;
    po::options_description desc("run_clean --scans_dir <dir> --pose_fname <file> [options]");
    desc.add_options()("help,h", "show this help")
        ("scans_dir", po::value<std::string>(&scans_dir), "folder with the <name>.ply scans")
        ("pose_fname", po::value<std::string>(&pose_fname), "pose file, one \"<name> x y z qx qy qz qw\" per line")
        ("config,c", po::value<std::string>(&config_fname), "yaml with the parameters (see config/clean.yaml); defaults if absent")
        ("out,o", po::value<std::string>(&out_arg), "output folder; default <scans_dir>/clean");
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
    Param prm;
    if (!config_fname.empty()) LoadParam(config_fname, &prm);
    LOG(INFO) << "voxel " << prm.voxel << " margin " << prm.margin << " max_range " << prm.max_range << " min_free "
              << prm.min_free << " ratio " << prm.ratio << " ground_dist " << prm.ground_dist << " min_hits "
              << prm.min_hits << " near_range " << prm.near_range << " up_z " << prm.up_z << " sensor_height "
              << prm.sensor_height << " min_range " << prm.min_range;
    const fs::path dir = scans_dir;
    if (!fs::is_directory(dir)) LOG(FATAL) << "not a folder: " << dir;
    const fs::path out = out_arg.empty() ? dir / "clean" : fs::path(out_arg);
    fs::create_directories(out);

    std::vector<PoseLine> pose_lines = LoadPoses(pose_fname);
    if (pose_lines.empty()) LOG(FATAL) << "no poses in " << pose_fname;
    const double inv = 1.0 / prm.voxel;

    // pass 1: count how many scans saw each voxel empty, and how many hit it
    std::unordered_map<Key, Count> map;
    std::cerr << "pass 1: count empty / hit scans per voxel\n";
    boost::progress_display bar1(pose_lines.size());
    for (const PoseLine& line : pose_lines) {
        ++bar1;
        Cloud cloud;
        if (!LoadScan(dir / (line.name + ".ply"), &cloud)) continue;
        const std::vector<Eigen::Vector3d> world = ToWorld(cloud, line.pose);
        const Eigen::Vector3d origin = line.pose.translation();

        // one ray per occupied voxel is enough
        std::unordered_map<Key, int> occupied;
        for (size_t i = 0; i < world.size(); ++i)
            if (std::isfinite(world[i].x())) occupied.emplace(KeyOf(world[i], inv), int(i));

        // a voxel next to a hit one is not called empty either: the surface may just have fallen between points
        std::unordered_set<Key> shield;
        for (const auto& kv : occupied) {
            const Eigen::Vector3d& w = world[kv.second];
            const int cx = int(std::floor(w.x() * inv)), cy = int(std::floor(w.y() * inv)),
                      cz = int(std::floor(w.z() * inv));
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) shield.insert(MakeKey(cx + dx, cy + dy, cz + dz));
        }

        std::vector<int> ids;
        ids.reserve(occupied.size());
        for (const auto& kv : occupied) ids.push_back(kv.second);
        std::vector<Key> free_keys;
#pragma omp parallel
        {
            std::vector<Key> local;
#pragma omp for nowait schedule(dynamic, 256)
            for (size_t i = 0; i < ids.size(); ++i) Traverse(origin, world[ids[i]], prm, &local);
#pragma omp critical
            free_keys.insert(free_keys.end(), local.begin(), local.end());
        }
        std::sort(free_keys.begin(), free_keys.end());
        free_keys.erase(std::unique(free_keys.begin(), free_keys.end()), free_keys.end());
        for (Key k : free_keys)
            if (!shield.count(k)) ++map[k].free;
        for (const auto& kv : occupied) ++map[kv.first].hit;
    }
    LOG(INFO) << "pass 1 done, " << map.size() << " voxels";

    // pass 2: write each scan without the points sitting in a voxel that was empty more often than hit
    size_t total = 0, removed = 0, written = 0, kept_ground = 0;
    std::cerr << "pass 2: write the scans without dynamic points\n";
    boost::progress_display bar2(pose_lines.size());
    for (const PoseLine& line : pose_lines) {
        ++bar2;
        Cloud cloud;
        if (!LoadScan(dir / (line.name + ".ply"), &cloud)) {
            LOG(WARNING) << "no readable scan for " << line.name << ", skipped";
            continue;
        }
        for (auto pt_it = cloud.begin(); pt_it != cloud.end(); ) {
            if (pt_it->getVector3fMap().norm() < prm.min_range)
                pt_it = cloud.erase(pt_it);
            else
                pt_it++;
        }
        const std::vector<Eigen::Vector3d> world = ToWorld(cloud, line.pose);
        std::vector<char> ground(cloud.size(), 0);
        if (prm.ground_dist > 0) {
            std::vector<Eigen::Vector3d> rel(cloud.size());  // around the sensor, world axes, so z is up
            for (size_t i = 0; i < cloud.size(); ++i) {
                rel[i] = world[i] - line.pose.translation();
                rel[i].z() *= prm.up_z;  // SegmentGround wants z up
            }
            ground = SegmentGround(rel, prm);
        }
        Cloud clean;
        clean.reserve(cloud.size());
        for (size_t i = 0; i < cloud.size(); ++i) {
            ++total;
            if (!std::isfinite(world[i].x())) continue;
            if (ground[i]) {
                ++kept_ground;
                clean.push_back(cloud[i]);
                continue;
            }
            const Count& c = map.at(KeyOf(world[i], inv));
            const int hit_others = c.hit - 1;  // this scan's own hit does not count against the point
            const bool seen_empty = c.free >= prm.min_free && c.free >= prm.ratio * hit_others;
            // Rays cannot clear what is only ever seen with someone standing in front of it.  But a
            // real surface this close is hit scan after scan, something walking by is not.
            const bool fleeting = c.hit <= prm.min_hits && (world[i] - line.pose.translation()).norm() < prm.near_range;
            if (seen_empty || fleeting) {
                ++removed;
                continue;
            }
            clean.push_back(cloud[i]);
        }
        clean.width = clean.size();
        clean.height = 1;
        clean.is_dense = true;
        pcl::io::savePLYFileBinary((out / (line.name + ".ply")).string(), clean);
        ++written;
    }
    LOG(INFO) << "wrote " << written << " scans to " << out << ", removed " << removed << " of " << total
              << " points, " << kept_ground << " ground points protected (" << 100.0 * removed / std::max<size_t>(total, 1) << " %)";
    return 0;
}

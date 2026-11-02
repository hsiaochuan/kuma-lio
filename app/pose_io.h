//
// The pose file shared by run_pgo and run_clean: one "<name> x y z qx qy qz qw" (T_world_body) per
// scan, <name> being both the scan's file stem and its time in seconds.
//
#ifndef FASTER_LIO_POSE_IO_H
#define FASTER_LIO_POSE_IO_H

#include <glog/logging.h>

#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct PoseLine {
    std::string name;
    double time;
    Eigen::Isometry3d pose;
};

inline std::vector<PoseLine> LoadPoses(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) LOG(FATAL) << "cannot open " << file;
    std::vector<PoseLine> poses;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        PoseLine p;
        double v[7];
        if (!(ss >> p.name >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6])) {
            LOG(WARNING) << "skip malformed pose line: " << line;
            continue;
        }
        try {
            p.time = std::stod(p.name);
        } catch (const std::exception&) {
            LOG(WARNING) << "skip pose line, the name is not a number: " << line;
            continue;
        }
        p.pose = Eigen::Isometry3d::Identity();
        p.pose.linear() = Eigen::Quaterniond(v[6], v[3], v[4], v[5]).normalized().toRotationMatrix();
        p.pose.translation() = Eigen::Vector3d(v[0], v[1], v[2]);
        poses.push_back(std::move(p));
    }
    return poses;
}


#endif  // FASTER_LIO_POSE_IO_H

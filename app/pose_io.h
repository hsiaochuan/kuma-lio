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

#pragma once

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

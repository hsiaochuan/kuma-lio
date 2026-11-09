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

#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_lib.h"
#include "cameras/cameras.h"
namespace kuma_lio {
namespace ba {

using IndexT = uint32_t;
constexpr IndexT kUndefinedIndex = static_cast<IndexT>(-1);

// T_world_body, indexed by Scan::id_pose
using Poses = std::map<IndexT, Eigen::Isometry3d, std::less<IndexT>,
                       Eigen::aligned_allocator<std::pair<const IndexT, Eigen::Isometry3d>>>;

// One laser frame
struct Scan {
    std::string path;             // its point cloud file
    IndexT id_pose = kUndefinedIndex;
    std::vector<Vec3> points;     // body frame
};

// All the laser frames, indexed by scan id
using Scans = std::map<IndexT, std::shared_ptr<Scan>>;

// One image; only its file is known, its pose is BaData::poses[id_pose] * BaData::extrin.
struct RgbImage {
    std::string path;
    IndexT id_pose = kUndefinedIndex;
    IndexT id_camera = kUndefinedIndex;
};
using RgbImages = std::map<IndexT, std::shared_ptr<RgbImage>>;
using Cameras = std::map<IndexT, std::shared_ptr<CamModel>>;
struct BaData {
    Poses poses;
    Scans scans;
    RgbImages rgb_images;
    Cameras cameras;
    // T_body_cam, the camera in the body frame of the poses: P_body = R P_cam + t
    Eigen::Isometry3d extrin = Eigen::Isometry3d::Identity();

    // Write `dir` as the --input_path of `colmap point_triangulator`: cameras.txt (the cameras, by CAMERA_ID;
    // left out when there is none), images.txt and an empty points3D.txt.  Every image uses camera 1, is named by
    // its file name (the name the database has under --image_path) and has cam_from_world = (pose * extrin)^-1,
    // with no 2D point.  False if a file cannot be written.
    bool SaveCOLMAP(const std::string& dir) const {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return false;
        const std::filesystem::path folder(dir);

        if (!cameras.empty()) {
            if (!cameras.count(1)) LOG(WARNING) << "no camera 1, which every image of images.txt refers to";
            if (!WriteCamerasToCamerasText((folder / "cameras.txt").string(), cameras)) return false;
        } else {
            LOG(WARNING) << "no camera: cameras.txt not written, point_triangulator needs one with camera 1";
        }

        std::ofstream points(folder / "points3D.txt");
        if (!points) return false;
        points << "# 3D point list with one line of data per point:\n"
               << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
               << "# Number of points: 0, mean track length: 0\n";
        if (!points.good()) return false;

        std::ofstream images(folder / "images.txt");
        if (!images) return false;
        images << std::setprecision(17);
        images << "# Image list with two lines of data per image:\n"
               << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
               << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
               << "# Number of images: " << rgb_images.size() << ", mean observations per image: 0\n";
        for (const auto& image_it : rgb_images) {
            const RgbImage& image = *image_it.second;
            const Eigen::Isometry3d cam_from_world = (poses.at(image.id_pose) * extrin).inverse();
            const Eigen::Quaterniond q(cam_from_world.rotation());
            const Eigen::Vector3d t = cam_from_world.translation();
            // COLMAP ids start at 1; the empty second line is the image's (absent) 2D points, and is required
            images << image_it.first + 1 << ' ' << q.w() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' '
                   << t.x() << ' ' << t.y() << ' ' << t.z() << ' ' << 1 << ' '
                   << std::filesystem::path(image.path).filename().string() << "\n\n";
        }
        return images.good();
    }

    bool SavePairList(const std::string& pair_list_fname, double hfov = 73.8, double vfov = 58.8,
                      double radius = 10.0) const {
        std::ofstream pair_ofs(pair_list_fname);
        if (!pair_ofs)
            return false;
        const double tan_h = std::tan(hfov * EIGEN_PI / 360.0);
        const double tan_v = std::tan(vfov * EIGEN_PI / 360.0);

        struct View {
            Eigen::Isometry3d world_from_cam, cam_from_world;
            std::string name;
        };
        std::vector<View, Eigen::aligned_allocator<View>> views;
        views.reserve(rgb_images.size());
        for (const auto& image_it : rgb_images) {
            const RgbImage& image = *image_it.second;
            View v;
            v.world_from_cam = poses.at(image.id_pose) * extrin;
            v.cam_from_world = v.world_from_cam.inverse();
            v.name = std::filesystem::path(image.path).filename().string();
            views.push_back(std::move(v));
        }

        // a sees b: a point of b's optical axis, up to `radius` deep, lies in a's view pyramid
        constexpr int kSamples = 8;
        auto IsSeen = [&](const View& a, const View& b) {
            for (int k = 1; k <= kSamples; ++k) {
                const Eigen::Vector3d pc = a.cam_from_world * (b.world_from_cam * Eigen::Vector3d(0, 0, radius * k / kSamples));
                if (pc.z() > 0 &&
                    pc.z() <= radius &&
                    std::abs(pc.x()) <= tan_h * pc.z() &&
                    std::abs(pc.y()) <= tan_v * pc.z())
                    return true;
            }
            return false;
        };
        for (size_t i = 0; i < views.size(); ++i)
            for (size_t j = i + 1; j < views.size(); ++j) {
                const View& a = views[i];
                const View& b = views[j];
                if ((a.world_from_cam.translation() - b.world_from_cam.translation()).norm() >= radius)
                    continue;
                if (IsSeen(a, b) && IsSeen(b, a))
                    pair_ofs << a.name << ' ' << b.name << '\n';
            }
        return pair_ofs.good();
    }
    bool SaveTumPoses(const std::string& tum_fname) const {
        std::ofstream ofs(tum_fname);
        if (!ofs) return false;
        ofs << std::setprecision(9);
        for (const auto& scan_it : scans) {
            const Scan& scan = *scan_it.second;
            const Eigen::Isometry3d& pose = poses.at(scan.id_pose);
            const Eigen::Vector3d t = pose.translation();
            const Eigen::Quaterniond q(pose.rotation());
            ofs << std::filesystem::path(scan.path).stem().string() << ' ' << t.x() << ' ' << t.y() << ' ' << t.z()
                << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
        }
        return ofs.good();
    }
};

}  // namespace ba
}  // namespace kuma_lio

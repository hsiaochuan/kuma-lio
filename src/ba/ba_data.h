#ifndef FASTER_LIO_BA_SFM_DATA_H
#define FASTER_LIO_BA_SFM_DATA_H

//
// The scene of run_ba, the lidar part of openMVG's SfM_Data (sfm_data.hpp, sfm_scan.hpp) that main_BALVM
// optimizes: the poses, and the laser frames that hang on them.  A scan names its pose by Scan::id_pose, a
// plane cluster names it by FrameCluster::xid, both keys of BaData::poses.
//

#include <Eigen/Geometry>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_lib.h"

namespace faster_lio {
namespace ba {

using IndexT = uint32_t;
constexpr IndexT kUndefinedIndex = static_cast<IndexT>(-1);

/// T_world_body, indexed by Scan::id_pose
using Poses = std::map<IndexT, Eigen::Isometry3d, std::less<IndexT>,
                       Eigen::aligned_allocator<std::pair<const IndexT, Eigen::Isometry3d>>>;

/// One laser frame
struct Scan {
    std::string path;             ///< its point cloud file
    IndexT id_pose = kUndefinedIndex;
    std::vector<Vec3> points;     ///< body frame
};

/// All the laser frames, indexed by scan id
using Scans = std::map<IndexT, std::shared_ptr<Scan>>;

/// One image; only its file is known, its pose is BaData::poses[id_pose] * BaData::extrin.
struct RgbImage {
    std::string path;
    IndexT id_pose = kUndefinedIndex;
};
using RgbImages = std::map<IndexT, std::shared_ptr<RgbImage>>;

struct BaData {
    Poses poses;
    Scans scans;
    RgbImages rgb_images;

    /// T_body_cam, the camera in the body frame of the poses: P_body = R P_cam + t
    Eigen::Isometry3d extrin = Eigen::Isometry3d::Identity();

    /// Write `dir` as the --input_path of `colmap point_triangulator`: images.txt and an empty points3D.txt.  The
    /// cameras.txt holding camera 1 is not written (config/cameras.txt is the i2Nav-Robot left camera).  Every image
    /// uses camera 1, is named by its file name (the name the database has under --image_path) and has
    /// cam_from_world = (pose * extrin)^-1, with no 2D point.  False if a file cannot be written.
    bool SaveCOLMAP(const std::string& dir) const {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return false;
        const std::filesystem::path folder(dir);

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

    bool SaveTumPoses(const std::string& file) const {
        std::ofstream out(file);
        if (!out) return false;
        out << std::setprecision(9);
        for (const auto& scan_it : scans) {
            const Scan& scan = *scan_it.second;
            const Eigen::Isometry3d& pose = poses.at(scan.id_pose);
            const Eigen::Vector3d t = pose.translation();
            const Eigen::Quaterniond q(pose.rotation());
            out << std::filesystem::path(scan.path).stem().string() << ' ' << t.x() << ' ' << t.y() << ' ' << t.z()
                << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
        }
        return out.good();
    }
};

}  // namespace ba
}  // namespace faster_lio

#endif  // FASTER_LIO_BA_SFM_DATA_H

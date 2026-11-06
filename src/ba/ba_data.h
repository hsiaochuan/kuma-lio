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

struct BaData {
    Poses poses;
    Scans scans;

    /// Write the pose of every scan, in scan order, as TUM "timestamp x y z qx qy qz qw", the timestamp being the
    /// stem of the scan's file; false if the file cannot be written.
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

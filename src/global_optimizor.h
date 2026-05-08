#pragma once
#include <pcl/filters/uniform_sampling.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/container/flat_map.hpp>
#include <boost/progress.hpp>
#include <cmath>
#include <future>
#include <unordered_set>
#include "common_lib.h"
#include "pose3.h"
#include "stamp_pose.h"
#include "thread_pool.h"

namespace fs = boost::filesystem;
namespace faster_lio {
using scan_pair_t = uint64_t;
struct ScanPair {
    scan_t scan_id1_;
    scan_t scan_id2_;
    ScanPair(scan_t scan_id1, scan_t scan_id2) : scan_id1_(scan_id1), scan_id2_(scan_id2) {
        CHECK(scan_id1 != scan_id2);
        if (scan_id1 > scan_id2) {
            std::swap(scan_id1_, scan_id2_);
        }
    }

    bool operator==(const ScanPair& other) const {
        return (scan_id1_ == other.scan_id1_ && scan_id2_ == other.scan_id2_);
    }
};
scan_pair_t ScanPairToId(const ScanPair& pair);
ScanPair IdToScanPair(scan_pair_t id);
}  // namespace faster_lio
namespace std {
template <>
struct hash<faster_lio::ScanPair> {
    size_t operator()(const faster_lio::ScanPair& scan_pair) const { return faster_lio::ScanPairToId(scan_pair); }
};
}  // namespace std

namespace faster_lio {
struct PairData {
    int source_points_count = 0;
    int target_points_count = 0;
    Pose3 ab_rel_pose;
    double average_error;
    int corres_count = 0;
};

class GlobalOptimizor {
   public:
    struct Options {
        // loop weight
        bool lc_enable = true;
        double loop_weight = 100.0;

        // ba
        bool ba_enable = false;
        int ba_iters = 3;

        // keyframe selection thresholds
        double keyframe_dist_threshold = 1.0;  // meters
        double keyframe_angle_threshold = 10;
        double keyframe_time_threshold = 1.0;  // seconds

        // lc
        bool output_all_loop_reigster_result = true;
        double lc_detect_dist_thr = 5.0;
        double lc_detect_temporal_dist_thr = 60.0;
        int sub_map_interval = 5;
        void LoadFromYaml(const std::string& config_fname);
    };

    GlobalOptimizor::Options options_;
    boost::container::flat_map<scan_t, ScanFrame::Ptr> scans_;
    boost::container::flat_map<scan_t, ScanFrame::Ptr> keyscans_;
    std::unordered_map<ScanPair, PairData> loops_buf;
    std::string output_dir;

    void AddScan(ScanFrame::Ptr scan);
    void ScanFilter();
    void BundleAdjustment();
    void PoseGraphOptimize();

    std::unordered_map<ScanPair, PairData> DetectLoopClosure();
    void SaveLoopToPcd(const std::string& save_fname);
    void ExportMap(const std::string& save_fname);
    Trajectory ExportStampedPoses();
    PointCloud::Ptr GetSubMap(const scan_t& scan_id);
};

}  // namespace faster_lio

#ifndef FASTER_LIO_IESKF_LIO_H
#define FASTER_LIO_IESKF_LIO_H

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "common_lib.h"
#include "eskf.h"
#include "imu_processing.hpp"
#include "ivox3d.h"
#include "laser_mapping_param.h"
#include "state_point.h"
#include "visual_manager.h"

namespace faster_lio {

/// faster-lio estimation: IESKF + iVox local map, with optional visual update
/// and prior-map localization. The pipeline (LaserMapping) feeds synced
/// measurements in and reads the estimated state and registered scans out.
class IeskfLio {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    using IVoxType = IVox;

    IeskfLio();

    bool Init(const std::shared_ptr<LaserMappingParam> &param);

    /// process one synced measurement, return true if the state got updated,
    /// false when the algorithm is still initializing or the scan is skipped
    bool ProcessMeasure(MeasureGroup &measures);

    /// current estimated state, shared with the pipeline
    StatePoint::Ptr State() { return state_point_; }

    /// outputs of the last processed measure
    PointCloud::Ptr ScanUndistort() { return scan_undistort_; }
    PointCloud::Ptr ScanDownWorld() { return scan_down_world_; }

    /// effective (inlier) points of the last update, for visualization
    const std::vector<char> &EffMask() const { return eff_mask_; }
    int EffNum() const { return eff_num_; }

    /// prior map based localization
    void LoadPriorMap(const std::string &prior_map_fname);
    void SetPriorInitPose(const Vec3 &pos, const Eigen::Quaterniond &rot) {
        prior_init_position_ = pos;
        prior_init_rotation_ = rot;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr PriorMap() { return map_cloud_; }

    /// custom observation model for IEKF update
    bool BuildLidarObservation(const StatePoint &s, LidarObservation &obs);
    void MapIncremental();

    void PrintState(const StatePoint &s);

   public:
    /// modules
    std::shared_ptr<IVoxType> ivox_ = nullptr;     // localmap in ivox
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;  // imu process
    std::shared_ptr<VisualManager> visual_manager = nullptr;

    /// point clouds data
    PointCloud::Ptr scan_undistort_{new PointCloud()};   // scan after undistortion, not downsampled
    PointCloud::Ptr scan_down_body_{new PointCloud()};   // downsampled scan in body
    PointCloud::Ptr scan_down_world_{new PointCloud()};  // downsampled scan in world
    std::vector<PointVector> nearest_points_;             // nearest points of current scan
    std::vector<Vec4f> plane_coeffs_;
    pcl::VoxelGrid<Point> scan_sampler_;  // voxel filter for current scan
    std::vector<char> eff_mask_;          // selected points

    bool if_local_map_init_ = true;
    int eff_num_ = 0;

    std::shared_ptr<StatePoint> state_point_;
    std::shared_ptr<LaserMappingParam> param;

    // prior map
    using PriorMapPoint = pcl::PointXYZ;
    pcl::PointCloud<PriorMapPoint>::Ptr map_cloud_;
    pcl::KdTreeFLANN<PriorMapPoint>::Ptr map_kd_tree_;
    std::vector<Vec3f> map_normals_;
    Eigen::Vector3d prior_init_position_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prior_init_rotation_ = Eigen::Quaterniond::Identity();
};

}  // namespace faster_lio

#endif  // FASTER_LIO_IESKF_LIO_H

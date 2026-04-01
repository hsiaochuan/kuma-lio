#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <condition_variable>
#include <thread>

#include "imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "options.h"
#include "pointcloud_preprocess.h"

namespace faster_lio {

class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
#endif

    LaserMapping();
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init with ros
    bool InitROS(ros::NodeHandle &nh);

    /// init without ros
    bool InitWithoutROS(const std::string &config_yaml);

    void Run();

    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);
    void ImageCallBack(const sensor_msgs::Image::ConstPtr &msg_in);
    void CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in);

    // sync lidar with imu
    bool SyncPackages();

    /// interface of mtk, customized obseravtion model
    void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
    void PublishPath(const ros::Publisher pub_path);
    void PublishOdometry(const ros::Publisher &pub_odom_aft_mapped);
    void PublishFrameWorld();
    void PublishFrameBody(const ros::Publisher &pub_laser_cloud_body);
    void PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world);
    void Savetrajectory(const std::string &traj_file);

    void Finish();

   private:
    template <typename T>
    void SetPosestamp(T &out);

    void PointBodyToWorld(PointType const *pi, PointType *const po);
    void PointBodyToWorld(const common::V3F &pi, PointType *const po);
    void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);

    void MapIncremental();

    void SubAndPubToROS(ros::NodeHandle &nh);

    bool LoadParams(ros::NodeHandle &nh);
    bool LoadParamsFromYAML(const std::string &yaml);

    void PrintState(const state_ikfom &s);

   public:
    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

    /// local map related
    float det_range_ = 300.0f;
    double cube_len_ = 0;
    double map_filter_size_ = 0;

    /// params
    std::vector<double> extrinT_{0.0, 0.0, 0.0};  // lidar-imu translation
    std::vector<double> extrinR_;                 // lidar-imu rotation

    /// point clouds data
    PointCloud::Ptr scan_undistort_{new PointCloud()};   // scan after undistortion, not downsampled
    PointCloud::Ptr scan_down_body_{new PointCloud()};   // downsampled scan in body
    PointCloud::Ptr scan_down_world_{new PointCloud()};  // downsampled scan in world
    std::vector<PointVector> nearest_points_;            // nearest points of current scan
    common::VV4F corr_pts_;                              // inlier pts
    common::VV4F corr_norm_;                             // inlier plane norms
    pcl::VoxelGrid<PointType> scan_sampler_;             // voxel filter for current scan
    std::vector<float> residuals_;                       // point-to-plane residuals
    std::vector<char> point_selected_surf_;              // selected points
    common::VV4F plane_coef_;                            // plane coeffs

    /// topics
    std::string lidar_topic_;
    std::string imu_topic_;
    std::string camera_topic_;
    bool camera_enable_;
    double lidar_time_offset_ = 0.;
    double camera_time_offset_ = 0.;
    /// ros pub and sub stuffs
    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_body_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;
    std::string tf_imu_frame_;
    std::string tf_world_frame_;

    double first_scan_time_;
    bool if_first_scan_ = true;

    std::mutex mtx_buffer_;
    std::deque<Point> points_buffer_;
    std::deque<Imu> imu_buffer_;
    std::deque<double> img_time_buffer_;

    double last_timestamp_lidar_ = 0;
    double last_timestamp_imu_ = -1.0;
    double last_timestamp_camera_ = 0.0;
    /// options
    double scan_interval_ = 0.1;
    double lidar_end_time_ = 0;
    float esti_plane_thr = 0.1;
    int max_iteraions = 4;
    /// statistics and flags ///
    bool if_local_map_init_ = true;
    int pcd_index_ = 0;
    int effect_feat_num_ = 0;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    common::MeasureGroup measures_;                    // sync IMU and lidar scan
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
    state_ikfom state_point_;                          // ekf current state
    bool extrinsic_est_en_ = true;

    /////////////////////////  debug show / save /////////////////////////////////////////////////////////
    bool run_in_offline_ = false;
    bool path_pub_en_ = true;
    bool scan_pub_en_ = false;
    bool dense_pub_en_ = false;
    bool scan_body_pub_en_ = false;
    bool scan_effect_pub_en_ = false;
    bool pcd_save_en_ = false;
    int pcd_save_interval_ = -1;
    bool path_save_en_ = false;

    PointCloud::Ptr pcl_wait_save_{new PointCloud()};  // debug save
    nav_msgs::Path path_;
   public:
    std::string output_dir;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H
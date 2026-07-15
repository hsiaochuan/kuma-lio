#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H
#include <pcl/kdtree/kdtree_flann.h>
#include "laser_mapping_param.h"
#include "livox_ros_driver/CustomMsg.h"
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <visualization_msgs/Marker.h>

// Heavy dependencies are forward-declared below to reduce rebuilds.
#include <rosbag/bag.h>

#include "eskf.h"
#include "global_optimizor.h"
#include "imu_processing.hpp"
#include "ivox3d.h"
#include "pointcloud_preprocess.h"
#include "pose3.h"
#include "stamp_pose.h"
#include "types.h"
#include "global_optimizor.h"
#include "visual_manager.h"
namespace faster_lio {

class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    using IVoxType = IVox;

    LaserMapping();
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        bag_.close();
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init without ros
    bool Init(const std::string &config_fname);
    void LoadPriorMap(const std::string &prior_map_fname);
    void Run();
    void PostUpdate();
    void PublishROSMsg();
    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void VelodyneScanCallBack(const velodyne_msgs::VelodyneScan::ConstPtr &msg);
    void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);
    void ImageCallBack(Image& image);
    void ImageMsgCallBack(const sensor_msgs::Image::ConstPtr &msg_in);
    void CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in);
    void AddScanToBuffer(const PointCloud::Ptr &scan);
    // sync lidar with imu
    bool SyncPackages();

    /// custom observation model for IEKF update
    bool BuildLidarObservation(const StatePoint &s, LidarObservation &obs);

    ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
    void PublishOdometry();
    void PublishFrameWorld();
    void PublishFrameEffectWorld();
    void PublishImage();
    void PublishFrustrum();
    void Savetrajectory(const std::string &traj_file);

    void Finish();
    void MapIncremental();

    void SubAndPubToROS(ros::NodeHandle &nh);

    void PrintState(const StatePoint &s);

   public:
    /// modules
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process
    std::shared_ptr<GlobalOptimizor> mapper = nullptr;
    std::shared_ptr<VisualManager> visual_manager = nullptr;
    /// point clouds data
    PointCloud::Ptr scan_undistort_{new PointCloud()};   // scan after undistortion, not downsampled
    PointCloud::Ptr scan_down_body_{new PointCloud()};   // downsampled scan in body
    PointCloud::Ptr scan_down_world_{new PointCloud()};  // downsampled scan in world
    ColorPointCloud::Ptr color_scan_world_{new ColorPointCloud()};  // downsampled scan in world with color
    std::vector<PointVector> nearest_points_;            // nearest points of current scan
    std::vector<Vec4f> plane_coeffs_;
    pcl::VoxelGrid<Point> scan_sampler_;             // voxel filter for current scan
    std::vector<char> eff_mask_;              // selected points

    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;
    ros::Publisher pub_image_;
    ros::Publisher pub_frustrum_;

    double first_scan_time_ = std::numeric_limits<double>::quiet_NaN();

    std::mutex mtx_buffer_;
    std::deque<Point> points_buffer_;
    std::deque<Imu> imu_buffer_;
    std::deque<Image> image_buffer_;

    double last_timestamp_lidar_ = 0;
    double last_timestamp_imu_ = -1.0;
    double last_timestamp_camera_ = 0.0;
    bool if_local_map_init_ = true;
    int eff_num_ = 0;

    MeasureGroup measures_;
    std::shared_ptr<StatePoint> state_point_;
    int pcd_idx = 0;
    PointCloud::Ptr pcl_wait_save_{new PointCloud()};
    Trajectory trajectory_;

    std::shared_ptr<LaserMappingParam> param;
    rosbag::Bag bag_;

    // prior map
    using PriorMapPoint = pcl::PointXYZ;
    pcl::PointCloud<PriorMapPoint>::Ptr map_cloud_;
    pcl::KdTreeFLANN<PriorMapPoint>::Ptr map_kd_tree_;
    std::vector<Vec3f> map_normals_;
    Eigen::Vector3d prior_init_position_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prior_init_rotation_ = Eigen::Quaterniond::Identity();
   public:
    std::string output_dir;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H
#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H
#include "laser_mapping_param.h"
#include "livox_ros_driver/CustomMsg.h"
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/CompressedImage.h>
#include <visualization_msgs/Marker.h>

// Heavy dependencies are forward-declared below to reduce rebuilds.
#include <rosbag/bag.h>

#include "global_optimizor.h"
#include "ieskf_lio.h"
#include "pointcloud_preprocess.h"
#include "pose3.h"
#include "stamp_pose.h"
#include "types.h"
namespace faster_lio {

/// Data pipeline around the odometry algorithm: ROS IO, buffering and time sync,
/// publishing and result saving. The estimation itself lives in IeskfLio.
class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    LaserMapping();
    ~LaserMapping() {
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

    /// build the colored scan for publishing / saving
    void BuildColorScan();

    ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
    void PublishOdometry();
    void PublishFrameWorld();
    void PublishFrameEffectWorld();
    void PublishImage();
    void PublishFrustrum();
    void Savetrajectory(const std::string &traj_file);

    void Finish();

    void SubAndPubToROS(ros::NodeHandle &nh);

   public:
    /// modules
    std::shared_ptr<IeskfLio> ieskf_lio = nullptr;                    // state estimation algorithm
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<GlobalOptimizor> mapper = nullptr;

    /// point clouds data
    ColorPointCloud::Ptr color_scan_world_{new ColorPointCloud()};  // downsampled scan in world with color
    pcl::VoxelGrid<Point> scan_sampler_;  // voxel filter for saved map clouds

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

    MeasureGroup measures_;
    int pcd_idx = 0;
    PointCloud::Ptr pcl_wait_save_{new PointCloud()};
    Trajectory trajectory_;

    std::shared_ptr<LaserMappingParam> param;
    rosbag::Bag bag_;

    // prior pose for localization, forwarded to the odometry in Init()
    Eigen::Vector3d prior_init_position_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prior_init_rotation_ = Eigen::Quaterniond::Identity();
   public:
    std::string output_dir;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H

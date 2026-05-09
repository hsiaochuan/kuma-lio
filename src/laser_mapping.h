#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H
#include "laser_mapping_param.h"
#include "livox_ros_driver/CustomMsg.h"
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

// Heavy dependencies are forward-declared below to reduce rebuilds.
#include "imu_processing.hpp"
#include "ivox3d/ivox3d.h"
#include "pointcloud_preprocess.h"
#include "pose3.h"
#include "types.h"
#include "stamp_pose.h"
#include "sfm_data.h"
#include "global_optimizor.h"
namespace faster_lio {

class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, Point>;
#endif

    LaserMapping();
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init without ros
    bool Init(const std::string &config_fname);

    void Run();

    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);
    void ImageCallBack(Image& image);
    void ImageMsgCallBack(const sensor_msgs::Image::ConstPtr &msg_in);
    void CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in);

    // sync lidar with imu
    bool SyncPackages();

    /// interface of mtk, customized obseravtion model
    void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
    void PublishPath();
    void PublishOdometry();
    void PublishFrameWorld() const;
    void PublishFrameEffectWorld();
    void Savetrajectory(const std::string &traj_file);

    void Finish();
    void MapIncremental();

    void SubAndPubToROS(ros::NodeHandle &nh);

    void PrintState(const state_ikfom &s);

   public:
    /// modules
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process
    std::shared_ptr<GlobalOptimizor> mapper = nullptr;

    /// point clouds data
    PointCloud::Ptr scan_undistort_{new PointCloud()};   // scan after undistortion, not downsampled
    PointCloud::Ptr scan_down_body_{new PointCloud()};   // downsampled scan in body
    PointCloud::Ptr scan_down_world_{new PointCloud()};  // downsampled scan in world
    std::vector<PointVector> nearest_points_;            // nearest points of current scan
    std::vector<Vec4f> corr_pts_;                              // inlier pts
    std::vector<Vec4f> corr_norm_;                             // inlier plane norms
    pcl::VoxelGrid<Point> scan_sampler_;             // voxel filter for current scan
    std::vector<float> residuals_;                       // point-to-plane residuals
    std::vector<char> point_selected_surf_;              // selected points
    std::vector<Vec4f> plane_coef_;                            // plane coeffs

    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;

    double first_scan_time_ = std::numeric_limits<double>::quiet_NaN();

    std::mutex mtx_buffer_;
    std::deque<Point> points_buffer_;
    std::deque<Imu> imu_buffer_;
    std::deque<Image> image_buffer_;

    double last_timestamp_lidar_ = 0;
    double last_timestamp_imu_ = -1.0;
    double last_timestamp_camera_ = 0.0;
    double lidar_end_time_ = 0;
    bool if_local_map_init_ = true;
    int effect_feat_num_ = 0;

    MeasureGroup measures_;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
    state_ikfom state_point_;
    int pcd_idx = 0;
    PointCloud::Ptr pcl_wait_save_{new PointCloud()};
    nav_msgs::Path path_;
    Trajectory trajectory_;
    sfm_data sfm_data_;


    std::shared_ptr<LaserMappingParam> param;
   public:
    std::string output_dir;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H
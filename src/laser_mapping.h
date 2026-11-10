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

#include "laser_mapping_param.h"
#include "livox_ros_driver/CustomMsg.h"
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>

#include <rosbag/bag.h>
#include "fast_lio.h"
#include "pointcloud_preprocess.h"
#include "resple_lio.h"
#include "stamp_pose.h"

namespace kuma_lio {


class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    LaserMapping();
    ~LaserMapping() {
        bag_.close();
        LOG(INFO) << "laser mapping deconstruct";
    }

    // init without ros
    bool Init(const std::string &config_fname, const std::string& camera_fname);
    void RunRESPLE();
    void RunFastLio();
    enum class DynamicInitResult { SUCCESS, STATIONARY, FAIL };
    DynamicInitResult TryDynamicInit(std::int64_t window_begin_ns, std::int64_t window_end_ns);
    void PostUpdate();
    void PostUpdateFastLio();
    void PublishROSMsg();
    void PublishROSMsgFastLio();
    // callbacks of lidar and imu
    void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);
    void ImageCallBack(const sensor_msgs::Image::ConstPtr &msg);
    void CompressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr &msg);
    void GnssSolutionCallback(const sensor_msgs::NavSatFix::ConstPtr &msg);
    bool ReadObsNav(const std::string &obs_fname, const std::string &nav_fname, const std::string &sol_fname);
    void AddScanToBuffer(const PointCloud::Ptr &scan);
    bool CollectMeasures(const std::int64_t& meas_end);
    // The next frame of the FAST-LIO front end: lba.frame_dt long, or up to the next image.
    void ClearMeasuresBefore(const std::int64_t& time_ns);
    bool CollectMeasuresFastLio();
    void PublishOdometry(const Eigen::Isometry3d& pub_pose, std::int64_t time_ns);
    void PublishFrameWorld();
    void PublishFrameEffectWorld();
    void PublishFrustrum(std::int64_t time_ns);
    void Savetrajectory();

    void Finish();

    void InitialSubscribers(ros::NodeHandle& nh);
    void InitialPublishers(ros::NodeHandle& nh);
   public:
    // modules
    std::shared_ptr<RESPLE_LIO> resple_lio_ = nullptr;  // front_end: resple
    std::shared_ptr<FastLio> fast_lio_ = nullptr;       // front_end: fast_lio
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess

    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;
    ros::Publisher pub_frustrum_;

    double global_offset_time = std::numeric_limits<double>::quiet_NaN();

    std::mutex mtx_buffer_;
    std::deque<Point> points_buffer_;
    std::deque<Imu> imu_buffer_;
    std::deque<RgbImage> image_buff_;
    obs_t obss{};
    nav_t nav{};
    std::deque<int64_t> obss_time_ns_buff_;
    std::deque<int> obss_id_buff_;

    MeasureGroup measures_;
    Trajectory trajectory_;

    std::shared_ptr<LaserMappingParam> param;
    rosbag::Bag bag_;

    bool if_inertial_initialized = false;
    bool if_localmap_initialized = false;
   public:
    std::string output_dir;
    bool dynamic_init_ = false;
};

}  // namespace kuma_lio

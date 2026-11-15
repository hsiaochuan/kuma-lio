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

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <deque>
#include <memory>
#include <mutex>

#include "livox_ros_driver/CustomMsg.h"
#include "pointcloud_preprocess.h"
#include "resple_lio.h"
#include "resple_param.h"
#include "stamp_pose.h"

namespace kuma_lio {

// run_resple: the RESPLE spline LIO on its ikd-tree local map, nothing else (no BA, no PGO, no camera, no GNSS).
class RespleMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    RespleMapping() = default;
    ~RespleMapping() { bag_.close(); }

    bool Init(const std::string& config_fname);
    void InitialPublishers(ros::NodeHandle& nh);
    // Runs the filter on everything the buffers hold.
    void Run();

    void IMUCallBack(const sensor_msgs::Imu::ConstPtr& msg_in);
    PointCloud::Ptr ScanFromPCL(const sensor_msgs::PointCloud2::ConstPtr& msg) const;
    PointCloud::Ptr ScanFromLivox(const livox_ros_driver::CustomMsg::ConstPtr& msg) const;
    void PushScan(const PointCloud::Ptr& scan);

    void Savetrajectory();
    void Finish();

   public:
    std::shared_ptr<RespleParam> param;
    std::string output_dir;

   private:
    void AddScanToBuffer(const PointCloud::Ptr& scan);
    // Voxel-centroid downsampling of a whole scan, ds_scan_voxel leaf.
    void DownsampleScan(const PointCloud::Ptr& scan) const;
    bool CollectMeasures(const std::int64_t& meas_end);
    void PostUpdate();
    void PublishROSMsg();
    void PublishOdometry(const Eigen::Isometry3d& pub_pose, std::int64_t time_ns);
    void PublishFrameWorld();
    void PublishFrameEffectWorld();

    std::shared_ptr<RESPLE_LIO> resple_lio_ = nullptr;
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;

    ros::Publisher pub_laser_cloud_world_;
    ros::Publisher pub_laser_cloud_effect_world_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;

    std::mutex mtx_buffer_;
    std::deque<Point> points_buffer_;
    std::deque<Imu> imu_buffer_;

    MeasureGroup measures_;
    Trajectory trajectory_;
    rosbag::Bag bag_;

    bool if_inertial_initialized = false;
    bool if_localmap_initialized = false;
};

}  // namespace kuma_lio

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

#include "pointcloud_preprocess.h"

#include <glog/logging.h>
#include <execution>

namespace kuma_lio {
PointCloud::Ptr PointCloudPreprocess::LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg,
                                                   std::int64_t scan_start) {
    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(msg->point_num);
    for (int i = 0; i < msg->point_num; i++) {
        if ((msg->points[i].tag & param.livox_tag_mask) != 0) continue;

        if (i % param.point_filter_num != 0) continue;
        double range = msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y +
                       msg->points[i].z * msg->points[i].z;
        if (range < (param.blind * param.blind) || range > (param.det_range_ * param.det_range_))
            continue;

        kuma_lio::Point added_pt;
        added_pt.x = msg->points[i].x;
        added_pt.y = msg->points[i].y;
        added_pt.z = msg->points[i].z;
        added_pt.intensity = msg->points[i].reflectivity;
        // unit of offset_time: nanosecond.  It is a uint32 holding a signed value
        added_pt.SetTimeNs(scan_start + static_cast<std::int32_t>(msg->points[i].offset_time));

        cloud_out->points.push_back(added_pt);
    }
    return cloud_out;
}

PointCloud::Ptr PointCloudPreprocess::OusterHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, std::int64_t scan_start) {
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % param.point_filter_num != 0) continue;
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < (param.blind * param.blind) || range > (param.det_range_ * param.det_range_))
            continue;

        kuma_lio::Point added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // unit of offset_time: nanosecond
        added_pt.SetTimeNs(scan_start + pl_orig.points[i].t);
        cloud_out->points.push_back(added_pt);
    }
    return cloud_out;
}

PointCloud::Ptr PointCloudPreprocess::VelodynePointsHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, std::int64_t scan_start) {
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % param.point_filter_num != 0) continue;
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < (param.blind * param.blind) || range > (param.det_range_ * param.det_range_))
            continue;
        kuma_lio::Point added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // unit of offset_time: nanosecond
        added_pt.SetTimeNs(scan_start + S_TO_NS(pl_orig.points[i].time));
        cloud_out->points.push_back(added_pt);
    }
    return cloud_out;
}
PointCloud::Ptr PointCloudPreprocess::HesaiHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, std::int64_t scan_start) {
    pcl::PointCloud<hesai_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
    double base_time = msg->header.stamp.toSec();
    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % param.point_filter_num != 0) continue;
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (param.blind * param.blind) || range > (param.det_range_ * param.det_range_))
            continue;

        kuma_lio::Point added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        std::int64_t scan_start_offset = S_TO_NS(pl_orig.points[i].timestamp - base_time);
        added_pt.SetTimeNs(scan_start + scan_start_offset);
        cloud_out->push_back(added_pt);
    }
    return cloud_out;
}
}  // namespace kuma_lio

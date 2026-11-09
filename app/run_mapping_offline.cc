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

#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <unistd.h>
#include <csignal>

#include <boost/filesystem.hpp>
#include "laser_mapping.h"

namespace fs = boost::filesystem;

DEFINE_string(config_file, "./config/avia.yaml", "path to config file");
DEFINE_string(bag_file, "", "path to the ros bag");
DEFINE_string(output_dir, "", "save the result to the dir");
DEFINE_double(start, 0.0, "start time in seconds from beginning of bag");
DEFINE_double(duration, -1.0, "duration in seconds, -1 means till end");
DEFINE_bool(dynamic_init, false, "try the dynamic (moving-start) initialization before the stationary one");
DEFINE_string(camera_fname, "", "camera yaml like config/i2nav_robot_camera.yaml: extrin_R, extrin_t (T_imu_cam) and camera (cameras.txt lines); empty = no camera");
DEFINE_string(gnss_obs_fname, "", "");
DEFINE_string(nav_fname, "", "");
bool if_exit = false;
void SigHandle(int sig) {
    if_exit = true;
    LOG(INFO) << "Catch Kill Signal";
}
int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const std::string bag_file = FLAGS_bag_file;
    const std::string config_file = FLAGS_config_file;
    const std::string output_dir = FLAGS_output_dir;
    const std::string camera_fname = FLAGS_camera_fname;
    const std::string gnss_obs_fname = FLAGS_gnss_obs_fname;
    const std::string nav_fname = FLAGS_nav_fname;
    fs::create_directories(output_dir);
    // glog
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // ros
    ros::init(argc, argv, "faster_lio");
    ros::NodeHandle nh;

    // laser mapping
    auto laser_mapping = std::make_shared<kuma_lio::LaserMapping>();
    if (!laser_mapping->Init(config_file,camera_fname)) {
        LOG(ERROR) << "laser mapping init failed.";
        return -1;
    }
    laser_mapping->output_dir = output_dir;
    laser_mapping->dynamic_init_ = FLAGS_dynamic_init;
    laser_mapping->InitialPublishers(nh);

    // handle ctrl-c
    signal(SIGINT, SigHandle);

    // read the bag
    LOG(INFO) << "Opening rosbag, be patient";
    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    if (!bag.isOpen()) {
        LOG(ERROR) << "Failed to open bag";
        return -1;
    }
    rosbag::View full_view(bag);
    ros::Time bag_start = full_view.getBeginTime();
    ros::Time bag_end = full_view.getEndTime();
    ros::Time start_time = bag_start + ros::Duration(FLAGS_start);

    ros::Time end_time;
    if (FLAGS_duration < 0) {
        end_time = bag_end;
    } else {
        end_time = start_time + ros::Duration(FLAGS_duration);
        if (end_time > bag_end) {
            end_time = bag_end;
        }
    }
    LOG(INFO) << "Go!";
    rosbag::View view(bag, start_time, end_time);
    for (const rosbag::MessageInstance &m : view) {
        auto imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (m.getTopic() == laser_mapping->param->imu_topic_ && imu_msg) {
            double start_offset_time = imu_msg->header.stamp.toSec();
            CHECK(std::isnan(laser_mapping->global_offset_time));
            laser_mapping->global_offset_time = start_offset_time;
            LOG(INFO) << "Offset time: " << std::fixed << start_offset_time;
            break;
        }
    }

    if (!nav_fname.empty() && !gnss_obs_fname.empty()) {
        fs::path sol_fname = fs::path(output_dir) / "rtkpos.txt";
        laser_mapping->ReadObsNav(gnss_obs_fname, nav_fname, sol_fname.string());
    }
    const auto run_front_end = [&laser_mapping]() {
        if (laser_mapping->param->front_end == "fastlio") {
            laser_mapping->RunFastLio();
        } else {
            laser_mapping->RunRESPLE();
        }
    };
    for (const rosbag::MessageInstance &m : view) {
        auto livox_msg = m.instantiate<livox_ros_driver::CustomMsg>();
        if (m.getTopic() == laser_mapping->param->lidar_topic_ && livox_msg) {
            laser_mapping->LivoxPCLCallBack(livox_msg);
            run_front_end();
            continue;
        }

        auto point_cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == laser_mapping->param->lidar_topic_ && point_cloud_msg) {
            laser_mapping->StandardPCLCallBack(point_cloud_msg);
            run_front_end();
            continue;
        }

        auto imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (m.getTopic() == laser_mapping->param->imu_topic_ && imu_msg) {
            laser_mapping->IMUCallBack(imu_msg);
            continue;
        }

        if (m.getTopic() == laser_mapping->param->img_topic_) {
            if (auto compressed = m.instantiate<sensor_msgs::CompressedImage>()) {
                laser_mapping->CompressedImageCallback(compressed);
            } else if (auto image = m.instantiate<sensor_msgs::Image>()) {
                laser_mapping->ImageCallBack(image);
            }
            continue;
        }

        auto gnss_nav_msg = m.instantiate<sensor_msgs::NavSatFix>();
        if (laser_mapping->param->gnss_enable_ && m.getTopic() == laser_mapping->param->gnss_topic_ && gnss_nav_msg) {
            laser_mapping->GnssSolutionCallback(gnss_nav_msg);
            continue;
        }
        if (if_exit) {
            break;
        }
    }

    LOG(INFO) << "Finish bag iteration";
    laser_mapping->Finish();
    laser_mapping->Savetrajectory();

    return 0;
}

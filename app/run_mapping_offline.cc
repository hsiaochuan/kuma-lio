//
// Created by xiang on 2021/10/9.
//

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
DEFINE_string(prior_map_file, "", "path to the prior map file");
DEFINE_string(output_dir, "", "save the result to the dir");
DEFINE_double(start, 0.0, "start time in seconds from beginning of bag");
DEFINE_double(duration, -1.0, "duration in seconds, -1 means till end");
DEFINE_string(prior_map_fname, "", "path to prior map file");
DEFINE_string(prior_init_pose, "", "");
bool if_exit = false;
void SigHandle(int sig) {
    if_exit = true;
    LOG(INFO) << "Catch Kill Signal";
}
void StringToPose(const std::string &str, Vec3 &pos, Eigen::Quaterniond &q) {
    sscanf(str.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &pos[0], &pos[1], &pos[2], &q.x(), &q.y(), &q.z(), &q.w());
}
int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const std::string bag_file = FLAGS_bag_file;
    const std::string config_file = FLAGS_config_file;
    const std::string output_dir = FLAGS_output_dir;
    const std::string prior_map_file = FLAGS_prior_map_file;

    // glog
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // ros
    ros::init(argc, argv, "faster_lio");
    ros::NodeHandle nh;

    // laser mapping
    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    if (!laser_mapping->Init(FLAGS_config_file)) {
        LOG(ERROR) << "laser mapping init failed.";
        return -1;
    }
    laser_mapping->output_dir = output_dir;
    laser_mapping->InitialPublishers(nh);

    Eigen::Vector3d prior_init_position(0, 0, 0);
    Eigen::Quaterniond prior_init_rotation(1, 0, 0, 0);
    if (!FLAGS_prior_init_pose.empty()) {
        StringToPose(FLAGS_prior_init_pose, prior_init_position, prior_init_rotation);
        laser_mapping->prior_init_rotation_ = prior_init_rotation;
        laser_mapping->prior_init_position_ = prior_init_position;
        LOG(INFO) << "prior init position: " << prior_init_position.transpose();
        LOG(INFO) << "prior init rotation: " << prior_init_rotation.coeffs().transpose();
    }

    /// handle ctrl-c
    signal(SIGINT, SigHandle);

    // read the bag
    LOG(INFO) << "Opening rosbag, be patient";
    rosbag::Bag bag(FLAGS_bag_file, rosbag::bagmode::Read);

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
        auto livox_msg = m.instantiate<livox_ros_driver::CustomMsg>();
        if (m.getTopic() == laser_mapping->param->lidar_topic_ && livox_msg) {
            laser_mapping->LivoxPCLCallBack(livox_msg);
            laser_mapping->Run();

            continue;
        }

        auto point_cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == laser_mapping->param->lidar_topic_ && point_cloud_msg) {
            laser_mapping->StandardPCLCallBack(point_cloud_msg);
            laser_mapping->Run();

            continue;
        }

        auto vel_msg = m.instantiate<velodyne_msgs::VelodyneScan>();
        if (m.getTopic() == laser_mapping->param->lidar_topic_ && vel_msg) {
            laser_mapping->VelodyneScanCallBack(vel_msg);
            laser_mapping->Run();

            continue;
        }
        auto imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (m.getTopic() == laser_mapping->param->imu_topic_ && imu_msg) {
            static int once = [&]() {
                double start_offset_time = imu_msg->header.stamp.toSec();
                if (std::isnan(laser_mapping->global_offset_time)) {
                    laser_mapping->global_offset_time = start_offset_time;
                    LOG(INFO) << "Offset time: " << std::fixed << start_offset_time;
                }
                return 0;
            }();
            laser_mapping->IMUCallBack(imu_msg);
            continue;
        }

        auto img_msg = m.instantiate<sensor_msgs::Image>();
        if (laser_mapping->param->camera_enable_ && m.getTopic() == laser_mapping->param->camera_topic_ && img_msg) {
            laser_mapping->ImageMsgCallBack(img_msg);
            continue;
        }
        auto compress_img = m.instantiate<sensor_msgs::CompressedImage>();
        if (laser_mapping->param->camera_enable_ && m.getTopic() == laser_mapping->param->camera_topic_ &&
            compress_img) {
            laser_mapping->CompressedImageCallBack(compress_img);
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

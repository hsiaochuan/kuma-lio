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
#include "utils.h"
namespace fs = boost::filesystem;

DEFINE_string(config_file, "./config/avia.yaml", "path to config file");
DEFINE_string(bag_file, "", "path to the ros bag");
DEFINE_string(output_dir, "", "save the result to the dir");
DEFINE_double(start, 0.0, "start time in seconds from beginning of bag");
DEFINE_double(duration, -1.0, "duration in seconds, -1 means till end");
void SigHandle(int sig) {
    faster_lio::options::FLAG_EXIT = true;
    ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    const std::string bag_file = FLAGS_bag_file;
    const std::string config_file = FLAGS_config_file;
    const std::string output_dir = FLAGS_output_dir;
    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    laser_mapping->output_dir = output_dir;
    if (!laser_mapping->InitWithoutROS(FLAGS_config_file)) {
        LOG(ERROR) << "laser mapping init failed.";
        return -1;
    }

    /// handle ctrl-c
    signal(SIGINT, SigHandle);

    // just read the bag and send the data
    LOG(INFO) << "Opening rosbag, be patient";
    rosbag::Bag bag(FLAGS_bag_file, rosbag::bagmode::Read);

    rosbag::View full_view(bag);
    ros::Time bag_start = full_view.getBeginTime();
    ros::Time bag_end   = full_view.getEndTime();
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
        if (m.getTopic() == laser_mapping->lidar_topic_ && livox_msg) {
            faster_lio::Timer::Evaluate(
                [&laser_mapping, &livox_msg]() {
                    laser_mapping->LivoxPCLCallBack(livox_msg);
                    laser_mapping->Run();
                },
                "Laser Mapping Single Run");
            continue;
        }

        auto point_cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == laser_mapping->lidar_topic_ && point_cloud_msg) {
            faster_lio::Timer::Evaluate(
                [&laser_mapping, &point_cloud_msg]() {
                    laser_mapping->StandardPCLCallBack(point_cloud_msg);
                    laser_mapping->Run();
                },
                "Laser Mapping Single Run");
            continue;
        }

        auto imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (m.getTopic() == laser_mapping->imu_topic_ && imu_msg) {
            laser_mapping->IMUCallBack(imu_msg);
            continue;
        }

        auto img_msg = m.instantiate<sensor_msgs::Image>();
        if (laser_mapping->camera_enable_ && m.getTopic() == laser_mapping->camera_topic_ && img_msg) {
            laser_mapping->ImageMsgCallBack(img_msg);
            continue;
        }
        auto compress_img = m.instantiate<sensor_msgs::CompressedImage>();
        if (laser_mapping->camera_enable_ && m.getTopic() == laser_mapping->camera_topic_ && compress_img) {
            laser_mapping->CompressedImageCallBack(compress_img);
            continue;
        }
        if (faster_lio::options::FLAG_EXIT) {
            break;
        }
    }

    LOG(INFO) << "finishing mapping";
    laser_mapping->Finish();

    /// print the fps
    double fps = 1.0 / (faster_lio::Timer::GetMeanTime("Laser Mapping Single Run") / 1000.);
    LOG(INFO) << "Faster LIO average FPS: " << fps;

    std::string traj_fname = output_dir + "/traj_log.txt";
    LOG(INFO) << "save trajectory to: " << traj_fname;
    laser_mapping->Savetrajectory(traj_fname);

    faster_lio::Timer::PrintAll();
    faster_lio::Timer::DumpIntoFile(output_dir + "/time_log.txt");

    return 0;
}

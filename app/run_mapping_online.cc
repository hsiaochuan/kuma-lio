//
// Created by xiang on 2021/10/8.
//
#include <gflags/gflags.h>
#include <unistd.h>
#include <csignal>
#include "utils.h"
#include "laser_mapping.h"

/// run the lidar mapping in online mode

DEFINE_string(output_dir, "./Log/traj.txt", "path to traj log file");
DEFINE_string(config_fname, "./config/avia.yaml", "path to config file");
DEFINE_string(prior_map_fname, "", "path to prior map file");
DEFINE_string(prior_init_pose, "", "");
void SigHandle(int sig) {
    faster_lio::options::FLAG_EXIT = true;
    ROS_WARN("catch sig %d", sig);
}
void StringToPose(const std::string &str, Vec3 & pos, Eigen::Quaterniond& q) {
    sscanf(str.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &pos[0], &pos[1], &pos[2], &q.x(), &q.y(), &q.z(), &q.w());
}

int main(int argc, char **argv) {
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    ros::init(argc, argv, "faster_lio");
    ros::NodeHandle nh;

    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    Eigen::Vector3d prior_init_position(0, 0, 0);
    Eigen::Quaterniond prior_init_rotation(1, 0, 0, 0);
    if (!FLAGS_prior_init_pose.empty()) {
        StringToPose(FLAGS_prior_init_pose, prior_init_position, prior_init_rotation);
        laser_mapping->prior_init_rotation_ = prior_init_rotation;
        laser_mapping->prior_init_position_ = prior_init_position;
        LOG(INFO) << "prior init position: " << prior_init_position.transpose();
        LOG(INFO) << "prior init rotation: " << prior_init_rotation.coeffs().transpose();
    }
    laser_mapping->output_dir = FLAGS_output_dir;
    laser_mapping->Init(FLAGS_config_fname);
    if (!FLAGS_prior_map_fname.empty()) {
        laser_mapping->param->localization_enable_ = true;
    }
    laser_mapping->LoadPriorMap(FLAGS_prior_map_fname);
    laser_mapping->SubAndPubToROS(nh);
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);

    // online, almost same with offline, just receive the messages from ros
    while (ros::ok()) {
        if (faster_lio::options::FLAG_EXIT) {
            break;
        }
        ros::spinOnce();
        laser_mapping->Run();
        rate.sleep();
    }


    laser_mapping->Finish();
    faster_lio::Timer::PrintAll();
    laser_mapping->Savetrajectory(FLAGS_output_dir + "/traj_log.txt");

    return 0;
}

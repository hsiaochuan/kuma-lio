#include "laser_mapping.h"
#include <memory>
#include "global_optimizor.h"
#include "ieskf_lio.h"
#include "utils.h"

namespace faster_lio {

LaserMapping::LaserMapping() = default;

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    if (!param->LoadFromYaml(config_fname))
        return false;
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->blind_ = param->blind;
    preprocess_->point_filter_num_ = param->point_filter_num;
    preprocess_->lidar_type_ = LidarTypeFromString(param->lidar_type);
    preprocess_->max_range = param->det_range_;

    // the estimation algorithm
    odom_ = std::make_shared<IeskfLio>();
    odom_->SetPriorInitPose(prior_init_position_, prior_init_rotation_);
    if (!odom_->Init(param))
        return false;

    scan_sampler_.setLeafSize(param->scan_filter_size, param->scan_filter_size, param->scan_filter_size);

    mapper = std::make_shared<GlobalOptimizor>();
    GlobalOptimizor::Options global_options;
    global_options.LoadFromYaml(config_fname);
    mapper->options_ = global_options;
    mapper->output_dir = output_dir;

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}
void LaserMapping::LoadPriorMap(const std::string &prior_map_fname) {
    odom_->LoadPriorMap(prior_map_fname);
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    if (preprocess_->lidar_type_ == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else if (preprocess_->lidar_type_ == LidarType::VELODYNE_SCAN) {
        sub_pcl_ = nh.subscribe<velodyne_msgs::VelodyneScan>(
            param->lidar_topic_, 200000, [this](const velodyne_msgs::VelodyneScan::ConstPtr &msg) { VelodyneScanCallBack(msg); });
    }else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });
    sub_img_ = nh.subscribe<sensor_msgs::Image>(param->camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr &msg) {
        ImageMsgCallBack(msg);
    });

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    if (param->camera_enable_)
        pub_image_ = nh.advertise<sensor_msgs::Image>("/image_raw", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
    if (param->localization_enable_) {
        auto map_cloud = odom_->PriorMap();
        LOG(INFO) << "publish prior map " << map_cloud->size() << " points";
        auto pub_prior_map = nh.advertise<sensor_msgs::PointCloud2>("/prior_map", 100000);
        sensor_msgs::PointCloud2 prior_map_msg;
        pcl::toROSMsg(*map_cloud, prior_map_msg);
        prior_map_msg.header.frame_id = "world";
        prior_map_msg.header.stamp = ros::Time::now();
        while (1) {
            if (pub_prior_map.getNumSubscribers() > 0) {
                for (int i = 0; i < 3; ++i) {
                    pub_prior_map.publish(prior_map_msg);
                    LOG(INFO) << "publish prior map";
                    sleep(1);
                }
                break;
            }
        }
    }
}

}  // namespace faster_lio

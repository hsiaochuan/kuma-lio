#include "laser_mapping.h"
#include <yaml-cpp/yaml.h>

#include <memory>
#include "cameras/cameras.h"
#include "global_optimizor.h"
#include "utils.h"

namespace faster_lio {

LaserMapping::LaserMapping() {
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    p_imu_ = std::make_shared<ImuProcess>();
}

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    if (!param->LoadFromYaml(config_fname))
        return false;
    preprocess_->SetLidarType(LidarTypeFromString(param->lidar_type));
    preprocess_->Blind() = param->blind;
    preprocess_->PointFilterNum() = param->point_filter_num;
    if (param->ivox_nearby_type == 0) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (param->ivox_nearby_type == 6) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (param->ivox_nearby_type == 18) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (param->ivox_nearby_type == 26) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    scan_sampler_.setLeafSize(param->scan_filter_size, param->scan_filter_size, param->scan_filter_size);

    p_imu_->SetExtrinsic(param->extrin_il_.Trans(), param->extrin_il_.Mat3d());
    p_imu_->SetGyrCov(Vec3(param->gyr_cov, param->gyr_cov, param->gyr_cov));
    p_imu_->SetAccCov(Vec3(param->acc_cov, param->acc_cov, param->acc_cov));
    p_imu_->SetGyrBiasCov(Vec3(param->b_gyr_cov, param->b_gyr_cov, param->b_gyr_cov));
    p_imu_->SetAccBiasCov(Vec3(param->b_acc_cov, param->b_acc_cov, param->b_acc_cov));

    ivox_ = std::make_shared<IVoxType>(param->ivox_options_);
    mapper = std::make_shared<GlobalOptimizor>();
    GlobalOptimizor::Options global_options;
    global_options.LoadFromYaml(config_fname);
    mapper->options_ = global_options;
    mapper->output_dir = output_dir;

    if (param->image_save_en_ && param->camera_) {
        camera_t cam_id = 1;
        sfm_data_.cameras_[cam_id] = param->camera_;
    }

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        param->max_iteraions, epsi.data());

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using phc ivox";
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using default ivox";
    }

    return true;
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    if (preprocess_->GetLidarType() == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });

    sub_img_ = nh.subscribe<sensor_msgs::Image>(param->camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr &msg) {
        ImageMsgCallBack(msg);
    });
    // ROS publisher init
    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "world";

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
}

}  // namespace faster_lio


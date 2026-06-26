#include "laser_mapping.h"
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

#include <memory>
#include "cameras/cameras.h"
#include "global_optimizor.h"
#include "utils.h"

namespace faster_lio {

LaserMapping::LaserMapping() {
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    p_imu_ = std::make_shared<ImuProcess>();
    state_point_ = std::make_shared<StatePoint>();
    visual_manager = std::make_shared<VisualManager>();

}

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    if (!param->LoadFromYaml(config_fname))
        return false;
    preprocess_->blind_ = param->blind;
    preprocess_->point_filter_num_ = param->point_filter_num;
    preprocess_->lidar_type_ = LidarTypeFromString(param->lidar_type);
    preprocess_->max_range = param->det_range_;
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

    ivox_ = std::make_shared<IVoxType>(param->ivox_options_);

    scan_sampler_.setLeafSize(param->scan_filter_size, param->scan_filter_size, param->scan_filter_size);

    p_imu_->cov_gyr_ = Vec3(param->gyr_cov, param->gyr_cov, param->gyr_cov);
    p_imu_->cov_acc_ = Vec3(param->acc_cov, param->acc_cov, param->acc_cov);
    p_imu_->cov_bias_gyr_ = Vec3(param->b_gyr_cov, param->b_gyr_cov, param->b_gyr_cov);
    p_imu_->cov_bias_acc_ = Vec3(param->b_acc_cov, param->b_acc_cov, param->b_acc_cov);
    p_imu_->state_point_ = state_point_;

    if (param->camera_enable_) {
        visual_manager->state_point_ = state_point_;
        visual_manager->ivox_ = ivox_;
        visual_manager->param = param;
        visual_manager->Initialize();
    }

    mapper = std::make_shared<GlobalOptimizor>();
    GlobalOptimizor::Options global_options;
    global_options.LoadFromYaml(config_fname);
    mapper->options_ = global_options;
    mapper->output_dir = output_dir;

    if (param->image_save_en_ && param->camera_) {
        camera_t cam_id = 1;
        sfm_data_.cameras_[cam_id] = param->camera_;
    }

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}
void LaserMapping::LoadPriorMap(const std::string &prior_map_fname) {
    if (!param->localization_enable_)
        return;
    map_cloud_.reset(new pcl::PointCloud<PriorMapPoint>());
    // load points
    pcl::io::loadPCDFile(prior_map_fname, *map_cloud_);
    if (map_cloud_->size() == 0) {
        LOG(WARNING) << "no prior map found";
        return;
    }
    // downsample
    if (param->map_filter_size_ > 0) {
        pcl::VoxelGrid<PriorMapPoint> voxel_grid;
        voxel_grid.setLeafSize(param->map_filter_size_, param->map_filter_size_, param->map_filter_size_);
        voxel_grid.setInputCloud(map_cloud_);
        pcl::PointCloud<PriorMapPoint>::Ptr tmp_cloud(new pcl::PointCloud<PriorMapPoint>());
        voxel_grid.filter(*tmp_cloud);
        map_cloud_ = tmp_cloud;
    }

    // estimate normal
    pcl::NormalEstimation<PriorMapPoint, pcl::Normal> ne;
    ne.setInputCloud(map_cloud_);
    pcl::search::KdTree<PriorMapPoint>::Ptr tree(new pcl::search::KdTree<PriorMapPoint>());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(20);
    ne.compute(*cloud_normals);

    map_normals_.resize(map_cloud_->size());
    for (size_t i = 0; i < map_cloud_->size(); ++i) {
        map_normals_[i] = cloud_normals->points[i].getNormalVector3fMap();
    }

    // build kd tree
    map_kd_tree_.reset(new pcl::KdTreeFLANN<PriorMapPoint>());
    map_kd_tree_->setInputCloud(map_cloud_);
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
}

}  // namespace faster_lio


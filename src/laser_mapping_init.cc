#include "laser_mapping.h"
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
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

    pcl::PCLPointCloud2 cloud_blob;
    if (pcl::io::loadPCDFile(prior_map_fname, cloud_blob) < 0) {
        LOG(WARNING) << "failed to load prior map: " << prior_map_fname;
        return;
    }
    bool has_normals = false;
    for (const auto &field : cloud_blob.fields) {
        if (field.name == "normal_x") {
            has_normals = true;
            break;
        }
    }
    map_cloud_.reset(new pcl::PointCloud<PriorMapPoint>());

    if (has_normals) {
        LOG(INFO) << "prior map already contains normals, skip normal estimation";
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        pcl::fromPCLPointCloud2(cloud_blob, *cloud_with_normals);
        if (cloud_with_normals->empty()) {
            LOG(WARNING) << "no prior map found";
            return;
        }
        pcl::copyPointCloud(*cloud_with_normals, *map_cloud_);
        LOG(INFO) << "prior map loaded, size: " << map_cloud_->size();
        map_normals_.resize(cloud_with_normals->size());
        for (size_t i = 0; i < cloud_with_normals->size(); ++i) {
            map_normals_[i] = cloud_with_normals->points[i].getNormalVector3fMap();
        }
    } else {
        pcl::fromPCLPointCloud2(cloud_blob, *map_cloud_);
        if (map_cloud_->empty()) {
            LOG(WARNING) << "no prior map found";
            return;
        }
        LOG(INFO) << "prior map loaded, size: " << map_cloud_->size();
        // downsample
        double leaf_size = -0.1;
        if (leaf_size > 0) {
            pcl::VoxelGrid<PriorMapPoint> voxel_grid;
            voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
            voxel_grid.setInputCloud(map_cloud_);
            pcl::PointCloud<PriorMapPoint>::Ptr tmp_cloud(new pcl::PointCloud<PriorMapPoint>());
            voxel_grid.filter(*tmp_cloud);
            map_cloud_ = tmp_cloud;
        }
        LOG(INFO) << "prior map downsampled, size: " << map_cloud_->size();
        // estimate normal
        pcl::NormalEstimationOMP<PriorMapPoint, pcl::Normal> ne;
        ne.setNumberOfThreads(8);
        ne.setInputCloud(map_cloud_);
        pcl::search::KdTree<PriorMapPoint>::Ptr tree(new pcl::search::KdTree<PriorMapPoint>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(12);
        ne.compute(*cloud_normals);
        LOG(INFO) << "prior map normals estimated";
        map_normals_.resize(map_cloud_->size());
        for (size_t i = 0; i < map_cloud_->size(); ++i) {
            map_normals_[i] = cloud_normals->points[i].getNormalVector3fMap();
        }
    }

    // build kd tree
    map_kd_tree_.reset(new pcl::KdTreeFLANN<PriorMapPoint>());
    map_kd_tree_->setInputCloud(map_cloud_);
    LOG(INFO) << "prior map kd tree built";
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
        LOG(INFO) << "publish prior map " << map_cloud_->size() << " points";
        auto pub_prior_map = nh.advertise<sensor_msgs::PointCloud2>("/prior_map", 100000);
        sensor_msgs::PointCloud2 prior_map_msg;
        pcl::toROSMsg(*map_cloud_, prior_map_msg);
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


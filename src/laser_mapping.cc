#include <iomanip>
#include <sstream>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include "laser_mapping.h"
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

namespace fs = boost::filesystem;
namespace faster_lio {
LaserMapping::LaserMapping() = default;

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    if (!param->LoadFromYaml(config_fname)) return false;
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->param = param;

    // the estimation algorithm
    resple_lio_ = std::make_shared<RESPLE_LIO>();
    resple_lio_->param = param;
    resple_lio_->Init();

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}

void LaserMapping::InitialSubscribers(ros::NodeHandle &nh) {
    LidarType lidar_type = LidarTypeFromString(param->lidar_type);
    if (lidar_type == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000,
            [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else if (lidar_type == LidarType::VELODYNE_SCAN) {
        sub_pcl_ = nh.subscribe<velodyne_msgs::VelodyneScan>(
            param->lidar_topic_, 200000,
            [this](const velodyne_msgs::VelodyneScan::ConstPtr &msg) { VelodyneScanCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000,
            [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });
    bool compressed_image = false;
    if (!compressed_image) {
        sub_img_ = nh.subscribe<sensor_msgs::Image>(
            param->camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr &msg) { ImageMsgCallBack(msg); });
    } else {
        sub_img_ = nh.subscribe<sensor_msgs::CompressedImage>(
            param->camera_topic_, 200000,
            [this](const sensor_msgs::CompressedImage::ConstPtr &msg) { CompressedImageCallBack(msg); });
    }
}
void LaserMapping::InitialPublishers(ros::NodeHandle &nh) {
    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    if (param->camera_enable_)
        pub_image_ = nh.advertise<sensor_msgs::Image>("/image_raw", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
}

void LaserMapping::Run() {
    static constexpr int kInitImuNum = 20;
    static constexpr int64_t kInitMapSpanNs = S_TO_NS(0.3);
    if (!resple_lio_->if_localmap_initialized) {
        if (points_buffer_.empty()) return;

        std::int64_t first_point_time_ns = points_buffer_.front().GetTimeNs();

        if (!resple_lio_->if_inertial_initialized) {
            if (static_cast<int>(imu_buffer_.size()) < kInitImuNum) return;
            Vec3 acc_sum = Vec3::Zero();
            Vec3 acc_gyro = Vec3::Zero();
            for (int i = 0; i < kInitImuNum; ++i) {
                acc_sum += imu_buffer_[i].linear_acceleration;
                acc_gyro += imu_buffer_[i].angular_velocity;
            }

            Vec3 mean_acc = acc_sum / kInitImuNum;
            Vec3 gravity_body = mean_acc.normalized() * RESPLE_LIO::GRAVITY_NORM;
            Eigen::Matrix3d R0 =
                Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
            double yaw = std::atan2(R0(1, 0), R0(0, 0));
            R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;
            Eigen::Quaterniond q0(R0);
            resple_lio_->bg = acc_gyro / kInitImuNum;
            resple_lio_->gravity = q0 * gravity_body;
            resple_lio_->spl.init(param->dt_ns, 0, q0);
            resple_lio_->if_inertial_initialized = true;
            LOG(INFO)  << " s, gravity " << resple_lio_->gravity.transpose();
        }


        if (points_buffer_.back().GetTimeNs() < first_point_time_ns + kInitMapSpanNs) return;
        std::int64_t end_time_ns = first_point_time_ns + kInitMapSpanNs;
        PointCloud::Ptr init_points(new PointCloud);
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < end_time_ns) {
            init_points->emplace_back(points_buffer_.front());
            points_buffer_.pop_front();
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < end_time_ns) {
            imu_buffer_.pop_front();
        }
        while (resple_lio_->spl.maxTimeNs() < end_time_ns) {
            resple_lio_->spl.addOneStateKnot(Vec3::Zero(),Vec3::Zero());
        }
        if (init_points->size() < 100) {
            LOG(WARNING) << "only " << init_points->size() << " points for map init, waiting";
            return;
        }
        PointCloud::Ptr init_points_world(new PointCloud);
        Pose3 world_from_lidar(resple_lio_->spl.q_idle, Vec3::Zero());
        world_from_lidar = world_from_lidar * param->extrin_il_;
        pcl::transformPointCloud(*init_points, *init_points_world, world_from_lidar.Mat4d(), true);
        resple_lio_->kd_tree->Build(init_points_world->points);
        resple_lio_->if_localmap_initialized = true;
        LOG(INFO) << "localmap initialized with " << init_points_world->size() << " points";
        return;
    }

    while (CollectMeasures(resple_lio_->spl.maxTimeNs() + param->dt_ns)) {
        resple_lio_->propRCP();
        resple_lio_->ProcessMeasurement(measures_);
        PublishROSMsg();
        PostUpdate();
    }

}

void LaserMapping::PostUpdate() {
    int64_t t_ns = resple_lio_->spl.maxTimeNs() - param->dt_ns;
    Eigen::Isometry3d body_pose = resple_lio_->spl.Interpolate(t_ns);
    trajectory_.emplace_back(global_offset_time + t_ns * 1e-9, body_pose);

    if (param->pcd_save_en_) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "map");
            return 0;
        }();
        wait_save_points_->reserve(wait_save_points_->size() + resple_lio_->pt_meas.size());
        for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
            Point point{};
            point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
            point.intensity = resple_lio_->pt_meas[i].point.intensity;
            point.SetTimeNs(resple_lio_->pt_meas[i].time_ns);
            wait_save_points_->push_back(point);
        }

        if (wait_save_points_->size() >= 2 && wait_save_points_->back().GetTimeNs() - wait_save_points_->front().GetTimeNs() > S_TO_NS(param->pcd_save_interval_)) {
            double time_s = global_offset_time + wait_save_points_->front().GetTimeSec();
            std::stringstream stamp_string;
            stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s;

            pcl::UniformSampling<Point> sampler;
            sampler.setInputCloud(wait_save_points_);
            sampler.setRadiusSearch(param->ds_scan_voxel);
            sampler.filter(*wait_save_points_);
            pcl::io::savePCDFileBinary((fs::path(output_dir) / "map" / stamp_string.str()).string(), *wait_save_points_);
            wait_save_points_->clear();
        }
    }

}
void LaserMapping::PublishROSMsg() {
    // publish
    PublishOdometry();
    PublishFrameWorld();
    PublishFrameEffectWorld();
    PublishImage();
    PublishFrustrum();
}
bool LaserMapping::CollectMeasures(const std::int64_t &meas_end) {
    if (points_buffer_.empty()) return false;
    if (param->imu_enable_ && imu_buffer_.empty()) return false;

    // imu buffer is not beyond the measures end
    if (imu_buffer_.back().time_ns < meas_end) return false;
    // points buffer is not beyond the measures end
    if (points_buffer_.back().GetTimeNs() < meas_end) return false;

    // push the imu data
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < meas_end) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    measures_.lidar_->clear();
    int num_collected = 0;
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < meas_end) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
        num_collected++;
    }

    if (measures_.lidar_->empty()) {
        LOG(INFO) << "Empty lidar points in measures";
    }
    return true;
}

}  // namespace faster_lio

#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>
#include <cv_bridge/cv_bridge.h>
#include "laser_mapping.h"

#include <cameras/pinhole_fisheye_camera.h>
#include <cameras/pinhole_camera.h>
#include <cameras/spherial_camera.h>
#include <cameras/pinhole_radial.h>
#include "utils.h"
namespace fs = boost::filesystem;
namespace faster_lio {

bool LaserMapping::InitROS(ros::NodeHandle &nh) {
    LoadParams(nh);
    SubAndPubToROS(nh);

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        max_iteraions, epsi.data());

    return true;
}

bool LaserMapping::InitWithoutROS(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        max_iteraions, epsi.data());

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using phc ivox";
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
        LOG(INFO) << "using default ivox";
    }

    return true;
}

bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
    // get params from param server
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double scan_filter_size;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    try {
        nh.param<bool>("path_save_en", path_save_en_, true);
        nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
        nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
        nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
        nh.param<bool>("publish/scan_effect_pub_en", scan_effect_pub_en_, false);
        nh.param<std::string>("publish/tf_imu_frame", tf_imu_frame_, "body");
        nh.param<std::string>("publish/tf_world_frame", tf_world_frame_, "camera_init");

        nh.param<std::string>("common/lid_topic", lidar_topic_, "/livox/lidar");
        nh.param<std::string>("common/imu_topic", imu_topic_, "/livox/imu");
        nh.param<std::string>("common/camera_topic", camera_topic_, "/livox/imu");
        nh.param<bool>("common/camera_enable", camera_enable_, "/livox/imu");
        nh.param<double>("common/scan_interval", scan_interval_, 0.1);
        nh.param<double>("common/lidar_time_offset", lidar_time_offset_, 0.1);
        nh.param<double>("common/camera_time_offset", camera_time_offset_, 0.1);
        nh.param<int>("common/image_skip", image_skip_, 3);
        nh.param<int>("max_iteration", max_iteraions, 4);
        nh.param<float>("esti_plane_threshold", esti_plane_thr, 0.1);
        nh.param<double>("scan_filter_size", scan_filter_size, 0.5);
        nh.param<double>("map_filter_size", map_filter_size_, 0.0);
        nh.param<double>("cube_side_length", cube_len_, 200);
        nh.param<float>("mapping/det_range", det_range_, 300.f);
        nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
        nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
        nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
        nh.param<int>("preprocess/lidar_type", lidar_type, 1);
        nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
        nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
        nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
        nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
        nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
        nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());

        nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
        nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);
    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }
    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar (livox_ros_driver::CustomMsg)";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = tf_world_frame_;

    scan_sampler_.setLeafSize(scan_filter_size, scan_filter_size, scan_filter_size);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    if (extrinR_.size() == 9) lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);
    else if (extrinR_.size() == 4)
        lidar_R_wrt_IMU = common::QuatFromArray<double>(extrinR_).toRotationMatrix();
    else
        throw "extrinsic should be 9 or 4";
    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double scan_filter_size;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        path_pub_en_ = yaml["publish"]["path_publish_en"].as<bool>();
        scan_pub_en_ = yaml["publish"]["scan_publish_en"].as<bool>();
        dense_pub_en_ = yaml["publish"]["dense_publish_en"].as<bool>();
        scan_body_pub_en_ = yaml["publish"]["scan_bodyframe_pub_en"].as<bool>();
        scan_effect_pub_en_ = yaml["publish"]["scan_effect_pub_en"].as<bool>();
        tf_imu_frame_ = yaml["publish"]["tf_imu_frame"].as<std::string>("body");
        tf_world_frame_ = yaml["publish"]["tf_world_frame"].as<std::string>("camera_init");
        path_save_en_ = yaml["path_save_en"].as<bool>();

        max_iteraions = yaml["max_iteration"].as<int>();
        esti_plane_thr = yaml["esti_plane_threshold"].as<float>();

        scan_filter_size = yaml["scan_filter_size"].as<float>();
        map_filter_size_ = yaml["map_filter_size"].as<float>();
        cube_len_ = yaml["cube_side_length"].as<int>();
        det_range_ = yaml["mapping"]["det_range"].as<float>();
        gyr_cov = yaml["mapping"]["gyr_cov"].as<float>();
        acc_cov = yaml["mapping"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["mapping"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["preprocess"]["blind"].as<double>();
        lidar_type = yaml["preprocess"]["lidar_type"].as<int>();
        preprocess_->PointFilterNum() = yaml["point_filter_num"].as<int>();
        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        extrinT_ = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["ivox_nearby_type"].as<int>();

        lidar_topic_ = yaml["common"]["lid_topic"].as<std::string>();
        imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
        camera_topic_ = yaml["common"]["camera_topic"].as<std::string>();
        scan_interval_ = yaml["common"]["scan_interval"].as<double>();
        camera_enable_ = yaml["common"]["camera_enable"].as<bool>();
        camera_time_offset_ = yaml["common"]["camera_time_offset"].as<double>();
        lidar_time_offset_ = yaml["common"]["lidar_time_offset"].as<double>();
        image_skip_ = yaml["common"]["image_skip"].as<int>();
    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

    if (camera_enable_) {
        std::string camera_type;
        try {
            camera_type = yaml["cam"]["camera_model"].as<std::string>();
        } catch (...) {
            LOG(ERROR) << "fail to load the camera type";
            return false;
        }
        CAMERA_MODEL camera_model = ToCameraModel(camera_type);
        std::vector<double> resolution;
        std::vector<double> distort_param;
        std::vector<double> pinhole_param;
        try {
            resolution = yaml["cam"]["resolution"].as<std::vector<double>>();
            if (IsPinhole(camera_model)) {
                pinhole_param = yaml["cam"]["pinhole_param"].as<std::vector<double>>();
            }
            if (IsDistorted(camera_model)) {
                distort_param = yaml["cam"]["distortion_param"].as<std::vector<double>>();
            }
        } catch (...) {
            LOG(ERROR) << "bad conversion in camera load";
            return false;
        }

        std::vector<double> param;
        param.insert(param.end(),pinhole_param.begin(), pinhole_param.end());
        param.insert(param.end(),distort_param.begin(), distort_param.end());
        bool update_param = false;
        switch (camera_model) {
            case PINHOLE:
                camera_.reset(new PinholeCamera);
                break;
            case PINHOLE_RADIAL:
                camera_.reset(new PinholeRadialCamera);
                update_param = camera_->updateFromParams(param);
                break;
            case PINHOLE_FISHEYE:
                camera_.reset(new PinholeFisheyeCamera);
                update_param = camera_->updateFromParams(param);
                break;
            case SPHERICAL:
                camera_.reset(new PinholeCamera);
                break;
        }
        if (!update_param) {
            LOG(ERROR) << "fail to update the param";
            return false;
        }
        camera_->w_ = static_cast<unsigned int>(resolution[0]);
        camera_->h_ = static_cast<unsigned int>(resolution[1]);
        LOG(INFO) << "camera type: " << camera_model;
    }

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar (livox_ros_driver::CustomMsg)";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    scan_sampler_.setLeafSize(scan_filter_size, scan_filter_size, scan_filter_size);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    if (extrinR_.size() == 9) lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);
    else if (extrinR_.size() == 4)
        lidar_R_wrt_IMU = common::QuatFromArray<double>(extrinR_).toRotationMatrix();
    else
        throw "extrinsic should be 9 or 4";

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    run_in_offline_ = true;
    return true;
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    if (preprocess_->GetLidarType() == LidarType::AVIA) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            lidar_topic_, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            lidar_topic_, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });

    sub_img_ = nh.subscribe<sensor_msgs::Image>(camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr &msg) {
        ImageMsgCallBack(msg);
    });
    // ROS publisher init
    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = tf_world_frame_;

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
}

LaserMapping::LaserMapping() {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
}

void LaserMapping::Run() {
    if (!SyncPackages()) {
        return;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }

    /// the first scan
    if (if_local_map_init_) {
        state_point_ = kf_.get_x();
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            PointBodyToWorld(&scan_undistort_->points[i], &scan_down_world_->points[i]);
        }
        ivox_->AddPoints(scan_down_world_->points);
        if_local_map_init_ = false;
        return;
    }

    /// downsample
    Timer::Evaluate(
        [&, this]() {
            scan_sampler_.setInputCloud(scan_undistort_);
            scan_sampler_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, common::V4F::Zero());

    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            // iterated state estimation
            double solve_H_time = 0;
            // update the observation model, will call nn and point-to-plane residual computation
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            // save the state
            state_point_ = kf_.get_x();
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    // publish or save map pcd
    if (run_in_offline_) {
        if (pcd_save_en_) {
            PublishFrameWorld();
        }
        if (path_save_en_) {
            PublishPath(pub_path_);
        }
    } else {
        if (pub_odom_aft_mapped_) {
            PublishOdometry(pub_odom_aft_mapped_);
        }
        if (path_pub_en_ || path_save_en_) {
            PublishPath(pub_path_);
        }
        if (scan_pub_en_ || pcd_save_en_) {
            PublishFrameWorld();
        }
        if (scan_pub_en_ && scan_body_pub_en_) {
            PublishFrameBody(pub_laser_cloud_body_);
        }
        if (scan_pub_en_ && scan_effect_pub_en_) {
            PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
        }
    }
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            double timestamp = msg->header.stamp.toSec();

            // time offset
            timestamp += lidar_time_offset_;

            // loop
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
            }
            last_timestamp_lidar_ = timestamp;

            // set start offset
            if (if_first_scan_) {
                first_scan_time_ = timestamp;
                if_first_scan_ = false;
            }

            timestamp = timestamp - first_scan_time_;

            // push to buffer
            PointCloud::Ptr scan(new PointCloud());
            preprocess_->Process(msg, scan, timestamp);
            for (int i = 0; i < scan->size(); ++i) {
                points_buffer_.emplace_back(scan->points[i]);
            }
        },
        "Preprocess (Standard)");
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            double timestamp = msg->header.stamp.toSec();

            // time offset
            timestamp += lidar_time_offset_;

            // loop
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
            }
            last_timestamp_lidar_ = timestamp;

            // set start offset
            if (if_first_scan_) {
                first_scan_time_ = timestamp;
                if_first_scan_ = false;
            }

            timestamp = timestamp - first_scan_time_;

            // push to buffer
            PointCloud::Ptr scan(new PointCloud());
            preprocess_->Process(msg, scan, timestamp);
            for (int i = 0; i < scan->size(); ++i) {
                points_buffer_.emplace_back(scan->points[i]);
            }
        },
        "Preprocess (Livox)");
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    double timestamp = msg_in->header.stamp.toSec();

    // loop
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }
    last_timestamp_imu_ = timestamp;

    // set start offset
    if (if_first_scan_) {
        return;
    }else {
        timestamp = timestamp - first_scan_time_;
    }

    // push to buffer
    Imu imu;
    imu.timestamp = timestamp;
    imu.angular_velocity.x() = msg_in->angular_velocity.x;
    imu.angular_velocity.y() = msg_in->angular_velocity.y;
    imu.angular_velocity.z() = msg_in->angular_velocity.z;
    imu.linear_acceleration.x() = msg_in->linear_acceleration.x;
    imu.linear_acceleration.y() = msg_in->linear_acceleration.y;
    imu.linear_acceleration.z() = msg_in->linear_acceleration.z;
    imu_buffer_.emplace_back(imu);
}
void LaserMapping::ImageMsgCallBack(const sensor_msgs::Image::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % image_skip_ == 0) {
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        ImageCallBack(img,msg_in->header.stamp.toSec());
    }
    img_count++;
}

void LaserMapping::ImageCallBack(const cv::Mat& img, double timestamp) {

    // time offset
    timestamp += camera_time_offset_;

    // loop
    if (timestamp < last_timestamp_camera_) {
        LOG(WARNING) << "image loop back, clear buffer";
        img_time_buffer_.clear();
    }
    last_timestamp_camera_ = timestamp;

    // set start offset
    if (if_first_scan_)
        return;
    else
        timestamp = timestamp - first_scan_time_;

    // push to buffer
    img_time_buffer_.emplace_back(timestamp);
}
void LaserMapping::CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % image_skip_ == 0) {
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        ImageCallBack(img,msg_in->header.stamp.toSec());
    }
    img_count++;
}
bool LaserMapping::SyncPackages() {
    if (points_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    if (camera_enable_ && img_time_buffer_.empty())
        return false;

    // set the measure end timestamp
    if (lidar_end_time_ == 0) {
        if (camera_enable_) {
            lidar_end_time_ = img_time_buffer_.front();
            img_time_buffer_.pop_front();
        }else
            lidar_end_time_ = points_buffer_.front().timestamp + scan_interval_;
    } else if (measures_.lidar_end_time_ == lidar_end_time_) {
        // after the update, incre the end time
        if (camera_enable_) {
            lidar_end_time_ = img_time_buffer_.front();
            img_time_buffer_.pop_front();
        } else
            lidar_end_time_ = lidar_end_time_ + scan_interval_;
    } else {
        // the measure is not synced, no need to set the lidar end time
        lidar_end_time_ = lidar_end_time_;
    }

    if (imu_buffer_.back().timestamp < lidar_end_time_) return false;
    if (points_buffer_.back().timestamp < lidar_end_time_) return false;

    // push the imu data
    measures_.imu_.clear();
    while (imu_buffer_.front().timestamp < lidar_end_time_ && !imu_buffer_.empty()) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    // push the lidar points
    measures_.lidar_->clear();
    while (points_buffer_.front().timestamp < lidar_end_time_ && !points_buffer_.empty()) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }

    measures_.lidar_end_time_ = lidar_end_time_;
    if (measures_.lidar_->empty() || measures_.imu_.empty())
        return false;
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
              << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty()) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / map_filter_size_).array().floor() + 0.5) * map_filter_size_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * map_filter_size_ && fabs(dis_2_center.y()) > 0.5 * map_filter_size_ &&
                fabs(dis_2_center.z()) > 0.5 * map_filter_size_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    points_near.clear();
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] =
                            common::esti_plane(plane_coef_[i], points_near, esti_plane_thr);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        point_selected_surf_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                common::V3F point_this_be = corr_pts_[i].head<3>();
                common::M3F point_be_crossmat = Hat(point_this_be);
                common::V3F point_this = off_R * point_this_be + off_t;
                common::M3F point_crossmat = Hat(point_this);

                /*** get the normal vector of closest surface/corner ***/
                common::V3F norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                common::V3F C(Rt * norm_vec);
                common::V3F A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    common::V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }

                /*** Measurement: distance to the closest surface/corner ***/
                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

void LaserMapping::PublishPath(const ros::Publisher pub_path) {
    geometry_msgs::PoseStamped msg_body_pose;
    SetPosestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose.header.frame_id = tf_world_frame_;

    /*** if path is too large, the rvis will crash ***/
    path_.poses.push_back(msg_body_pose);
    if (!run_in_offline_) {
        pub_path.publish(path_);
    }
}

void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped) {
    nav_msgs::Odometry odom_aft_mapped;
    odom_aft_mapped.header.frame_id = tf_world_frame_;
    odom_aft_mapped.child_frame_id = tf_imu_frame_;
    odom_aft_mapped.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    SetPosestamp(odom_aft_mapped.pose);
    pub_odom_aft_mapped.publish(odom_aft_mapped);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odom_aft_mapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odom_aft_mapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odom_aft_mapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odom_aft_mapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odom_aft_mapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped.pose.pose.position.x, odom_aft_mapped.pose.pose.position.y,
                                    odom_aft_mapped.pose.pose.position.z));
    q.setW(odom_aft_mapped.pose.pose.orientation.w);
    q.setX(odom_aft_mapped.pose.pose.orientation.x);
    q.setY(odom_aft_mapped.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped.header.stamp, tf_world_frame_, tf_imu_frame_));
}

void LaserMapping::PublishFrameWorld() {
    if (!(run_in_offline_ == false && scan_pub_en_) && !pcd_save_en_) {
        return;
    }

    PointCloud::Ptr scan_world;
    if (dense_pub_en_) {
        PointCloud::Ptr scan_full(scan_undistort_);
        int size = scan_full->points.size();
        scan_world.reset(new PointCloud(size, 1));
        for (int i = 0; i < size; i++) {
            PointBodyToWorld(&scan_full->points[i], &scan_world->points[i]);
        }
    } else {
        scan_world = scan_down_world_;
    }

    if (run_in_offline_ == false && scan_pub_en_) {
        sensor_msgs::PointCloud2 scan_msg;
        pcl::toROSMsg(*scan_world, scan_msg);
        scan_msg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        scan_msg.header.frame_id = tf_world_frame_;
        pub_laser_cloud_world_.publish(scan_msg);
    }

    if (pcd_save_en_) {
        static auto once = fs::create_directories(output_dir + "/scans");
        std::string pcd_save_fname(output_dir + "/scans/" + std::to_string(measures_.lidar_end_time_) + ".pcd");
        pcl::io::savePCDFileBinary(pcd_save_fname, *scan_undistort_);
    }

    if (pcd_save_en_) {
        *pcl_wait_save_ += *scan_world;
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
            pcd_index_++;
            static auto once = fs::create_directories(output_dir + "/maps");
            scan_sampler_.setInputCloud(pcl_wait_save_);
            scan_sampler_.filter(*pcl_wait_save_);
            std::string pcd_save_fname(output_dir + "/maps/map_" + std::to_string(pcd_index_) + ".pcd");
            LOG(INFO) << "current scan saved to " << pcd_save_fname;
            pcl::io::savePCDFileBinary(pcd_save_fname, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::PublishFrameBody(const ros::Publisher &pub_laser_cloud_body) {
    int size = scan_undistort_->points.size();
    PointCloud::Ptr laser_cloud_imu_body(new PointCloud(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "body";
    pub_laser_cloud_body.publish(laserCloudmsg);
}

void LaserMapping::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
    int size = corr_pts_.size();
    PointCloud::Ptr laser_cloud(new PointCloud(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyToWorld(corr_pts_[i].head<3>(), &laser_cloud->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = tf_world_frame_;
    pub_laser_cloud_effect_world.publish(laserCloudmsg);
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open()) {
        LOG(ERROR) << "Failed to open traj_file: " << traj_file;
        return;
    }

    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : path_.poses) {
        ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
            << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
            << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
    }

    ofs.close();
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
template <typename T>
void LaserMapping::SetPosestamp(T &out) {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = state_point_.rot.coeffs()[0];
    out.pose.orientation.y = state_point_.rot.coeffs()[1];
    out.pose.orientation.z = state_point_.rot.coeffs()[2];
    out.pose.orientation.w = state_point_.rot.coeffs()[3];
}

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z());
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    common::V3D p_body_lidar(pi->x, pi->y, pi->z);
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LaserMapping::Finish() {
    if (pcl_wait_save_->size() > 0 && pcd_save_en_ && pcd_save_interval_ < 0) {
        std::string pcd_save_fname(output_dir + "/map.pcd");
        pcl::PCDWriter pcd_writer;
        LOG(INFO) << "current scan saved to " << pcd_save_fname;
        pcd_writer.writeBinary(pcd_save_fname, *pcl_wait_save_);
    }

    LOG(INFO) << "finish done";
}
}  // namespace faster_lio
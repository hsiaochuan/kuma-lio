#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cv_bridge/cv_bridge.h>
#include "laser_mapping.h"
#include <pcl/filters/uniform_sampling.h>
#include <cameras/pinhole_fisheye_camera.h>
#include <cameras/pinhole_camera.h>
#include <cameras/spherial_camera.h>
#include <cameras/pinhole_radial.h>
#include "utils.h"
namespace fs = boost::filesystem;
namespace faster_lio {

bool LaserMapping::InitROS(ros::NodeHandle &nh, const std::string & config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    if (!LoadParamsFromYAML(config_fname))
        return false;
    SubAndPubToROS(nh);


    run_in_offline_ = false;
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
    run_in_offline_ = true;
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

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    std::string lidar_type;
    int ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double scan_filter_size;

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
        lidar_type = yaml["preprocess"]["lidar_type"].as<std::string>();
        preprocess_->PointFilterNum() = yaml["point_filter_num"].as<int>();
        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        image_save_en_ = yaml["image_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        final_map_voxel_size_ = yaml["pcd_save"]["final_map_voxel_size"].as<double>();
        extrin_il_.q_ = common::RotationFromArray<double>(yaml["mapping"]["extrin_R_il"].as<std::vector<double>>());
        extrin_il_.t_ = common::VecFromArray<double>(yaml["mapping"]["extrin_t_il"].as<std::vector<double>>());
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
            if (yaml["cam"]["extrin_R_cl"].IsDefined()) {
                Pose3 extrin_cl;
                extrin_cl.q_ = common::RotationFromArray<double>(yaml["cam"]["extrin_R_cl"].as<std::vector<double>>());
                extrin_cl.t_ = common::VecFromArray<double>(yaml["cam"]["extrin_t_cl"].as<std::vector<double>>());
                extrin_ic_ = extrin_il_ * extrin_cl.GetInverse();
            } else if (yaml["cam"]["extrin_R_ic"].IsDefined()) {
                extrin_ic_.q_ = common::RotationFromArray<double>(yaml["cam"]["extrin_R_ic"].as<std::vector<double>>());
                extrin_ic_.t_ = common::VecFromArray<double>(yaml["cam"]["extrin_t_ic"].as<std::vector<double>>());
            } else
                throw std::runtime_error("cam extrinsic does not exist");
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
    preprocess_->SetLidarType(LidarTypeFromString(lidar_type));

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

    p_imu_->SetExtrinsic(extrin_il_.Trans(), extrin_il_.Mat3d());
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    return true;
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    if (preprocess_->GetLidarType() == LidarType::LIVOX) {
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
        Image image;
        image.timestamp_ = msg_in->header.stamp.toSec();
        image.image_data_ = img;
        ImageCallBack(image);
    }
    img_count++;
}

void LaserMapping::ImageCallBack(Image& image) {

    // time offset
    image.timestamp_ += camera_time_offset_;

    // loop
    if (image.timestamp_ < last_timestamp_camera_) {
        LOG(WARNING) << "image loop back, clear buffer";
        image_buffer_.clear();
    }
    last_timestamp_camera_ = image.timestamp_;

    // set start offset
    if (if_first_scan_)
        return;
    else
        image.timestamp_ = image.timestamp_ - first_scan_time_;

    // push to buffer
    image_buffer_.emplace_back(image);
}
void LaserMapping::CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % image_skip_ == 0) {
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        Image image;
        image.timestamp_ = msg_in->header.stamp.toSec();
        image.image_data_ = img;
        ImageCallBack(image);
    }
    img_count++;
}
bool LaserMapping::SyncPackages() {
    if (points_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    if (camera_enable_ && image_buffer_.empty())
        return false;

    // set the measure end timestamp
    if (lidar_end_time_ == 0) {
        // for first time
        if (camera_enable_) {
            lidar_end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        }else
            lidar_end_time_ = points_buffer_.front().timestamp + scan_interval_;
    } else if (measures_.lidar_end_time_ == lidar_end_time_) {
        // after the update, incre the end time
        if (camera_enable_) {
            lidar_end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
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
    Eigen::Isometry3d body_pose;
    body_pose.linear() = Eigen::Quaterniond(msg_body_pose.pose.orientation.w, msg_body_pose.pose.orientation.x, msg_body_pose.pose.orientation.y,
                                      msg_body_pose.pose.orientation.z).toRotationMatrix();
    body_pose.translation() = Eigen::Vector3d(msg_body_pose.pose.position.x,msg_body_pose.pose.position.y,msg_body_pose.pose.position.z);
    trajectory_.emplace_back(lidar_end_time_,body_pose);

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
    if (image_save_en_) {
        static auto once = fs::create_directories(output_dir + "/images");
        std::stringstream ss;
        ss << std::setw(15) << std::setfill('0') << std::fixed << std::setprecision(8) << measures_.lidar_end_time_;
        std::string img_save_fname(output_dir + "/images/" + ss.str() + ".jpg");
        if (!measures_.img_.empty())
            cv::imwrite(img_save_fname, measures_.img_);
    }

    if (pcd_save_en_) {
        static auto once = fs::create_directories(output_dir + "/scans");
        std::stringstream ss;
        ss << std::setw(15) << std::setfill('0') << std::fixed << std::setprecision(8) << measures_.lidar_end_time_;
        std::string pcd_save_fname(output_dir + "/scans/" + ss.str() + ".pcd");
        pcl::io::savePCDFileBinary(pcd_save_fname, *scan_undistort_);
    }

    if (pcd_save_en_) {
        *pcl_wait_save_ += *scan_world;
        *final_map_ += *scan_world;
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
            static auto once = fs::create_directories(output_dir + "/maps");

            // sample
            scan_sampler_.setInputCloud(pcl_wait_save_);
            scan_sampler_.filter(*pcl_wait_save_);

            // load pcd
            std::ostringstream pcd_save_fname_ss;
            pcd_save_fname_ss << output_dir << "/maps/" << std::setw(6) << std::setfill('0') << pcd_idx
                              << ".pcd";
            std::string pcd_save_fname(pcd_save_fname_ss.str());
            if (!pcl_wait_save_->empty())
                pcl::io::savePCDFileBinary(pcd_save_fname, *pcl_wait_save_);
            pcd_idx++;
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
    TrajectoryGenerator::save_to_tumtxt(trajectory_, traj_file);

    Trajectory cam_traj;
    for (auto stamp_pose : trajectory_) {
        stamp_pose.pose = stamp_pose.pose * extrin_ic_.Isometry3d();
        cam_traj.emplace_back(stamp_pose);
    }
    std::string cam_traj_file = fs::path(traj_file).parent_path().string() + "/cam_traj_log.txt";
    TrajectoryGenerator::save_to_tumtxt(cam_traj,cam_traj_file);
    TrajectoryGenerator::save_to_pcd(cam_traj,fs::path(traj_file).parent_path().string() + "/cam_traj_log.ply");
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
    if (!final_map_->empty() && pcd_save_en_) {
        // sample
        pcl::UniformSampling<PointType> map_sampler;
        map_sampler.setRadiusSearch(final_map_voxel_size_);
        map_sampler.setInputCloud(final_map_);
        map_sampler.filter(*final_map_);
        // load pcd
        std::string pcd_save_fname(output_dir + "/final_map.pcd");
        pcl::io::savePCDFileBinary(pcd_save_fname, *final_map_);
    }

    if (pcd_save_interval_ > 0) {
        static auto once = fs::create_directories(output_dir + "/maps");

        // sample
        scan_sampler_.setInputCloud(pcl_wait_save_);
        scan_sampler_.filter(*pcl_wait_save_);

        // load pcd
        std::ostringstream pcd_save_fname_ss;
        pcd_save_fname_ss << output_dir << "/maps/" << std::setw(6) << std::setfill('0') << pcd_idx
                          << ".pcd";
        std::string pcd_save_fname(pcd_save_fname_ss.str());
        if (!pcl_wait_save_->empty())
            pcl::io::savePCDFileBinary(pcd_save_fname, *pcl_wait_save_);
        pcd_idx++;
    }
    LOG(INFO) << "finish done";
}
}  // namespace faster_lio
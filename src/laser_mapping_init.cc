#include "laser_mapping.h"
#include <yaml-cpp/yaml.h>
#include "cameras/cameras.h"
#include "global_optimizor.h"
#include "utils.h"

namespace faster_lio {

LaserMapping::LaserMapping() {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
}

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    if (!LoadParamsFromYAML(config_fname))
        return false;

    ivox_ = std::make_shared<IVoxType>(ivox_options_);
    mapper = std::make_shared<GlobalOptimizor>();
    GlobalOptimizor::Options global_options;
    global_options.LoadFromYaml(config_fname);
    mapper->options_ = global_options;
    mapper->output_dir = output_dir;

    if (image_save_en_ && camera_) {
        camera_t cam_id = 1;
        sfm_data_.cameras_[cam_id] = camera_;
    }

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
        extrin_il_.q_ = RotationFromArray<double>(yaml["mapping"]["extrin_R_il"].as<std::vector<double>>());
        extrin_il_.t_ = VecFromArray<double>(yaml["mapping"]["extrin_t_il"].as<std::vector<double>>());
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
        try {
            std::vector<double> resolution;
            std::vector<double> distort_param;
            std::vector<double> pinhole_param;
            auto camera_type = yaml["cam"]["camera_model"].as<std::string>();
            CAMERA_MODEL camera_model = ToCameraModel(camera_type);
            resolution = yaml["cam"]["resolution"].as<std::vector<double>>();
            if (IsPinhole(camera_model)) {
                pinhole_param = yaml["cam"]["pinhole_param"].as<std::vector<double>>();
            }
            if (IsDistorted(camera_model)) {
                distort_param = yaml["cam"]["distortion_param"].as<std::vector<double>>();
            }
            if (yaml["cam"]["extrin_R_cl"].IsDefined()) {
                Pose3 extrin_cl;
                extrin_cl.q_ = RotationFromArray<double>(yaml["cam"]["extrin_R_cl"].as<std::vector<double>>());
                extrin_cl.t_ = VecFromArray<double>(yaml["cam"]["extrin_t_cl"].as<std::vector<double>>());
                extrin_ic_ = extrin_il_ * extrin_cl.GetInverse();
            } else if (yaml["cam"]["extrin_R_ic"].IsDefined()) {
                extrin_ic_.q_ = RotationFromArray<double>(yaml["cam"]["extrin_R_ic"].as<std::vector<double>>());
                extrin_ic_.t_ = VecFromArray<double>(yaml["cam"]["extrin_t_ic"].as<std::vector<double>>());
            } else
                throw std::runtime_error("cam extrinsic does not exist");

            std::vector<double> param;
            param.insert(param.end(), pinhole_param.begin(), pinhole_param.end());
            param.insert(param.end(), distort_param.begin(), distort_param.end());
            switch (camera_model) {
                case PINHOLE:
                    camera_ = std::make_shared<PinholeCamera>();
                    break;
                case PINHOLE_RADIAL:
                    camera_ = std::make_shared<PinholeRadialCamera>();
                    break;
                case PINHOLE_FISHEYE:
                    camera_ = std::make_shared<PinholeFisheyeCamera>();
                    break;
                case SPHERICAL:
                    camera_ = std::make_shared<SphericalCamera>();
                    break;
            }
            camera_->update_params(param);
            camera_->w_ = static_cast<unsigned int>(resolution[0]);
            camera_->h_ = static_cast<unsigned int>(resolution[1]);
        } catch (...) {
            LOG(ERROR) << "bad conversion in camera load";
            return false;
        }
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
    p_imu_->SetGyrCov(Vec3(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(Vec3(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(Vec3(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(Vec3(b_acc_cov, b_acc_cov, b_acc_cov));
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
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
}

}  // namespace faster_lio


//
// Created by hsiaochuan on 2026/05/09.
//

#ifndef FASTER_LIO_LASER_MAPPING_PARAM_H
#define FASTER_LIO_LASER_MAPPING_PARAM_H
#include <yaml-cpp/yaml.h>
#include "cameras/cameras.h"
#include "pose3.h"
using namespace faster_lio;
class LaserMappingParam {
   public:
    std::string lidar_topic_;
    std::string imu_topic_;
    std::string camera_topic_;
    bool camera_enable_ = false;
    bool visual_update_ = false;
    bool imu_enable_ = true;
    bool acc_ratio_ = false;
    double lidar_time_offset_ = 0.;
    double camera_time_offset_ = 0.;
    std::shared_ptr<CamModel> camera_;
    int image_skip_ = 3;

    std::string lidar_type;
    float det_range_ = 300.0f;
    double blind = 2.0;
    int point_filter_num = 1;

    Pose3 extrin_il_ = Pose3::Identity();
    Pose3 extrin_ic_ = Pose3::Identity();

    double cov_P0 = 0.02;
    Vec3 cov_ba = Vec3(0.2, 0.2, 0.2);
    Vec3 cov_bg = Vec3(0.2, 0.2, 0.2);
    double cov_RCP_pos_old = 0.1;
    double cov_RCP_ort_old = 0.1;
    double cov_RCP_pos_new = 1.0;
    double cov_RCP_ort_new = 1.0;
    double std_sys_pos = 0.1;
    double std_sys_ort = 0.1;
    Vec3 cov_acc = Vec3(1.0, 1.0, 1.0);
    Vec3 cov_gyro = Vec3(0.1, 0.1, 0.1);
    int64_t dt_ns = S_TO_NS(0.01);   // 1 / knot_hz [s]
    double ds_scan_voxel = 0.5;  // per-scan uniform-sampling leaf [m]
    double ds_lm_voxel = 0.5;    // ikd-tree local-map downsample leaf [m]
    double cube_len = 2000.0;    // local map cube edge length [m]
    double nn_search_radius = 2.236; // ikd-tree Nearest_Search max_dist [m]
    double plane_thresh = 0.1;       // max |point-to-plane| of the 5 neighbours when fitting [m]
    double ppl_thr = 0.5;          // point-to-plane residual gate [m]
    double coeff_cov = 2.0;          // lid_cov < w_pt * coeff_cov gate
    double w_pt = 0.01;              // per-point measurement variance (RESPLE `w_pt`)
    double imu_acc_outlier = 10.0;   // |acc - predicted| gate per axis [m/s^2]
    double imu_gyro_outlier = 5.0;   // |gyro - predicted| gate per axis [rad/s]

    bool image_save_en_ = false;
    bool pcd_save_en_ = false;
    double pcd_save_interval_ = 30;
    bool bag_save_en_ = false;
    bool LoadFromYaml(const std::string& config_fname) {
        auto yaml = YAML::LoadFile(config_fname);
        try {
            lidar_topic_ = yaml["common"]["lid_topic"].as<std::string>();
            imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
            camera_topic_ = yaml["common"]["camera_topic"].as<std::string>();
            camera_enable_ = yaml["common"]["camera_enable"].as<bool>();
            visual_update_ = yaml["common"]["visual_update"].as<bool>();
            imu_enable_ = yaml["common"]["imu_enable"].as<bool>();
            acc_ratio_ = yaml["common"]["acc_ratio"].as<bool>();
            camera_time_offset_ = yaml["common"]["camera_time_offset"].as<double>();
            lidar_time_offset_ = yaml["common"]["lidar_time_offset"].as<double>();
            image_skip_ = yaml["common"]["image_skip"].as<int>();
            blind = yaml["blind"].as<double>();
            lidar_type = yaml["lidar_type"].as<std::string>();
            point_filter_num = yaml["point_filter_num"].as<int>();
            det_range_ = yaml["det_range"].as<float>();

            extrin_il_.q_ = RotationFromArray(yaml["extrin_R_il"].as<std::vector<double>>());
            extrin_il_.t_ = VecFromArray(yaml["extrin_t_il"].as<std::vector<double>>());

            ds_scan_voxel = yaml["ds_scan_voxel"].as<double>();
            ds_lm_voxel = yaml["ds_lm_voxel"].as<double>();

            dt_ns = S_TO_NS(1.0 / yaml["knot_hz"].as<double>());
            w_pt = yaml["w_pt"].as<double>();
            plane_thresh = yaml["plane_thresh"].as<double>();
            ppl_thr = yaml["nn_thresh"].as<double>();
            coeff_cov = yaml["coeff_cov"].as<double>();
            std_sys_pos = yaml["std_sys_pos"].as<double>();
            std_sys_ort = yaml["std_sys_ort"].as<double>();
            cov_acc = VecFromArray(yaml["cov_acc"].as<std::vector<double>>());
            cov_gyro = VecFromArray(yaml["cov_gyro"].as<std::vector<double>>());

            pcd_save_en_ = yaml["pcd_save_en"].as<bool>();
            pcd_save_interval_ = yaml["pcd_save_interval"].as<int>();
            bag_save_en_ = yaml["bag_save_en"].as<bool>();
            image_save_en_ = yaml["image_save_en"].as<bool>();
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
                    extrin_cl.q_ = RotationFromArray(yaml["cam"]["extrin_R_cl"].as<std::vector<double>>());
                    extrin_cl.t_ = VecFromArray(yaml["cam"]["extrin_t_cl"].as<std::vector<double>>());
                    extrin_ic_ = extrin_il_ * extrin_cl.GetInverse();
                } else if (yaml["cam"]["extrin_R_ic"].IsDefined()) {
                    extrin_ic_.q_ = RotationFromArray(yaml["cam"]["extrin_R_ic"].as<std::vector<double>>());
                    extrin_ic_.t_ = VecFromArray(yaml["cam"]["extrin_t_ic"].as<std::vector<double>>());
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
        return true;
    }
};

#endif  // FASTER_LIO_LASER_MAPPING_PARAM_H

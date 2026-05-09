//
// Created by hsiaochuan on 2026/05/09.
//

#ifndef FASTER_LIO_LASER_MAPPING_PARAM_H
#define FASTER_LIO_LASER_MAPPING_PARAM_H
#include <yaml-cpp/yaml.h>
#include "cameras/cameras.h"
#include "ivox3d/ivox3d.h"
#include "pose3.h"
using namespace faster_lio;
class LaserMappingParam {
   public:
    // parameters
#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, Point>;
#endif

    IVoxType::Options ivox_options_;
    bool pcd_save_en_ = false;
    bool image_save_en_ = false;
    int pcd_save_interval_ = -1;
    bool path_save_en_ = false;
    std::string lidar_type;
    int ivox_nearby_type;
    double gyr_cov;
    double acc_cov;
    double b_gyr_cov;
    double b_acc_cov;
    double scan_filter_size;
    float det_range_ = 300.0f;
    double cube_len_ = 0;
    double map_filter_size_ = 0;
    Pose3 extrin_il_ = Pose3::Identity();
    double scan_interval_ = 0.1;
    float esti_plane_thr = 0.1;
    int max_iteraions = 4;
    std::string lidar_topic_;
    std::string imu_topic_;
    std::string camera_topic_;
    bool camera_enable_;
    double lidar_time_offset_ = 0.;
    double camera_time_offset_ = 0.;
    int image_skip_ = 3;
    int point_filter_num = 1;
    double blind = 2.0;

    Pose3 extrin_ic_ = Pose3::Identity();
    std::shared_ptr<CamModel> camera_;

    bool LoadFromYaml(const std::string& config_fname) {
        auto yaml = YAML::LoadFile(config_fname);
        try {
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
            blind = yaml["preprocess"]["blind"].as<double>();
            lidar_type = yaml["preprocess"]["lidar_type"].as<std::string>();
            point_filter_num = yaml["point_filter_num"].as<int>();
            pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
            image_save_en_ = yaml["image_save_en"].as<bool>();
            pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
            extrin_il_.q_ = RotationFromArray(yaml["mapping"]["extrin_R_il"].as<std::vector<double>>());
            extrin_il_.t_ = VecFromArray(yaml["mapping"]["extrin_t_il"].as<std::vector<double>>());
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

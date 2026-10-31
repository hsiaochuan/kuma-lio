//
// Created by hsiaochuan on 2026/05/09.
//

#ifndef FASTER_LIO_LASER_MAPPING_PARAM_H
#define FASTER_LIO_LASER_MAPPING_PARAM_H
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include "cameras/camera_io.h"
#include "cameras/cameras.h"
#include "pose3.h"
#include "local_ba.h"
using namespace faster_lio;
class LaserMappingParam {
   public:
    LaserMappingParam() {
        prcopt = prcopt_default;
        prcopt.mode = PMODE_SINGLE;
        prcopt.soltype = 0;
        prcopt.nf = 1;
        prcopt.navsys = SYS_GPS | SYS_GLO | SYS_GAL | SYS_CMP;
        prcopt.glomodear = GLO_ARMODE_ON;
        prcopt.refpos = POSOPT_SINGLE;
        prcopt.thresdop = 1.0;
        for (int f = 0; f < NFREQ; ++f) {
            for (int i = 0; i < 9; ++i) prcopt.snrmask.mask[f][i] = 0;
        }
        prcopt.snrmask.ena[0] = prcopt.snrmask.ena[1] = 0;

        solopt = solopt_default;
        solopt.posf = SOLF_XYZ;     // ECEF, so readsolt() hands back sol.rr directly
        solopt.times = TIMES_GPST;  // same time base as the observations
        solopt.timef = 0;           // week + tow
        solopt.timeu = 6;
        solopt.outhead = 1;    // readsolt() needs the header to pick the decoder
        solopt.outvel = 1;     // velocity round-trips through the file as well
        solopt.solstatic = 0;  // one line per epoch, even in a static mode
        solopt.sstat = 0;
        solopt.maxsolstd = 0.0;
    }
    std::string lidar_topic_;
    std::string imu_topic_;
    std::string gnss_topic_;
    bool gnss_enable_ = false;
    bool acc_ratio_ = false;
    double lidar_time_offset_ = 0.;

    std::string lidar_type;
    float det_range_ = 300.0f;
    double blind = 2.0;
    int point_filter_num = 1;

    Pose3 extrin_il_ = Pose3::Identity();
    Pose3 extrin_ic_ = Pose3::Identity();
    CamModel::Ptr cam_model_ = nullptr;

    double cov_P0 = 0.02;
    Vec3 cov_ba = Vec3(0.2, 0.2, 0.2);
    Vec3 cov_bg = Vec3(0.2, 0.2, 0.2);
    double q_std_pos_old = 0.0316228;
    double q_std_ort_old = 0.0316228;
    double q_std_pos_new = 0.1;
    double q_std_ort_new = 0.1;
    Vec3 cov_acc = Vec3(1.0, 1.0, 1.0);
    Vec3 cov_gyro = Vec3(0.1, 0.1, 0.1);
    int64_t dt_ns = S_TO_NS(0.01);   // 1 / knot_hz [s]
    double ds_scan_voxel = 0.5;  // per-scan uniform-sampling leaf [m]
    double ppl_thr = 0.5;          // point-to-plane residual gate [m]
    double coeff_cov = 2.0;          // lid_cov < w_pt * coeff_cov gate
    double w_pt = 0.01;              // per-point measurement variance when `lba.cov_en` is off
    double plane_thresh = 0.1;       // dynamic init only: max |point-to-plane| when fitting a plane [m]
    double imu_acc_outlier = 10.0;   // |acc - predicted| gate per axis [m/s^2]
    double imu_gyro_outlier = 5.0;   // |gyro - predicted| gate per axis [rad/s]
    int num_threads = 4;

    Vec3 level_arm = Vec3(0.0, 0.0, 0.0);
    double max_gap = 0.1;
    double ali_min_travel = 50.0;
    bool tdcp_enable = true;
    double tdcp_std = 0.02;           // TDCP noise at zenith, scaled by 1/sin(el) [m]
    double tdcp_outlier = 0.5;        // |residual - epoch median| hard cap [m]
    double tdcp_chi2 = 10.83;         // chi-square(1) gate on the normalized innovation, 99.9%
    int tdcp_min_sat = 5;             // skip the epoch below this many TDCP rows
    double tdcp_init_weight = 1.0;    // 0 disables the factor
    double tdcp_init_min_horiz = 0.05;  // below this GNSS horizontal step only the up residual is kept [m]
    double tdcp_init_max_std = 0.5;   // reject the epoch pair when the WLS position std exceeds this [m]
    double std_clk = 10.0;            // common receiver clock random walk [m/sqrt(s)]
    double std_isb = 0.01;            // per-system inter-system bias random walk [m/sqrt(s)]
    double std_clk_init = 10.0;       // receiver clock std at alignment [m]
    double std_ext_yaw_init = 0.2;    // yaw std of theta at alignment [rad]
    double std_ext_pos_init = 5.0;    // t_ecef_enu std at alignment [m]
    prcopt_t prcopt;
    solopt_t solopt;

    bool pcd_save_en_ = false;
    double pcd_save_interval_ = 30;
    bool bag_save_en_ = false;

    faster_lio::LbaParam lba_;

    void LoadFromYaml(const std::string& config_fname) {
        auto yaml = YAML::LoadFile(config_fname);
        try {
            lidar_topic_ = yaml["common"]["lid_topic"].as<std::string>();
            imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
            gnss_topic_ = yaml["common"]["gnss_topic"].as<std::string>();
            gnss_enable_ = yaml["common"]["gnss_enable"].as<bool>();
            acc_ratio_ = yaml["common"]["acc_ratio"].as<bool>();
            lidar_time_offset_ = yaml["common"]["lidar_time_offset"].as<double>();
            blind = yaml["blind"].as<double>();
            lidar_type = yaml["lidar_type"].as<std::string>();
            point_filter_num = yaml["point_filter_num"].as<int>();
            det_range_ = yaml["det_range"].as<float>();

            extrin_il_.q_ = RotationFromArray(yaml["extrin_R_il"].as<std::vector<double>>());
            extrin_il_.t_ = VecFromArray(yaml["extrin_t_il"].as<std::vector<double>>());

            ds_scan_voxel = yaml["ds_scan_voxel"].as<double>();

            dt_ns = S_TO_NS(1.0 / yaml["knot_hz"].as<double>());
            w_pt = yaml["w_pt"].as<double>();
            if (yaml["plane_thresh"]) plane_thresh = yaml["plane_thresh"].as<double>();
            ppl_thr = yaml["nn_thresh"].as<double>();
            coeff_cov = yaml["coeff_cov"].as<double>();
            q_std_pos_old = yaml["q_std_pos_old"].as<double>();
            q_std_ort_old = yaml["q_std_ort_old"].as<double>();
            q_std_pos_new = yaml["q_std_pos_new"].as<double>();
            q_std_ort_new = yaml["q_std_ort_new"].as<double>();
            cov_acc = VecFromArray(yaml["cov_acc"].as<std::vector<double>>());
            cov_gyro = VecFromArray(yaml["cov_gyro"].as<std::vector<double>>());
            cov_bg = VecFromArray(yaml["cov_bg"].as<std::vector<double>>());
            cov_ba = VecFromArray(yaml["cov_ba"].as<std::vector<double>>());

            pcd_save_en_ = yaml["pcd_save_en"].as<bool>();
            pcd_save_interval_ = yaml["pcd_save_interval"].as<int>();
            bag_save_en_ = yaml["bag_save_en"].as<bool>();

            if (yaml["gnss"]) {
                const YAML::Node gnss = yaml["gnss"];
                if (gnss["lever_arm"]) level_arm = VecFromArray(gnss["lever_arm"].as<std::vector<double>>());
                ali_min_travel = gnss["ali_min_travel"].as<double>(ali_min_travel);
                max_gap = gnss["max_gap"].as<double>(max_gap);
                tdcp_enable = gnss["tdcp_enable"].as<bool>(tdcp_enable);
                tdcp_std = gnss["tdcp_std"].as<double>(tdcp_std);
                tdcp_outlier = gnss["tdcp_outlier"].as<double>(tdcp_outlier);
                tdcp_chi2 = gnss["tdcp_chi2"].as<double>(tdcp_chi2);
                tdcp_min_sat = gnss["tdcp_min_sat"].as<int>(tdcp_min_sat);
                tdcp_init_weight = gnss["tdcp_init_weight"].as<double>(tdcp_init_weight);
                tdcp_init_min_horiz = gnss["tdcp_init_min_horiz"].as<double>(tdcp_init_min_horiz);
                tdcp_init_max_std = gnss["tdcp_init_max_std"].as<double>(tdcp_init_max_std);
                std_clk = gnss["std_clk"].as<double>(std_clk);
                std_isb = gnss["std_isb"].as<double>(std_isb);
                std_clk_init = gnss["std_clk_init"].as<double>(std_clk_init);
                std_ext_yaw_init = gnss["std_ext_yaw_init"].as<double>(std_ext_yaw_init);
                std_ext_pos_init = gnss["std_ext_pos_init"].as<double>(std_ext_pos_init);
            }

            if (yaml["lba"]) {
                const YAML::Node b = yaml["lba"];
                lba_.win_size = b["win_size"].as<int>(lba_.win_size);
                lba_.voxel_size = b["voxel_size"].as<double>(lba_.voxel_size);
                lba_.plane_eigen_ratio = b["plane_eigen_ratio"].as<double>(lba_.plane_eigen_ratio);
                lba_.min_plane_size = b["min_plane_size"].as<double>(lba_.min_plane_size);
                lba_.clutter_points = b["clutter_points"].as<int>(lba_.clutter_points);
                lba_.min_points = b["min_points"].as<int>(lba_.min_points);
                lba_.max_fix_points = b["max_fix_points"].as<int>(lba_.max_fix_points);
                lba_.max_iter = b["max_iter"].as<int>(lba_.max_iter);
                lba_.imu_coef = b["imu_coef"].as<double>(lba_.imu_coef);
                lba_.cov_gyr = b["cov_gyr"].as<double>(lba_.cov_gyr);
                lba_.cov_acc = b["cov_acc"].as<double>(lba_.cov_acc);
                lba_.rdw_gyr = b["rdw_gyr"].as<double>(lba_.rdw_gyr);
                lba_.rdw_acc = b["rdw_acc"].as<double>(lba_.rdw_acc);
                lba_.sigma_num = b["sigma_num"].as<double>(lba_.sigma_num);
                lba_.pose_var = b["pose_var"].as<double>(lba_.pose_var);
                lba_.range_err = b["range_err"].as<double>(lba_.range_err);
                lba_.angle_err = b["angle_err"].as<double>(lba_.angle_err);
                lba_.cov_en = b["cov_en"].as<bool>(lba_.cov_en);
                lba_.r_base = b["r_base"].as<double>(lba_.r_base);
                lba_.fb_std_pos = b["fb_std_pos"].as<double>(lba_.fb_std_pos);
                lba_.fb_std_ort = b["fb_std_ort"].as<double>(lba_.fb_std_ort);
                lba_.fb_std_bg = b["fb_std_bg"].as<double>(lba_.fb_std_bg);
                lba_.fb_std_ba = b["fb_std_ba"].as<double>(lba_.fb_std_ba);
                lba_.fb_keep_cov = b["fb_keep_cov"].as<bool>(lba_.fb_keep_cov);
                lba_.frame_dt = b["frame_dt"].as<double>(lba_.frame_dt);
                lba_.map_range = b["map_range"].as<double>(lba_.map_range);
            }

            if (yaml["cam"]) {
                // intrinsics as one COLMAP cameras.txt line: MODEL WIDTH HEIGHT PARAMS[]
                if (yaml["cam"]["model"].IsDefined()) {
                    std::ostringstream line;
                    line << std::setprecision(17) << "0 " << yaml["cam"]["model"].as<std::string>() << " "
                         << yaml["cam"]["width"].as<int>() << " " << yaml["cam"]["height"].as<int>();
                    for (double p : yaml["cam"]["params"].as<std::vector<double>>()) line << " " << p;
                    std::istringstream stream(line.str());
                    cam_model_ = ReadCameraFromCamerasText(stream);
                    CHECK(cam_model_) << "cannot build a camera from the `cam:` block: " << line.str();
                }
                if (yaml["cam"]["extrin_R_cl"].IsDefined()) {
                    Pose3 extrin_cl;
                    extrin_cl.q_ = RotationFromArray(yaml["cam"]["extrin_R_cl"].as<std::vector<double>>());
                    extrin_cl.t_ = VecFromArray(yaml["cam"]["extrin_t_cl"].as<std::vector<double>>());
                    extrin_ic_ = extrin_il_ * extrin_cl.GetInverse();
                } else if (yaml["cam"]["extrin_R_ic"].IsDefined()) {
                    extrin_ic_.q_ = RotationFromArray(yaml["cam"]["extrin_R_ic"].as<std::vector<double>>());
                    extrin_ic_.t_ = VecFromArray(yaml["cam"]["extrin_t_ic"].as<std::vector<double>>());
                }
            }
        } catch (...) {
            throw std::runtime_error("bad conversion");
        }
    }
};

#endif  // FASTER_LIO_LASER_MAPPING_PARAM_H

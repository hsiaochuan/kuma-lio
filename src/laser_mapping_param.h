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
    std::string camera_topic_;
    std::string gnss_topic_;
    bool camera_enable_ = false;
    bool gnss_enable_ = false;
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
    CamModel::Ptr cam_model_ = nullptr;

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
    int num_threads = 4;

    Vec3 level_arm = Vec3(0.0, 0.0, 0.0);
    double dt_weight = 1.0;
    double ddt_weight = 1.0;
    double psr_weight = 1.0;
    double tdcp_weight = 1.0;
    double max_gap = 1.0;
    int slw_size = 200;
    double ali_min_travel = 5.0;
    double lio_pos_std = 0.1;

    // TDCP update inside RESPLE_LIO
    bool tdcp_enable = true;
    double tdcp_std = 0.02;           // TDCP noise at zenith, scaled by 1/sin(el) [m]
    double tdcp_outlier = 0.5;        // |residual - epoch median| hard cap [m]
    double tdcp_chi2 = 10.83;         // chi-square(1) gate on the normalized innovation, 99.9%
    int tdcp_min_sat = 5;             // skip the epoch below this many TDCP rows
    double std_clk = 10.0;            // common receiver clock random walk [m/sqrt(s)]
    double std_isb = 0.01;            // per-system inter-system bias random walk [m/sqrt(s)]
    double std_clk_init = 10.0;       // receiver clock std at alignment [m]
    double std_ext_yaw_init = 0.2;    // yaw std of theta at alignment [rad]
    double std_ext_pos_init = 5.0;    // t_ecef_enu std at alignment [m]
    prcopt_t prcopt;
    solopt_t solopt;

    bool image_save_en_ = false;
    bool pcd_save_en_ = false;
    double pcd_save_interval_ = 30;
    bool bag_save_en_ = false;
    void LoadFromYaml(const std::string& config_fname) {
        auto yaml = YAML::LoadFile(config_fname);
        try {
            lidar_topic_ = yaml["common"]["lid_topic"].as<std::string>();
            imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
            camera_topic_ = yaml["common"]["camera_topic"].as<std::string>();
            gnss_topic_ = yaml["common"]["gnss_topic"].as<std::string>();
            camera_enable_ = yaml["common"]["camera_enable"].as<bool>();
            gnss_enable_ = yaml["common"]["gnss_enable"].as<bool>();
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
                std_clk = gnss["std_clk"].as<double>(std_clk);
                std_isb = gnss["std_isb"].as<double>(std_isb);
                std_clk_init = gnss["std_clk_init"].as<double>(std_clk_init);
                std_ext_yaw_init = gnss["std_ext_yaw_init"].as<double>(std_ext_yaw_init);
                std_ext_pos_init = gnss["std_ext_pos_init"].as<double>(std_ext_pos_init);
            }

            if (camera_enable_) {
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

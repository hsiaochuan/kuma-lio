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
/// The `resple:` block: the spline filter of `front_end: resple`, and its feedback from the BA.
struct RespleParam {
    int64_t dt_ns;             // 1 / knot_hz [s]
    double ds_scan_voxel;      // per-scan uniform-sampling leaf [m]
    double w_pt;               // per-point measurement variance when `cov_en` is off
    double ppl_thr;            // point-to-plane residual gate [m]
    double coeff_cov;          // lid_cov < w_pt * coeff_cov gate
    double plane_thresh;       // dynamic init only: max |point-to-plane| when fitting a plane [m]
    double imu_acc_outlier;    // |acc - predicted| gate per axis [m/s^2]
    double imu_gyro_outlier;   // |gyro - predicted| gate per axis [rad/s]
    double q_std_pos_old;
    double q_std_ort_old;
    double q_std_pos_new;
    double q_std_ort_new;
    double q_std_bg;
    double q_std_ba;
    double cov_P0;
    Vec3 cov_acc;
    Vec3 cov_gyro;
    Vec3 cov_bg;
    Vec3 cov_ba;
    bool cov_en;               // per-point residual variance from the plane and the point, else w_pt
    double r_base;             // [m^2] added to the per-point residual variance when cov_en
    double fb_std_pos;         // [m] noise of the BA pose fed back to the spline
    double fb_std_ort;         // [rad]
    double fb_std_bg;          // [rad/s] noise of the BA gyroscope bias fed back with the pose
    double fb_std_ba;          // [m/s^2] noise of the BA accelerometer bias fed back with the pose
    bool fb_keep_cov;          // leave the filter covariance alone: the BA reuses the filter's own data
};

/// The `fastlio:` block: the FAST-LIO2 style front end, with the noises of Voxel-SLAM.
enum class FastLioMode {
    LIO = 1,
    LIVO,
    LIGO,
};
inline FastLioMode FastLioModeFromString(const std::string &fast_lio_mode_str) {
    if (fast_lio_mode_str == "LIO") {
        return FastLioMode::LIO;
    } else if (fast_lio_mode_str == "LIVO") {
        return FastLioMode::LIVO;
    } else if (fast_lio_mode_str == "LIGO") {
        return FastLioMode::LIGO;
    } else {
        throw std::runtime_error("unknown fast-lio mode: " + fast_lio_mode_str);
    }
}

struct FastLioParam {
    FastLioMode mode;
    double down_size;          // [m] per-frame voxel grid
    double cov_gyr;            // [(rad/s)^2] per gyroscope sample
    double cov_acc;            // [(m/s^2)^2] per accelerometer sample
    double rdw_gyr;            // per sample, as Voxel-SLAM
    double rdw_acc;
    double r_base;             // [m^2] added to the per-point residual variance
};

/// A key every config has: the configs all list the same keys.
template <typename T>
T ConfigKey(const YAML::Node& node, const char* key) {
    const YAML::Node value = node[key];
    CHECK(value.IsDefined()) << "config key missing: " << key;
    return value.as<T>();
}
inline Vec3 ConfigVec3(const YAML::Node& node, const char* key) {
    return VecFromArray(ConfigKey<std::vector<double>>(node, key));
}

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
    std::string img_topic_;  // empty: no images
    bool gnss_enable_;
    bool acc_ratio_;
    double lidar_time_offset_;

    std::string front_end;  // resple or fastlio
    int num_threads;
    std::string lidar_type;
    float det_range_;
    double blind;
    int point_filter_num;

    Pose3 extrin_il_ = Pose3::Identity();
    Pose3 extrin_ic_ = Pose3::Identity();
    CamModel::Ptr cam_model_ = nullptr;

    bool pcd_save_en_;
    bool bag_save_en_;
    bool image_save_en_;
    double image_save_interval_;

    RespleParam resple_;
    FastLioParam fast_lio_;
    faster_lio::LbaParam lba_;

    Vec3 level_arm;
    double max_gap;
    double ali_min_travel;
    bool tdcp_enable;
    double tdcp_std;             // TDCP noise at zenith, scaled by 1/sin(el) [m]
    double tdcp_outlier;         // |residual - epoch median| hard cap [m]
    double tdcp_chi2;            // chi-square(1) gate on the normalized innovation, 99.9%
    int tdcp_min_sat;            // skip the epoch below this many TDCP rows
    double tdcp_init_weight;     // 0 disables the factor
    double tdcp_init_min_horiz;  // below this GNSS horizontal step only the up residual is kept [m]
    double tdcp_init_max_std;    // reject the epoch pair when the WLS position std exceeds this [m]
    double std_clk;              // common receiver clock random walk [m/sqrt(s)]
    double std_isb;              // per-system inter-system bias random walk [m/sqrt(s)]
    double std_clk_init;         // receiver clock std at alignment [m]
    double std_ext_yaw_init;     // yaw std of theta at alignment [rad]
    double std_ext_pos_init;     // t_ecef_enu std at alignment [m]
    prcopt_t prcopt;
    solopt_t solopt;

    void LoadFromYaml(const std::string& config_fname) {
        const YAML::Node yaml = YAML::LoadFile(config_fname);
        try {
            const YAML::Node common = yaml["common"];
            lidar_topic_ = ConfigKey<std::string>(common, "lid_topic");
            imu_topic_ = ConfigKey<std::string>(common, "imu_topic");
            gnss_topic_ = ConfigKey<std::string>(common, "gnss_topic");
            img_topic_ = ConfigKey<std::string>(common, "img_topic");
            gnss_enable_ = ConfigKey<bool>(common, "gnss_enable");
            acc_ratio_ = ConfigKey<bool>(common, "acc_ratio");
            lidar_time_offset_ = ConfigKey<double>(common, "lidar_time_offset");

            front_end = ConfigKey<std::string>(yaml, "front_end");
            num_threads = ConfigKey<int>(yaml, "num_threads");
            lidar_type = ConfigKey<std::string>(yaml, "lidar_type");
            blind = ConfigKey<double>(yaml, "blind");
            point_filter_num = ConfigKey<int>(yaml, "point_filter_num");
            det_range_ = ConfigKey<float>(yaml, "det_range");
            extrin_il_.q_ = RotationFromArray(ConfigKey<std::vector<double>>(yaml, "extrin_R_il"));
            extrin_il_.t_ = ConfigVec3(yaml, "extrin_t_il");
            pcd_save_en_ = ConfigKey<bool>(yaml, "pcd_save_en");
            image_save_en_ = ConfigKey<bool>(yaml, "image_save_en");
            image_save_interval_ = ConfigKey<double>(yaml, "image_save_interval");
            bag_save_en_ = ConfigKey<bool>(yaml, "bag_save_en");

            const YAML::Node r = yaml["resple"];
            resple_.dt_ns = S_TO_NS(1.0 / ConfigKey<double>(r, "knot_hz"));
            resple_.ds_scan_voxel = ConfigKey<double>(r, "ds_scan_voxel");
            resple_.w_pt = ConfigKey<double>(r, "w_pt");
            resple_.ppl_thr = ConfigKey<double>(r, "nn_thresh");
            resple_.coeff_cov = ConfigKey<double>(r, "coeff_cov");
            resple_.plane_thresh = ConfigKey<double>(r, "plane_thresh");
            resple_.imu_acc_outlier = ConfigKey<double>(r, "imu_acc_outlier");
            resple_.imu_gyro_outlier = ConfigKey<double>(r, "imu_gyro_outlier");
            resple_.q_std_pos_old = ConfigKey<double>(r, "q_std_pos_old");
            resple_.q_std_ort_old = ConfigKey<double>(r, "q_std_ort_old");
            resple_.q_std_pos_new = ConfigKey<double>(r, "q_std_pos_new");
            resple_.q_std_ort_new = ConfigKey<double>(r, "q_std_ort_new");
            resple_.q_std_bg = ConfigKey<double>(r, "q_std_bg");
            resple_.q_std_ba = ConfigKey<double>(r, "q_std_ba");
            resple_.cov_P0 = ConfigKey<double>(r, "cov_P0");
            resple_.cov_acc = ConfigVec3(r, "cov_acc");
            resple_.cov_gyro = ConfigVec3(r, "cov_gyro");
            resple_.cov_bg = ConfigVec3(r, "cov_bg");
            resple_.cov_ba = ConfigVec3(r, "cov_ba");
            resple_.cov_en = ConfigKey<bool>(r, "cov_en");
            resple_.r_base = ConfigKey<double>(r, "r_base");
            resple_.fb_std_pos = ConfigKey<double>(r, "fb_std_pos");
            resple_.fb_std_ort = ConfigKey<double>(r, "fb_std_ort");
            resple_.fb_std_bg = ConfigKey<double>(r, "fb_std_bg");
            resple_.fb_std_ba = ConfigKey<double>(r, "fb_std_ba");
            resple_.fb_keep_cov = ConfigKey<bool>(r, "fb_keep_cov");

            const YAML::Node f = yaml["fastlio"];
            fast_lio_.mode = FastLioModeFromString(ConfigKey<std::string>(f, "mode"));
            fast_lio_.down_size = ConfigKey<double>(f, "down_size");
            fast_lio_.cov_gyr = ConfigKey<double>(f, "cov_gyr");
            fast_lio_.cov_acc = ConfigKey<double>(f, "cov_acc");
            fast_lio_.rdw_gyr = ConfigKey<double>(f, "rdw_gyr");
            fast_lio_.rdw_acc = ConfigKey<double>(f, "rdw_acc");
            fast_lio_.r_base = ConfigKey<double>(f, "r_base");

            const YAML::Node b = yaml["lba"];
            lba_.win_size = ConfigKey<int>(b, "win_size");
            lba_.voxel_size = ConfigKey<double>(b, "voxel_size");
            lba_.plane_eigen_ratio = ConfigKey<double>(b, "plane_eigen_ratio");
            lba_.min_plane_size = ConfigKey<double>(b, "min_plane_size");
            lba_.clutter_points = ConfigKey<int>(b, "clutter_points");
            lba_.min_points = ConfigKey<int>(b, "min_points");
            lba_.max_fix_points = ConfigKey<int>(b, "max_fix_points");
            lba_.max_iter = ConfigKey<int>(b, "max_iter");
            lba_.imu_coef = ConfigKey<double>(b, "imu_coef");
            lba_.cov_gyr = ConfigKey<double>(b, "cov_gyr");
            lba_.cov_acc = ConfigKey<double>(b, "cov_acc");
            lba_.rdw_gyr = ConfigKey<double>(b, "rdw_gyr");
            lba_.rdw_acc = ConfigKey<double>(b, "rdw_acc");
            lba_.sigma_num = ConfigKey<double>(b, "sigma_num");
            lba_.pose_var = ConfigKey<double>(b, "pose_var");
            lba_.range_err = ConfigKey<double>(b, "range_err");
            lba_.angle_err = ConfigKey<double>(b, "angle_err");
            lba_.frame_dt = ConfigKey<double>(b, "frame_dt");
            lba_.map_range = ConfigKey<double>(b, "map_range");

            const YAML::Node g = yaml["gnss"];
            level_arm = ConfigVec3(g, "lever_arm");
            ali_min_travel = ConfigKey<double>(g, "ali_min_travel");
            max_gap = ConfigKey<double>(g, "max_gap");
            tdcp_enable = ConfigKey<bool>(g, "tdcp_enable");
            tdcp_std = ConfigKey<double>(g, "tdcp_std");
            tdcp_outlier = ConfigKey<double>(g, "tdcp_outlier");
            tdcp_chi2 = ConfigKey<double>(g, "tdcp_chi2");
            tdcp_min_sat = ConfigKey<int>(g, "tdcp_min_sat");
            tdcp_init_weight = ConfigKey<double>(g, "tdcp_init_weight");
            tdcp_init_min_horiz = ConfigKey<double>(g, "tdcp_init_min_horiz");
            tdcp_init_max_std = ConfigKey<double>(g, "tdcp_init_max_std");
            std_clk = ConfigKey<double>(g, "std_clk");
            std_isb = ConfigKey<double>(g, "std_isb");
            std_clk_init = ConfigKey<double>(g, "std_clk_init");
            std_ext_yaw_init = ConfigKey<double>(g, "std_ext_yaw_init");
            std_ext_pos_init = ConfigKey<double>(g, "std_ext_pos_init");

            // an empty model means no camera
            const YAML::Node c = yaml["cam"];
            const std::string cam_model = ConfigKey<std::string>(c, "model");
            const int cam_width = ConfigKey<int>(c, "width");
            const int cam_height = ConfigKey<int>(c, "height");
            const std::vector<double> cam_params = ConfigKey<std::vector<double>>(c, "params");
            Pose3 extrin_cl;
            extrin_cl.q_ = RotationFromArray(ConfigKey<std::vector<double>>(c, "extrin_R_cl"));
            extrin_cl.t_ = ConfigVec3(c, "extrin_t_cl");
            if (!cam_model.empty()) {
                // intrinsics as one COLMAP cameras.txt line: MODEL WIDTH HEIGHT PARAMS[]
                std::ostringstream line;
                line << std::setprecision(17) << "0 " << cam_model << " " << cam_width << " " << cam_height;
                for (double p : cam_params) line << " " << p;
                std::istringstream stream(line.str());
                cam_model_ = ReadCameraFromCamerasText(stream);
                CHECK(cam_model_) << "cannot build a camera from the `cam:` block: " << line.str();
                extrin_ic_ = extrin_il_ * extrin_cl.GetInverse();
            }
        } catch (const YAML::Exception& e) {
            throw std::runtime_error(config_fname + ": " + e.what());
        }

        if (fast_lio_.mode == FastLioMode::LIVO)
            CHECK(!img_topic_.empty());
    }
};

#endif  // FASTER_LIO_LASER_MAPPING_PARAM_H

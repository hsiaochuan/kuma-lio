// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "common_lib.h"
#include "pointcloud_preprocess.h"
#include "pose3.h"

namespace kuma_lio {

// A key every config/resple/*.yaml has: no optional keys, no silent defaults.
template <typename T>
T RespleKey(const YAML::Node& node, const char* key) {
    const YAML::Node value = node[key];
    CHECK(value.IsDefined()) << "config key missing: " << key;
    return value.as<T>();
}
inline Vec3 RespleVec3(const YAML::Node& node, const char* key) {
    return VecFromArray(RespleKey<std::vector<double>>(node, key));
}

// The config of run_resple (config/resple/*.yaml): the sensors, the spline filter (`resple:`) and its ikd-tree
// local map.
class RespleParam {
   public:
    std::string lidar_topic_;
    std::string imu_topic_;
    bool acc_ratio_;
    double lidar_time_offset_;

    int num_threads;
    std::string lidar_type;
    int livox_tag_mask = 0x3F;
    double blind;
    int point_filter_num;
    float det_range_;
    Pose3 extrin_il_ = Pose3::Identity();
    bool bag_save_en_;

    // the spline filter
    int64_t dt_ns;             // 1 / knot_hz [s]
    double ds_scan_voxel;      // per-scan voxel-centroid leaf [m]
    double w_pt;               // per-point measurement variance
    double ppl_thr;            // point-to-plane residual gate [m]
    double coeff_cov;          // lid_cov < w_pt * coeff_cov gate
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

    // the ikd-tree local map
    double plane_thresh;      // max |point-to-plane| of the 5 neighbours when fitting a plane [m]
    double ds_lm_voxel;       // ikd-tree downsample leaf [m]
    double cube_len;          // local map cube edge length [m]
    double nn_search_radius;  // Nearest_Search max distance [m]

    PreprocessParam Preprocess() const {
        PreprocessParam p;
        p.livox_tag_mask = livox_tag_mask;
        p.point_filter_num = point_filter_num;
        p.blind = blind;
        p.det_range_ = det_range_;
        return p;
    }

    void LoadFromYaml(const std::string& config_fname) {
        const YAML::Node yaml = YAML::LoadFile(config_fname);
        try {
            const YAML::Node common = yaml["common"];
            lidar_topic_ = RespleKey<std::string>(common, "lid_topic");
            imu_topic_ = RespleKey<std::string>(common, "imu_topic");
            acc_ratio_ = RespleKey<bool>(common, "acc_ratio");
            lidar_time_offset_ = RespleKey<double>(common, "lidar_time_offset");

            num_threads = RespleKey<int>(yaml, "num_threads");
            lidar_type = RespleKey<std::string>(yaml, "lidar_type");
            livox_tag_mask = RespleKey<int>(yaml, "livox_tag_mask");
            blind = RespleKey<double>(yaml, "blind");
            point_filter_num = RespleKey<int>(yaml, "point_filter_num");
            det_range_ = RespleKey<float>(yaml, "det_range");
            extrin_il_.q_ = RotationFromArray(RespleKey<std::vector<double>>(yaml, "extrin_R_il"));
            extrin_il_.t_ = RespleVec3(yaml, "extrin_t_il");
            bag_save_en_ = RespleKey<bool>(yaml, "bag_save_en");

            const YAML::Node r = yaml["resple"];
            dt_ns = S_TO_NS(1.0 / RespleKey<double>(r, "knot_hz"));
            ds_scan_voxel = RespleKey<double>(r, "ds_scan_voxel");
            w_pt = RespleKey<double>(r, "w_pt");
            ppl_thr = RespleKey<double>(r, "nn_thresh");
            coeff_cov = RespleKey<double>(r, "coeff_cov");
            imu_acc_outlier = RespleKey<double>(r, "imu_acc_outlier");
            imu_gyro_outlier = RespleKey<double>(r, "imu_gyro_outlier");
            q_std_pos_old = RespleKey<double>(r, "q_std_pos_old");
            q_std_ort_old = RespleKey<double>(r, "q_std_ort_old");
            q_std_pos_new = RespleKey<double>(r, "q_std_pos_new");
            q_std_ort_new = RespleKey<double>(r, "q_std_ort_new");
            q_std_bg = RespleKey<double>(r, "q_std_bg");
            q_std_ba = RespleKey<double>(r, "q_std_ba");
            cov_P0 = RespleKey<double>(r, "cov_P0");
            cov_acc = RespleVec3(r, "cov_acc");
            cov_gyro = RespleVec3(r, "cov_gyro");
            cov_bg = RespleVec3(r, "cov_bg");
            cov_ba = RespleVec3(r, "cov_ba");
            plane_thresh = RespleKey<double>(r, "plane_thresh");
            ds_lm_voxel = RespleKey<double>(r, "ds_lm_voxel");
            cube_len = RespleKey<double>(r, "cube_len");
            nn_search_radius = RespleKey<double>(r, "nn_search_radius");
        } catch (const YAML::Exception& e) {
            throw std::runtime_error(config_fname + ": " + e.what());
        }
    }
};

}  // namespace kuma_lio

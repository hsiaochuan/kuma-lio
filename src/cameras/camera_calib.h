//
// The camera calibration yaml (e.g. config/i2nav_robot_camera.yaml), shared by run_mapping_offline
// (--camera_fname, LaserMappingParam::LoadCameraFromYaml) and run_ba (--calib_fname).
//

#ifndef CAMERAS_CAMERA_CALIB_H
#define CAMERAS_CAMERA_CALIB_H

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cameras.h"

namespace faster_lio {

struct CameraCalib {
    Eigen::Isometry3d T_imu_cam = Eigen::Isometry3d::Identity();  ///< P_imu = R P_cam + t
    bool from_lidar = false;                                       ///< T_imu_cam = T_imu_lidar T_cam_lidar^-1
    std::map<uint32_t, CamModel::Ptr> cameras;                     ///< by CAMERA_ID, never empty
};

/**
 * @brief Reads a camera calibration yaml.  The extrinsic comes in one of two forms (rotations 3x3 row major,
 * 9 values; translations 3 values [m]; T_ab means P_a = R P_b + t):
 *   1. extrin_R, extrin_t: T_imu_cam directly
 *   2. extrin_Rcl, extrin_tcl: T_cam_lidar, and extrin_Ril, extrin_til: T_imu_lidar; T_imu_cam = T_il T_cl^-1
 * and the intrinsics:
 *   camera: COLMAP cameras.txt lines, "CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]", one per camera
 * Fatal if an entry is missing or wrong, if both extrinsic forms are given, or if no camera parses.
 */
inline CameraCalib LoadCameraCalib(const std::string& fname) {
    YAML::Node y;
    try {
        y = YAML::LoadFile(fname);
    } catch (const YAML::Exception& e) {
        LOG(FATAL) << "cannot read " << fname << ": " << e.what();
    }
    auto key = [&](const char* name) {
        const YAML::Node value = y[name];
        CHECK(value.IsDefined()) << "key missing: " << name << " in " << fname;
        return value;
    };
    // T_ab from a rotation and a translation key: P_a = R P_b + t
    auto read_pose = [&](const char* r_key, const char* t_key) {
        const std::vector<double> r = key(r_key).as<std::vector<double>>();
        const std::vector<double> t = key(t_key).as<std::vector<double>>();
        CHECK_EQ(r.size(), 9u) << r_key << " must have 9 values (row major) in " << fname;
        CHECK_EQ(t.size(), 3u) << t_key << " must have 3 values in " << fname;
        const Eigen::Matrix3d M = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r.data());
        // a calibration printed with a few digits is not exactly orthonormal: take the nearest rotation
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
        CHECK(R.determinant() > 0.0 && (R - M).norm() < 1e-3) << r_key << " is not a rotation in " << fname;
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = R;
        T.translation() = Eigen::Vector3d(t[0], t[1], t[2]);
        return T;
    };

    CameraCalib calib;
    const bool direct = y["extrin_R"] || y["extrin_t"];
    calib.from_lidar = y["extrin_Rcl"] || y["extrin_tcl"] || y["extrin_Ril"] || y["extrin_til"];
    CHECK(direct != calib.from_lidar) << fname << " needs either extrin_R, extrin_t (T_imu_cam) or extrin_Rcl, "
                                      << "extrin_tcl, extrin_Ril, extrin_til (T_cam_lidar, T_imu_lidar), not both";
    if (direct) {
        calib.T_imu_cam = read_pose("extrin_R", "extrin_t");
    } else {
        const Eigen::Isometry3d T_cl = read_pose("extrin_Rcl", "extrin_tcl");
        const Eigen::Isometry3d T_il = read_pose("extrin_Ril", "extrin_til");
        calib.T_imu_cam = T_il * T_cl.inverse();
    }

    std::istringstream camera_lines(key("camera").as<std::string>());
    calib.cameras = ReadCamerasFromCamerasText(camera_lines);
    CHECK(!calib.cameras.empty()) << "no camera in the camera entry of " << fname;
    return calib;
}

}  // namespace faster_lio

#endif  // CAMERAS_CAMERA_CALIB_H

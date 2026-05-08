#pragma once
#include <Eigen/Eigen>
namespace faster_lio {

struct Pose3 {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static Pose3 Identity();
    static Pose3 InValid();
    bool IsValid() const;
    Pose3();

    explicit Pose3(const Eigen::Matrix3d& rot, const Eigen::Vector3d& t);
    explicit Pose3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    explicit Pose3(const Eigen::Isometry3d& iso);
    explicit Pose3(const Eigen::Matrix4d& mat44);

    Pose3 GetInverse() const;

    Pose3& operator*=(const Pose3& other);
    Pose3 operator*(const Pose3& other) const;
    Eigen::Vector3d operator*(const Eigen::Vector3d& vec3) const;

    Eigen::Vector3d Trans() const;
    Eigen::Quaterniond Quat() const;
    Eigen::Matrix3d Mat3d() const;
    Eigen::Matrix4d Mat4d() const;
    Eigen::Matrix<double, 3, 4> Mat34() const;
    Eigen::Isometry3d Isometry3d() const;

    double* QuatData();
    double* PosData();

    void SetQuat(const Eigen::Quaterniond& q);
    void SetTrans(const Eigen::Vector3d& t) noexcept;

    static Pose3 Interpolate(const Pose3& a, const Pose3& b, double ratio);
    void AddNoise(double noise_trans, double noise_rot_angle);

    Eigen::Quaterniond q_;
    Eigen::Vector3d t_;
};

}  // namespace faster_lio

#pragma once
#include <Eigen/Eigen>
#include <random>
namespace faster_lio {

struct Pose3 {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose3() noexcept
        : q_(Eigen::Quaterniond::Identity()), t_(Eigen::Vector3d::Zero()) {}

    explicit Pose3(const Eigen::Matrix3d& rot, const Eigen::Vector3d& t)
        : q_(rot), t_(t) {
        q_.normalize();
    }

    explicit Pose3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
        : q_(q), t_(t) {
        q_.normalize();
    }

    explicit Pose3(const Eigen::Isometry3d& iso)
        : q_(iso.linear()), t_(iso.translation()) {
        q_.normalize();
    }

    explicit Pose3(const Eigen::Matrix4d& mat44)
        : q_(mat44.block<3, 3>(0, 0)), t_(mat44.block<3, 1>(0, 3)) {
        q_.normalize();
    }

    Pose3 GetInverse() const {
        const Eigen::Quaterniond q_inv = q_.conjugate();
        return Pose3(q_inv, -(q_inv * t_));
    }

    Eigen::Isometry3d GetIsometry3d() const {
        Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
        iso.linear() = q_.toRotationMatrix();
        iso.translation() = t_;
        return iso;
    }

    Pose3& operator*=(const Pose3& other) {
        const Eigen::Quaterniond q_old = q_;
        q_ = q_old * other.q_;
        t_ = q_old * other.t_ + t_;
        q_.normalize();
        return *this;
    }

    Pose3 operator*(const Pose3& other) const {
        Pose3 out;
        out.q_ = q_ * other.q_;
        out.t_ = q_ * other.t_ + t_;
        out.q_.normalize();
        return out;
    }

    Eigen::Vector3d operator*(const Eigen::Vector3d& vec3) const {
        return q_ * vec3 + t_;
    }

    Eigen::Vector3d Trans() const { return t_; }
    Eigen::Quaterniond Quat() const { return q_; }
    Eigen::Matrix3d Mat3d() const { return q_.toRotationMatrix(); }
    Eigen::Matrix4d Mat4d() const {
        Eigen::Matrix4d mat4x4 = Eigen::Matrix4d::Identity();
        mat4x4.block<3, 3>(0, 0) = q_.toRotationMatrix();
        mat4x4.block<3, 1>(0, 3) = t_;
        return mat4x4;
    }

    double* QuatData() { return q_.coeffs().data(); }
    double* PosData() { return t_.data(); }

    void SetQuat(const Eigen::Quaterniond& q) {
        q_ = q;
        q_.normalize();
    }

    void SetTrans(const Eigen::Vector3d& t) noexcept { t_ = t; }


    static Pose3 Interpolate(const Pose3& a, const Pose3& b, double ratio) {
        ratio = std::clamp(ratio, 0.0, 1.0);
        return Pose3(a.q_.slerp(ratio, b.q_), a.t_ * (1.0 - ratio) + b.t_ * ratio);
    }

    void AddNoise(double noise_trans, double noise_rot_angle) {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::normal_distribution<double> trans_dist(0.0, noise_trans);
        Eigen::Vector3d trans_noise(trans_dist(gen), trans_dist(gen), trans_dist(gen));

        std::normal_distribution<double> rot_dist(0.0, noise_rot_angle);
        Eigen::Vector3d omega(rot_dist(gen), rot_dist(gen), rot_dist(gen));

        const double theta = omega.norm();
        Eigen::Matrix3d R_noise = Eigen::Matrix3d::Identity();
        if (theta >= 1e-12) {
            const Eigen::Vector3d axis = omega / theta;
            R_noise = Eigen::AngleAxisd(theta, axis).toRotationMatrix();
        }

        t_ += trans_noise;
        q_ = Eigen::Quaterniond(q_.toRotationMatrix() * R_noise);
        q_.normalize();
    }

private:
    Eigen::Quaterniond q_;
    Eigen::Vector3d t_;
};

}  // namespace faster_lio

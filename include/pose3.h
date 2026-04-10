#pragma once
#include <Eigen/Eigen>
#include <random>
namespace faster_lio {

struct Pose3 {
    Pose3() : q_(Eigen::Quaterniond::Identity()), t_(Eigen::Vector3d::Zero()) {}
    Pose3(const Eigen::Matrix3d& rot, const Eigen::Vector3d& t) : q_(rot), t_(t) {}
    Pose3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) : q_(q), t_(t) {}
    Pose3(const Eigen::Isometry3d& iso) {
        q_ = Eigen::Quaterniond(iso.linear());
        t_ = iso.translation();
    }
    explicit Pose3(const Eigen::Matrix4d& mat44) : t_(mat44.block(0, 3, 3, 1)) {
        Eigen::Matrix3d mat = mat44.block(0, 0, 3, 3);
        q_ = Eigen::Quaterniond(mat);
    }

    Pose3 GetInverse() const { return {q_.conjugate(), q_.conjugate() * t_ * double(-1.0)}; }
    Eigen::Isometry3d GetIsometry3d() const {
        Eigen::Isometry3d iso;
        iso.linear() = q_.toRotationMatrix();
        iso.translation() = t_;
        return iso;
    }
    Eigen::Matrix4d GetMat4d() const {
        Eigen::Matrix4d mat4x4;
        mat4x4.setIdentity();
        mat4x4.block(0, 0, 3, 3) = q_.toRotationMatrix();
        mat4x4.block(0, 3, 3, 1) = t_;
        return mat4x4;
    }
    Pose3& operator*=(const Pose3& other) {
        this->q_ = q_ * other.q_;
        this->t_ = q_ * other.t_ + t_;
        return *this;
    }
    Pose3 operator*(const Pose3& other) const { return {q_ * other.q_, q_ * other.t_ + t_}; }
    Eigen::Vector3d operator*(const Eigen::Vector3d& vec3) const { return q_ * vec3 + t_; }
    Eigen::Vector3d GetTrans() const { return t_; }
    Eigen::Quaterniond GetQuat() const { return q_; }
    Eigen::Vector3d& Trans() { return t_; }
    Eigen::Quaterniond& Quat() { return q_; }
    double* QuatData() { return Quat().coeffs().data(); }
    double* PosData() { return Trans().data(); }
    void SetQuat(const Eigen::Quaterniond& q) { q_ = q; }
    void SetTrans(const Eigen::Vector3d& t) { t_ = t; }
    Eigen::Matrix3d GetMat3d() const { return q_.toRotationMatrix(); }
    static Pose3 Interpolate(const Pose3& a, const Pose3& b, const double& ratio) {
        return {a.q_.slerp(ratio, b.q_), a.t_ * (1. - ratio) + b.t_ * ratio};
    }
    void AddNoise(double noise_trans, double noise_rot_angle) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<double> trans_dist(0.0, noise_trans);
        Eigen::Vector3d trans_noise(trans_dist(gen), trans_dist(gen), trans_dist(gen));
        std::normal_distribution<double> rot_dist(0.0, noise_rot_angle);
        Eigen::Vector3d omega(rot_dist(gen), rot_dist(gen), rot_dist(gen));
        double theta = omega.norm();
        Eigen::Matrix3d R_noise;
        if (theta < 1e-12) {
            R_noise = Eigen::Matrix3d::Identity();
        } else {
            Eigen::Vector3d axis = omega / theta;
            R_noise = Eigen::AngleAxisd(theta, axis).toRotationMatrix();
        }
        t_ += trans_noise;
        q_ = Eigen::Quaterniond(q_.toRotationMatrix() * R_noise);
    }

   private:
    Eigen::Quaterniond q_;
    Eigen::Vector3d t_;
};
}  // namespace faster_lio

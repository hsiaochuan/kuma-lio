#pragma once
#include <Eigen/Eigen>

#include "so3_math.h"
namespace faster_lio {
class S2 {
   public:
    S2() : vec_(Eigen::Vector3d::Zero()) {}
    S2(const Eigen::Vector3d &vec) : vec_(vec) {}
    Eigen::Matrix<double, 3, 2> B() const {
        Eigen::Vector3d vec_n = vec_.normalized();
        Eigen::Matrix<double, 3, 2> B;
        B(0, 0) = 1. - (vec_n.x() * vec_n.x()) / (1. + vec_n.z());
        B(0, 1) = -(vec_n.x() * vec_n.y()) / (1. + vec_n.z());
        B(1, 0) = -(vec_n.x() * vec_n.y()) / (1. + vec_n.z());
        B(1, 1) = 1. - (vec_n.y() * vec_n.y()) / (1. + vec_n.z());
        B(2, 0) = -vec_n.x();
        B(2, 1) = -vec_n.y();
        return B;
    }
    S2 operator+(const Eigen::Vector2d &vec) const { return S2(ExpMat(B() * vec) * vec_); }
    S2 &operator+=(const Eigen::Vector2d &vec) {
        vec_ = ExpMat(B() * vec) * vec_;
        return *this;
    }
    Eigen::Vector2d operator-(const S2 &b) const {
        Eigen::Quaterniond rot_quat = Eigen::Quaterniond::FromTwoVectors(vec_, b.vec_);
        Eigen::Vector3d rot_vec = LogQuat(rot_quat);
        return b.B().transpose() * rot_vec;
    }
    Eigen::Vector3d vec_;
};
}  // namespace faster_lio

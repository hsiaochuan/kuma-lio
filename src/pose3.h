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

#include <Eigen/Eigen>
namespace kuma_lio {

// A rigid transform T_ab (P_a = q P_b + t) with a unit quaternion; every constructor and product normalizes it.
struct Pose3 {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static Pose3 Identity();
    static Pose3 InValid();
    bool IsValid() const;
    // An invalid pose (all NaN, IsValid() is false), not the identity: use Pose3::Identity() for that.
    Pose3();

    // The matrix constructors take the rotation as it is: it must already be one (a non-orthonormal matrix
    // gives a rotation after normalization, but not the nearest one).
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

    // Slerp of the rotation and linear interpolation of the translation; ratio is clamped to [0, 1], so this
    // never extrapolates.
    static Pose3 Interpolate(const Pose3& a, const Pose3& b, double ratio);
    // Adds zero-mean Gaussian noise: noise_trans [m] per axis to t, a rotation of noise_rot_angle [rad] per axis
    // on the right (body frame).  A standard deviation <= 0 adds no noise of that kind.
    void AddNoise(double noise_trans, double noise_rot_angle);

    Eigen::Quaterniond q_;
    Eigen::Vector3d t_;
};

}  // namespace kuma_lio

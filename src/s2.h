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

#include "so3_math.h"
namespace kuma_lio {
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
}  // namespace kuma_lio

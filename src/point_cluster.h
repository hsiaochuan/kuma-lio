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
namespace kuma_lio{

class PointCluster {
public:
    Eigen::Matrix3d P;
    Eigen::Vector3d v;
    int N;

    PointCluster() {
        P.setZero();
        v.setZero();
        N = 0;
    }

    void SetZero() {
        P.setZero();
        v.setZero();
        N = 0;
    }

    void Push(const Eigen::Vector3d &vec) {
        N++;
        P += vec * vec.transpose();
        v += vec;
    }

    Eigen::Matrix3d Cov() const {
        Eigen::Vector3d center = v / N;
        return P / N - center * center.transpose();
    }

    Eigen::Vector3d Mean() const { return v / N; }

    PointCluster &operator+=(const PointCluster &sigv) {
        this->P += sigv.P;
        this->v += sigv.v;
        this->N += sigv.N;

        return *this;
    }
    PointCluster & operator-=(const PointCluster &sigv)
    {
        this->P -= sigv.P;
        this->v -= sigv.v;
        this->N -= sigv.N;

        return *this;
    }
    void Transform(const PointCluster &sigv, const Eigen::Matrix3d &R,
                   const Eigen::Vector3d &p) {
        N = sigv.N;
        v = R * sigv.v + N * p;
        Eigen::Matrix3d rp = R * sigv.v * p.transpose();
        P = R * sigv.P * R.transpose() + rp + rp.transpose() +
            N * p * p.transpose();
    }
};
}

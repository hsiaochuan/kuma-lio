//
// Created by hsiaochuan on 2026/04/02.
//

#ifndef POINT_CLUSTER_H
#define POINT_CLUSTER_H
#include <Eigen/Eigen>
namespace faster_lio{

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
#endif //POINT_CLUSTER_H

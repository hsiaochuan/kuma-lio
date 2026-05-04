//
// Created by hsiaochuan on 2026/03/23.
//

#ifndef FASTER_LIO_COST_FUNCTIONS_H
#define FASTER_LIO_COST_FUNCTIONS_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "pose3.h"

namespace faster_lio {
template <typename T>
using ConstEigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using ConstEigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;
template <typename T>
inline void EigenQuaternionLog(const T* eigen_quaternion, T* angle_axis) {
    // the qw in Eigen is the last one, the qw in ceres is the first one
    const T quaternion[4] = {eigen_quaternion[3], eigen_quaternion[0], eigen_quaternion[1], eigen_quaternion[2]};
    ceres::QuaternionToAngleAxis(quaternion, angle_axis);
}

struct TwoVectorRotationCostFunctor {
   public:
    explicit TwoVectorRotationCostFunctor(const Eigen::Vector3d& soure_vec, const Eigen::Vector3d& target_vec,
                                          double& weight)
        : weight(weight), source_vec(soure_vec), target_vec(target_vec) {}
    template <typename T>
    bool operator()(const T* const q_vec, T* res) const {
        ConstEigenQuaternionMap<T> q(q_vec);
        res[0] = T(weight) * (q * source_vec.cast<T>()).dot(target_vec.cast<T>());
        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector3d& soure_vec, const Eigen::Vector3d& target_vec,
                                       double& weight) {
        return new ceres::AutoDiffCostFunction<TwoVectorRotationCostFunctor, 1, 4>(
            new TwoVectorRotationCostFunctor(soure_vec, target_vec, weight));
    }

   private:
    Eigen::Vector3d source_vec;
    double weight;
    Eigen::Vector3d target_vec;
};
struct PointPlaneCostFunctor {
   public:
    PointPlaneCostFunctor(const Eigen::Vector3d& normal, const Eigen::Vector3d& center, const Eigen::Vector3d& point,
                          double& weight)
        : normal_(normal), center_(center), point(point), weight(weight) {}
    template <typename T>
    bool operator()(const T* const q_vec, const T* const t_vec, T* res) const {
        ConstEigenQuaternionMap<T> q(q_vec);
        ConstEigenVector3Map<T> t(t_vec);
        res[0] = T(weight) * normal_.cast<T>().dot(q * point.cast<T>() + t - center_.cast<T>());
        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector3d& normal, const Eigen::Vector3d& center,
                                       const Eigen::Vector3d& point, double& weight) {
        return new ceres::AutoDiffCostFunction<PointPlaneCostFunctor, 1, 4, 3>(
            new PointPlaneCostFunctor(normal, center, point, weight));
    }

   private:
    Eigen::Vector3d normal_;
    Eigen::Vector3d center_;
    Eigen::Vector3d point;
    double weight;
};

struct PointOnPlaneCostFunctor {
   public:
    PointOnPlaneCostFunctor(const Eigen::Vector3d& normal, const Eigen::Vector3d& center, const double& weight)
        : normal_(normal), center_(center), weight(weight) {}
    template <typename T>
    bool operator()(const T* const xyz, T* res) const {
        ConstEigenVector3Map<T> p(xyz);
        res[0] = T(weight) * normal_.cast<T>().dot(p - center_.cast<T>());
        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Vector3d& normal, const Eigen::Vector3d& center,
                                       const double& weight) {
        return new ceres::AutoDiffCostFunction<PointOnPlaneCostFunctor, 1, 3>(
            new PointOnPlaneCostFunctor(normal, center, weight));
    }

   private:
    Eigen::Vector3d normal_;
    Eigen::Vector3d center_;
    double weight;
};
struct RelativePoseCostFunctor {
   public:
    explicit RelativePoseCostFunctor(const Pose3& rel_pose, const double& pos_weight, const double& rot_weight)
        : rel_pose_inv(rel_pose.GetInverse()), pos_weight(pos_weight), rot_weight(rot_weight) {}

    template <typename T>
    bool operator()(const T* const q_world_i, const T* const t_world_i, const T* const q_world_j,
                    const T* const t_world_j, T* residuals_ptr) const {
        ConstEigenQuaternionMap<T> qi(q_world_i);
        ConstEigenVector3Map<T> ti(t_world_i);
        ConstEigenQuaternionMap<T> qj(q_world_j);
        ConstEigenVector3Map<T> tj(t_world_j);

        // r error
        const Eigen::Quaternion<T> R_error = rel_pose_inv.Quat().cast<T>() * qi.conjugate() * qj;
        EigenQuaternionLog(R_error.coeffs().data(), residuals_ptr);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> r_error(residuals_ptr);
        r_error = r_error * T(rot_weight);
        // t error
        // Rji * (Ri.t * tj - Ri.t * ti) + tji
        Eigen::Map<Eigen::Matrix<T, 3, 1>> t_error(residuals_ptr + 3);
        t_error = (rel_pose_inv.Quat().cast<T>() * (qi.conjugate() * (tj - ti)) + rel_pose_inv.Trans().cast<T>()) *
                  T(pos_weight);
        return true;
    }
    static ceres::CostFunction* Create(const Pose3& T_ij, const double& pos_weight, const double& rot_weight) {
        return new ceres::AutoDiffCostFunction<RelativePoseCostFunctor, 6, 4, 3, 4, 3>(
            new RelativePoseCostFunctor(T_ij, pos_weight, rot_weight));
    }

   private:
    const Pose3 rel_pose_inv;
    double pos_weight;
    double rot_weight;
};
}  // namespace faster_lio
#endif  // FASTER_LIO_COST_FUNCTIONS_H

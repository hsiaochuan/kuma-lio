//
// Created by hsiaochuan on 2026/03/23.
//

#ifndef FASTER_LIO_COST_FUNCTIONS_H
#define FASTER_LIO_COST_FUNCTIONS_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>
namespace faster_lio {
template <typename T>
using ConstEigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using ConstEigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;

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
}  // namespace faster_lio
#endif  // FASTER_LIO_COST_FUNCTIONS_H

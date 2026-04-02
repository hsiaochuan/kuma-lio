//
// Created by hsiaochuan on 2026/03/23.
//

#ifndef FASTER_LIO_BAREG_H
#define FASTER_LIO_BAREG_H
#include "cost_functions.h"
namespace faster_lio {

inline std::array<ceres::CostFunction *, 3> BaregCostFunctionCreate(int N_k, const Eigen::Matrix3d &cov_k,
                                                                    const Eigen::Vector3d &mean_k,
                                                                    const Eigen::Vector3d &normal,
                                                                    const Eigen::Vector3d &center, double scale) {
    std::array<ceres::CostFunction *, 3> cost_functions{};
    cost_functions.fill(nullptr);
    if (N_k) {
        double f3 = std::sqrt(N_k);
        f3 *= scale;
        cost_functions[0] = PointPlaneCostFunctor::Create(normal, center, mean_k, f3);
    }
    if (N_k > 3) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov_k);
        const Eigen::Matrix3d &vecs_k = solver.eigenvectors();
        const Eigen::Vector3d &vals_k = solver.eigenvalues();
        double f1 = std::sqrt(N_k) * std::sqrt(vals_k[2]);
        double f2 = std::sqrt(N_k) * std::sqrt(vals_k[1]);
        f1 *= scale;
        f2 *= scale;
        Eigen::Vector3d max_vec = vecs_k.col(2);
        if (!std::isnan(f1)) cost_functions[1] = TwoVectorRotationCostFunctor::Create(max_vec, normal, f1);

        Eigen::Vector3d mid_vec = vecs_k.col(1);
        if (!std::isnan(f2)) cost_functions[2] = TwoVectorRotationCostFunctor::Create(mid_vec, normal, f2);
    }
    return cost_functions;
}
}  // namespace faster_lio
#endif  // FASTER_LIO_BAREG_H

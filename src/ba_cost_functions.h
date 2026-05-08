//
// Created by hsiaochuan on 2026/04/29.
//

#ifndef FASTER_LIO_BA_COST_FUNCTIONS_H
#define FASTER_LIO_BA_COST_FUNCTIONS_H
#include <ceres/ceres.h>

namespace faster_lio {
template <typename T>
using ConstEigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using ConstEigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;

struct PinholeIntrinsicCostFunctor {
    PinholeIntrinsicCostFunctor(const Eigen::Vector2d& point2d, const double &weight) : measure_point(point2d), weight(weight) {}

    // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
    enum : uint8_t {
        OFFSET_FOCAL_X = 0,
        OFFSET_FOCAL_Y = 1,
        OFFSET_PRINCIPAL_POINT_X = 2,
        OFFSET_PRINCIPAL_POINT_Y = 3
    };

    /**
     * @param[in] cam_intrinsics: Camera intrinsics( focal, principal point [x,y] )
     * @param[in] cam_extrinsics: Camera parameterized using one block of 6 parameters [R;t]:
     *   - 3 for rotation(angle axis), 3 for translation
     * @param[in] pos_3dpoint
     * @param[out] out_residuals
     */
    template <typename T>
    bool operator()(const T* const cam_intrinsics, const T* const cam_q, const T* const cam_t,
                    const T* const pos_3dpoint, T* out_residuals) const {
        ConstEigenQuaternionMap<T> cam_R(cam_q);
        ConstEigenVector3Map<T> cam_translation(cam_t);
        ConstEigenVector3Map<T> point_w(pos_3dpoint);
        Eigen::Matrix<T, 3, 1> transformed_point = cam_R * point_w + cam_translation;
        const Eigen::Matrix<T, 2, 1> projected_point = transformed_point.hnormalized();

        //--
        // Apply intrinsic parameters
        //--

        const T& focal_x = cam_intrinsics[OFFSET_FOCAL_X];
        const T& focal_y = cam_intrinsics[OFFSET_FOCAL_Y];
        const T& principal_point_x = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
        const T& principal_point_y = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];

        // Apply focal length and principal point to get the final image coordinates

        // Compute and return the error is the difference between the predicted
        //  and observed position
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(out_residuals);
        residuals[0] = weight * (principal_point_x + projected_point.x() * focal_x - (T)measure_point.x());
        residuals[1] = weight * (principal_point_y + projected_point.y() * focal_y - (T)measure_point.y());
        return true;
    }

    static int num_residuals() { return 2; }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector2d& point2d, const double &weight) {
            return new ceres::AutoDiffCostFunction<PinholeIntrinsicCostFunctor, 2, 4, 4, 3, 3>(
                new PinholeIntrinsicCostFunctor(point2d, weight));
    }

    double weight;
    Eigen::Vector2d measure_point;
};
struct PinholeRadialIntrinsicCostFunctor {
    explicit PinholeRadialIntrinsicCostFunctor(const Eigen::Vector2d& point2d, const double &weight) : point2d(point2d), weight(weight) {}

    // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
    enum : uint8_t {
        OFFSET_FOCAL_X = 0,
        OFFSET_FOCAL_Y = 1,
        OFFSET_PRINCIPAL_POINT_X = 2,
        OFFSET_PRINCIPAL_POINT_Y = 3,
        OFFSET_DISTO_K1 = 4,
        OFFSET_DISTO_K2 = 5,
        OFFSET_DISTO_K3 = 6,
        OFFSET_DISTO_T1 = 7,
        OFFSET_DISTO_T2 = 8,
    };

    /**
     * @param[in] cam_intrinsics: Camera intrinsics( focal, principal point [x,y], k1, k2, k3, t1, t2 )
     * @param[in] cam_extrinsics: Camera parameterized using one block of 6 parameters [R;t]:
     *   - 3 for rotation(angle axis), 3 for translation
     * @param[in] pos_3dpoint
     * @param[out] out_residuals
     */
    template <typename T>
    bool operator()(const T* const cam_intrinsics, const T* const cam_q,const T* const cam_t, const T* const pos_3dpoint,
                    T* out_residuals) const {
        ConstEigenQuaternionMap<T> cam_R(cam_q);
        ConstEigenVector3Map<T> cam_translation(cam_t);
        ConstEigenVector3Map<T> point_w(pos_3dpoint);
        Eigen::Matrix<T, 3, 1> transformed_point = cam_R * point_w + cam_translation;
        const Eigen::Matrix<T, 2, 1> projected_point = transformed_point.hnormalized();

        const T& focal_x = cam_intrinsics[OFFSET_FOCAL_X];
        const T& focal_y = cam_intrinsics[OFFSET_FOCAL_Y];
        const T& principal_point_x = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
        const T& principal_point_y = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];
        const T& k1 = cam_intrinsics[OFFSET_DISTO_K1];
        const T& k2 = cam_intrinsics[OFFSET_DISTO_K2];
        const T& k3 = cam_intrinsics[OFFSET_DISTO_K3];
        const T& t1 = cam_intrinsics[OFFSET_DISTO_T1];
        const T& t2 = cam_intrinsics[OFFSET_DISTO_T2];

        // Apply distortion (xd,yd) = disto(x_u,y_u)
        const T x_u = projected_point.x();
        const T y_u = projected_point.y();
        const T r2 = projected_point.squaredNorm();
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;
        const T r_coeff = (1.0 + k1 * r2 + k2 * r4 + k3 * r6);
        const T t_x = t2 * (r2 + 2.0 * x_u * x_u) + 2.0 * t1 * x_u * y_u;
        const T t_y = t1 * (r2 + 2.0 * y_u * y_u) + 2.0 * t2 * x_u * y_u;

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(out_residuals);
        residuals[0] =  weight * (principal_point_x + (projected_point.x() * r_coeff + t_x) * focal_x - point2d[0]);
        residuals[1] =  weight * (principal_point_y + (projected_point.y() * r_coeff + t_y) * focal_y - point2d[1]);
        return true;
    }

    static int num_residuals() { return 2; }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector2d& point2d, const double &weight) {
        return new ceres::AutoDiffCostFunction<PinholeRadialIntrinsicCostFunctor, 2, 9, 4, 3, 3>(
            new PinholeRadialIntrinsicCostFunctor(point2d, weight));
    }

    Eigen::Vector2d point2d;
    double weight;
};
}  // namespace faster_lio
#endif  // FASTER_LIO_BA_COST_FUNCTIONS_H

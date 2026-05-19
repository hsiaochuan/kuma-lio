//
// Created by hsiaochuan on 2026/05/20.
//

#ifndef FASTER_LIO_EIGEN_TYPE_H
#define FASTER_LIO_EIGEN_TYPE_H
#include <Eigen/Eigen>
namespace faster_lio {
using MatX = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;

using Vec2 = Eigen::Vector2d;
using Vec2i = Eigen::Vector2i;
using Mat2 = Eigen::Matrix2d;
using Mat2X = Eigen::Matrix<double, 2, Eigen::Dynamic>;

using Vec3 = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Vec3i = Eigen::Vector3i;
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using Mat32 = Eigen::Matrix<double, 3, 2>;

using Vec4 = Eigen::Vector4d;
using Vec4f = Eigen::Vector4f;
using Mat4 = Eigen::Matrix<double, 4, 4>;
}
#endif  // FASTER_LIO_EIGEN_TYPE_H

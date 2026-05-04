#ifndef FASTER_LIO_SO3_MATH_H
#define FASTER_LIO_SO3_MATH_H

#include <Eigen/Core>
#include <cmath>

namespace faster_lio {

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> Hat(const Eigen::MatrixBase<Derived>& v) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> skew_mat;
    skew_mat.setZero();
    skew_mat(0, 1) = -v(2);
    skew_mat(0, 2) = +v(1);
    skew_mat(1, 2) = -v(0);
    skew_mat(1, 0) = +v(2);
    skew_mat(2, 0) = -v(1);
    skew_mat(2, 1) = +v(0);
    return skew_mat;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> ExpMat(const Eigen::MatrixBase<Derived>& v) {
    using std::cos;
    using std::sin;
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, 3, 3> R = Eigen::Matrix<Scalar, 3, 3>::Identity();
    Scalar theta = v.norm();
    if (theta > std::numeric_limits<Scalar>::epsilon()) {
        Eigen::Matrix<Scalar, 3, 1> v_normalized = v.normalized();
        R = cos(theta) * Eigen::Matrix<Scalar, 3, 3>::Identity() +
            (Scalar(1.0) - cos(theta)) * v_normalized * v_normalized.transpose() + sin(theta) * Hat(v_normalized);
        return R;
    } else {
        return R;
    }
}
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 1> LogMat(const Eigen::MatrixBase<Derived>& R) {
    using Scalar = typename Derived::Scalar;
    using std::atan2;
    Eigen::Quaternion<Scalar> q(R);
    if (q.norm() <= std::numeric_limits<Scalar>::epsilon()) {
        Eigen::Matrix<Scalar, 3, 1> phi_u;
        phi_u = Scalar(2.0) * q.vec() / q.w() * (Scalar(1.0) - q.vec().squaredNorm() / (Scalar(3.0) * q.w() * q.w()));
        return phi_u;
    }
    Scalar norm_vec = q.vec().norm();
    Scalar phi =
        Scalar(2.0) * ((q.w() < Scalar(0.0)) ? Scalar(atan2(-norm_vec, -q.w())) : Scalar(atan2(norm_vec, q.w())));
    Eigen::Matrix<Scalar, 3, 1> u;
    u = q.vec().normalized();
    return phi * u;
}
template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 1> LogQuat(const Eigen::Quaternion<Scalar>& quat) {
    using std::abs;
    using std::atan2;
    using std::sqrt;

    Scalar squared_v_norm = quat.vec().squaredNorm();
    Scalar w = quat.w();

    Scalar theta;

    if (squared_v_norm < Eigen::NumTraits<Scalar>::epsilon() * Eigen::NumTraits<Scalar>::epsilon()) {
        // If quaternion is normalized and n=0, then w should be 1;
        // w=0 should never happen here!
        if (abs(w) < Eigen::NumTraits<Scalar>::epsilon()) throw std::runtime_error("Quaternion should be normalized!");
        Scalar squared_w = w * w;
        theta = Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_v_norm) / (w * squared_w);
    } else {
        Scalar n = sqrt(squared_v_norm);
        Scalar atan_nbyw = (w < Scalar(0)) ? Scalar(atan2(-n, -w)) : Scalar(atan2(n, w));
        theta = Scalar(2) * atan_nbyw / n;
    }

    return theta * quat.vec();
}

template <typename Derived>
inline Eigen::Quaternion<typename Derived::Scalar> ExpQuat(const Eigen::MatrixBase<Derived>& omega) {
    eigen_assert(omega.size() == 3u);
    using std::cos;
    using std::sin;
    using std::sqrt;

    using Scalar = typename Derived::Scalar;
    Scalar theta_sq = omega.squaredNorm();
    Scalar img;
    Scalar real;

    if (theta_sq < Eigen::NumTraits<Scalar>::epsilon() * Eigen::NumTraits<Scalar>::epsilon()) {
        Scalar theta_po4 = theta_sq * theta_sq;
        img = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq + Scalar(1.0 / 3840.0) * theta_po4;
        real = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq + Scalar(1.0 / 384.0) * theta_po4;
    } else {
        Scalar theta = sqrt(theta_sq);
        Scalar half_theta = Scalar(0.5) * theta;
        Scalar sin_half_theta = sin(half_theta);
        img = sin_half_theta / theta;
        real = cos(half_theta);
    }

    return Eigen::Quaternion<Scalar>(real, img * omega.x(), img * omega.y(), img * omega.z()).normalized();
}


inline Eigen::Vector3d EulerZYX(const Eigen::Matrix3d& rot) {
    Eigen::Vector3d result;
    result.x() = std::atan2(rot(1, 0), rot(0, 0));
    result.y() = std::asin(-rot(2, 0));
    result.z() = std::atan2(rot(2, 1), rot(2, 2));
    return result;
}
inline Eigen::Matrix3d EulerZYXToRot(const Eigen::Vector3d& euler) {
    Eigen::Matrix3d z_rot = Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Matrix3d y_rot = Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d x_rot = Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitX()).toRotationMatrix();
    return z_rot * y_rot * x_rot;
}
}  // namespace faster_lio
#endif
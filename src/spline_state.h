#pragma once
#include <glog/logging.h>

#include <Eigen/Eigen>
#include "eigen_type.h"
#include "so3_math.h"
namespace faster_lio {
template <class MatT>
struct JacobianStruct {
    size_t start_idx;
    std::vector<MatT> d_val_d_knot;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef JacobianStruct<double> Jacobian;
typedef JacobianStruct<Eigen::Matrix<double, 4, 3>> Jacobian43;
typedef JacobianStruct<Eigen::Matrix3d> Jacobian33;

template <typename Derived>
inline void dRotInv(const Eigen::MatrixBase<Derived>& v,
                  const Eigen::Quaternion<typename Derived::Scalar>& q,
                  Eigen::Matrix<typename Derived::Scalar, 3, 4>& J) {
    using Scalar = typename Derived::Scalar;
    Scalar qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
    Scalar v1 = v(0), v2 = v(1), v3 = v(2);
    Eigen::Matrix<Scalar, 3, 1> vec;
    vec << (v1 * qw + v2 * qz - v3 * qy) * Scalar(2),
           (v2 * qw - v1 * qz + v3 * qx) * Scalar(2),
           (v3 * qw + v1 * qy - v2 * qx) * Scalar(2);
    Scalar tmp = (v1 * qx + v2 * qy + v3 * qz) * Scalar(2);
    J << vec(0), tmp, -vec(2), vec(1),
         vec(1), vec(2), tmp, -vec(0),
         vec(2), -vec(1), vec(0), tmp;
}


template <typename Derived>
inline void dRot(const Eigen::MatrixBase<Derived>& v,
                     const Eigen::Quaternion<typename Derived::Scalar>& q,
                     Eigen::Matrix<typename Derived::Scalar, 3, 4>& J) {
    using Scalar = typename Derived::Scalar;
    Scalar qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
    Scalar v1 = v(0), v2 = v(1), v3 = v(2);
    Eigen::Matrix<Scalar, 3, 1> vec;
    vec << (v1 * qw - v2 * qz + v3 * qy) * Scalar(2),
           (v2 * qw + v1 * qz - v3 * qx) * Scalar(2),
           (v3 * qw - v1 * qy + v2 * qx) * Scalar(2);
    Scalar tmp = (v1 * qx + v2 * qy + v3 * qz) * Scalar(2);
    J << vec(0), tmp, vec(2), -vec(1),
         vec(1), -vec(2), tmp, vec(0),
         vec(2), vec(1), -vec(0), tmp;
}

template <typename Derived>
inline Eigen::Quaternion<typename Derived::Scalar> ExpQuatHalf(const Eigen::MatrixBase<Derived>& v) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
    using Scalar = typename Derived::Scalar;
    Scalar n = v.norm();
    Scalar sinc;
    if (n < Scalar(1e-8)) {
        sinc = Scalar(1) - n * n / Scalar(6);
    } else {
        sinc = std::sin(n) / n;
    }
    Eigen::Quaternion<Scalar> q;
    q.w() = std::cos(n);
    q.vec() = sinc * v;
    return q;
}

template <typename Derived>
inline void dExpQuatHalf(const Eigen::MatrixBase<Derived>& v,
                         Eigen::Quaternion<typename Derived::Scalar>& q,
                         Eigen::Matrix<typename Derived::Scalar, 4, 3>& J) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
    using Scalar = typename Derived::Scalar;
    Scalar v_norm = v.norm();
    if (v_norm == Scalar(0)) {
        J.row(0).setZero();
        J.template bottomRows<3>().setIdentity();
        q.setIdentity();
        return;
    }
    Scalar sinc = (v_norm < Scalar(1e-8)) ? Scalar(1) - v_norm * v_norm / Scalar(6) : std::sin(v_norm) / v_norm;
    q.w() = std::cos(v_norm);
    q.vec() = sinc * v;
    Scalar v1 = v(0), v2 = v(1), v3 = v(2);
    Scalar tmp = (q.w() - sinc) / (v_norm * v_norm);
    Scalar tmp_v1 = tmp * v1;
    Scalar tmp_v2 = tmp * v2;
    Scalar tmp_v11 = tmp_v1 * v1 + sinc;
    Scalar tmp_v12 = tmp_v1 * v2;
    Scalar tmp_v13 = tmp_v1 * v3;
    Scalar tmp_v22 = tmp_v2 * v2 + sinc;
    Scalar tmp_v23 = tmp_v2 * v3;
    Scalar tmp_v33 = tmp * v3 * v3 + sinc;
    J << -v1 * sinc, -v2 * sinc, -v3 * sinc,
         tmp_v11, tmp_v12, tmp_v13,
         tmp_v12, tmp_v22, tmp_v23,
         tmp_v13, tmp_v23, tmp_v33;
}
class SplineState {
   public:
    void init(int64_t dt_ns_, int64_t start_t_ns_,
              const Eigen::Quaterniond& q0 = Eigen::Quaterniond::Identity());
    Eigen::Isometry3d Interpolate(const std::int64_t& time_ns);
    int64_t maxTimeNs() const;
    void addOneStateKnot(const Eigen::Vector3d& pos, const Eigen::Vector3d& ort_del);
    Eigen::Matrix<double, 24, 1> getRCPs();
    template <typename _KnotT, int Derivative = 0>
    _KnotT itpEuclidean(int64_t t_ns, const Eigen::aligned_deque<_KnotT>& knots, Jacobian* J = nullptr) const {
        double u;
        int64_t idx0;
        int idx_r;
        std::array<_KnotT, 4> cps;
        prepareInterpolation(t_ns, knots, idx0, u, cps, idx_r);
        Eigen::Vector4d p, coeff;
        baseCoeffsWithTime<Derivative>(p, u);
        coeff = pow_inv_dt[Derivative] * (blending_matrix * p);
        _KnotT res_out = coeff[0] * cps.at(0) + coeff[1] * cps.at(1) + coeff[2] * cps.at(2) + coeff[3] * cps.at(3);
        if (J) {
            J->d_val_d_knot.resize(4);
            for (int i = 0; i < 4; i++) {
                J->d_val_d_knot[i] = coeff[i];
            }
            J->start_idx = idx0;
        }
        return res_out;
    }
    template <int Derivative = 0>
    Eigen::Vector3d itpPosition(int64_t time_ns, Jacobian* J = nullptr) const {
        return itpEuclidean<Eigen::Vector3d, Derivative>(time_ns, t_knots, J);
    }

    void itpQuaternion(int64_t t_ns, Eigen::Quaterniond* q_out = nullptr, Eigen::Vector3d* w_out = nullptr,
                       Jacobian43* J_q = nullptr, Jacobian33* J_w = nullptr) const;
    static const Eigen::Matrix4d blending_matrix;
    static const Eigen::Matrix4d base_coefficients;
    static const Eigen::Matrix4d cumulative_blending_matrix;
    int64_t dt_ns;
    double inv_dt;
    std::array<double, 4> pow_inv_dt;
    int64_t num_knot;
    int64_t start_t_ns;

    Eigen::aligned_deque<Eigen::Vector3d> t_knots;
    Eigen::aligned_deque<Eigen::Quaterniond> q_knots;
    Eigen::aligned_deque<Eigen::Vector3d> ort_delta;
    Eigen::Quaterniond q_idle;
    template <typename _KnotT>
    void prepareInterpolation(int64_t t_ns, const Eigen::aligned_deque<_KnotT>& knots, int64_t& idx0, double& u,
                              std::array<_KnotT, 4>& cps, int& idx_r) const {
        int64_t t_ns_rel = t_ns - start_t_ns;
        int idx_l = floor(double(t_ns_rel) / double(dt_ns));
        idx_r = idx_l + 1;
        idx0 = idx_l - 2;
        CHECK(idx0 >= 0);
        CHECK(idx_r >= 3);
        CHECK(idx0 + 4 <= static_cast<int64_t>(knots.size()))
            << "interpolation time " << t_ns << " is at/beyond the newest knot";
        for (int i = 0; i < 4; i++) {
            cps[i] = knots[idx0 + i];
        }
        u = (t_ns - start_t_ns - idx_l * dt_ns) / double(dt_ns);
    }

    template <int Derivative, class Derived>
    static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const, double t) {
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 4);
        Eigen::MatrixBase<Derived>& res = const_cast<Eigen::MatrixBase<Derived>&>(res_const);
        res.setZero();
        res[Derivative] = base_coefficients(Derivative, Derivative);
        double ti = t;
        for (int j = Derivative + 1; j < 4; j++) {
            res[j] = base_coefficients(Derivative, j) * ti;
            ti = ti * t;
        }
    }

    template <bool _Cumulative = false>
    static Eigen::Matrix4d computeBlendingMatrix() {
        Eigen::Matrix4d m;
        m.setZero();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sum = 0;
                for (int s = j; s < 4; ++s) {
                    sum += std::pow(-1.0, s - j) * binomialCoefficient(4, s - j) * std::pow(4 - s - 1.0, 4 - 1.0 - i);
                }
                m(j, i) = binomialCoefficient(3, 3 - i) * sum;
            }
        }
        if (_Cumulative) {
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    m.row(i) += m.row(j);
                }
            }
        }
        uint64_t factorial = 1;
        for (int i = 2; i < 4; ++i) {
            factorial *= i;
        }
        return m / factorial;
    }

    constexpr static inline uint64_t binomialCoefficient(uint64_t n, uint64_t k) {
        if (k > n) return 0;
        uint64_t r = 1;
        for (uint64_t d = 1; d <= k; ++d) {
            r *= n--;
            r /= d;
        }
        return r;
    }

    static Eigen::Matrix4d computeBaseCoefficients();
};
}  // namespace faster_lio

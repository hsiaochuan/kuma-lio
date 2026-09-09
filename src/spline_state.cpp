#include "spline_state.h"

namespace faster_lio {

const Eigen::Matrix4d SplineState::base_coefficients = SplineState::computeBaseCoefficients();
const Eigen::Matrix4d SplineState::blending_matrix = SplineState::computeBlendingMatrix();
const Eigen::Matrix4d SplineState::cumulative_blending_matrix = SplineState::computeBlendingMatrix<true>();

void SplineState::init(int64_t dt_ns_, int64_t start_t_ns_, const Eigen::Quaterniond& q0) {
    dt_ns = dt_ns_;
    start_t_ns = start_t_ns_;
    num_knot = 0;
    inv_dt = 1e9 / dt_ns;
    pow_inv_dt[0] = 1.0;
    pow_inv_dt[1] = inv_dt;
    pow_inv_dt[2] = inv_dt * inv_dt;
    pow_inv_dt[3] = pow_inv_dt[2] * inv_dt;
    t_knots.clear();
    q_knots.clear();
    ort_delta.clear();
    q_idle = q0;
}

Eigen::Isometry3d SplineState::Interpolate(const std::int64_t& time_ns) {
    Vec3 pos = itpPosition(time_ns, nullptr);
    Eigen::Quaterniond q;
    itpQuaternion(time_ns, &q, nullptr, nullptr, nullptr);
    Eigen::Isometry3d iso;
    iso.linear() = q.toRotationMatrix();
    iso.translation() = pos;
    return iso;
}

int64_t SplineState::maxTimeNs() const {
    if (num_knot == 0)
        return start_t_ns - 1;
    return start_t_ns + (num_knot - 1) * dt_ns;
}

void SplineState::addOneStateKnot(const Eigen::Vector3d& pos, const Eigen::Vector3d& ort_del) {
    t_knots.push_back(pos);
    ort_delta.push_back(ort_del);
    Eigen::Quaterniond q;
    Eigen::Quaterniond q_del = faster_lio::ExpQuatHalf(ort_del);
    if (num_knot == 0) q = q_idle * q_del;
    else
        q = q_knots.back() * q_del;
    q_knots.push_back(q);
    num_knot++;
}

Eigen::Matrix<double, 24, 1> SplineState::getRCPs() {
    Eigen::Matrix<double, 24, 1> RCPs;
    for (int i = 0; i < 4; i++) {
        RCPs.block<3, 1>(i * 6, 0) = t_knots[num_knot - 4 + i];
        RCPs.block<3, 1>(i * 6 + 3, 0) = ort_delta[num_knot - 4 + i];
    }
    return RCPs;
}

void SplineState::itpQuaternion(int64_t t_ns, Eigen::Quaterniond* q_out, Eigen::Vector3d* w_out,
                                 Jacobian43* J_q, Jacobian33* J_w) const {
    double u;
    int64_t idx0;
    int idx_r;
    std::array<Eigen::Vector3d, 4> t_delta;
    prepareInterpolation(t_ns, ort_delta, idx0, u, t_delta, idx_r);
    Eigen::Vector4d p;
    Eigen::Vector4d coeff;
    baseCoeffsWithTime<0>(p, u);
    coeff = cumulative_blending_matrix * p;

    Eigen::Quaterniond cp0;
    CHECK(idx0 >= 0);
    if (idx0 == 0) cp0 = q_idle;
    else if (idx0 > 0)
        cp0 = q_knots[idx0 - 1];

    Eigen::Vector3d t_delta_scale[4];
    Eigen::Quaterniond q_delta_scale[4];
    Eigen::Quaterniond q_itps[4];
    Eigen::Vector3d w_itps[4];
    Eigen::Vector4d dcoeff;
    if (J_q || J_w) {
        Eigen::Matrix<double, 4, 3> dexp_dt[4];
        t_delta_scale[0] = t_delta[0] * coeff[0];
        t_delta_scale[1] = t_delta[1] * coeff[1];
        t_delta_scale[2] = t_delta[2] * coeff[2];
        t_delta_scale[3] = t_delta[3] * coeff[3];
        dExpQuatHalf(t_delta_scale[0], q_delta_scale[0], dexp_dt[0]);
        dExpQuatHalf(t_delta_scale[1], q_delta_scale[1], dexp_dt[1]);
        dExpQuatHalf(t_delta_scale[2], q_delta_scale[2], dexp_dt[2]);
        dExpQuatHalf(t_delta_scale[3], q_delta_scale[3], dexp_dt[3]);
        int size_J = 4;
        Eigen::Quaterniond q_r_all[4];
        q_r_all[3] = Eigen::Quaterniond::Identity();
        for (int i = 2; i >= 0; i--) {
            q_r_all[i] = q_delta_scale[i + 1] * q_r_all[i + 1];
        }
        if (J_q) {
            q_itps[0] = cp0 * q_delta_scale[0];
            q_itps[1] = q_itps[0] * q_delta_scale[1];
            q_itps[2] = q_itps[1] * q_delta_scale[2];
            q_itps[3] = q_itps[2] * q_delta_scale[3];
            q_itps[3].normalize();
            *q_out = q_itps[3];
            Eigen::Matrix4d Q_l_all[4];
            Q_l_all[0] = Qleft(cp0);
            Q_l_all[1] = Qleft(q_itps[0]);
            Q_l_all[2] = Qleft(q_itps[1]);
            Q_l_all[3] = Qleft(q_itps[2]);
            J_q->d_val_d_knot.resize(size_J);
            J_q->start_idx = idx0;
            for (int i = size_J - 1; i >= 0; i--) {
                Eigen::Matrix4d Q_r_all;
                Q_r_all = Qright(q_r_all[i]);
                J_q->d_val_d_knot[i].noalias() = coeff[i] * Q_r_all * Q_l_all[i] * dexp_dt[i];
            }
        }
        if (J_w) {
            baseCoeffsWithTime<1>(p, u);
            dcoeff = inv_dt * cumulative_blending_matrix * p;
            w_itps[0].setZero();
            w_itps[1] = 2 * dcoeff[1] * t_delta[1];
            w_itps[2] = q_delta_scale[2].inverse() * w_itps[1] + 2 * dcoeff[2] * t_delta[2];
            w_itps[3] = q_delta_scale[3].inverse() * w_itps[2] + 2 * dcoeff[3] * t_delta[3];
            *w_out = w_itps[3];
            Eigen::Matrix<double, 3, 4> drot_dq[2];
            drot(w_itps[1], q_delta_scale[2], drot_dq[0]);
            drot(w_itps[2], q_delta_scale[3], drot_dq[1]);
            J_w->d_val_d_knot.resize(size_J);
            J_w->start_idx = idx0;
            J_w->d_val_d_knot[0].setZero();
            J_w->d_val_d_knot[1] = 2 * dcoeff[1] * q_delta_scale[3].inverse().toRotationMatrix() *
                                   q_delta_scale[2].inverse().toRotationMatrix();
            Eigen::Matrix3d tmp = coeff[2] * drot_dq[0] * dexp_dt[2];
            J_w->d_val_d_knot[2] =
                q_delta_scale[3].inverse().toRotationMatrix() * (tmp + 2 * dcoeff[2] * Eigen::Matrix3d::Identity());
            J_w->d_val_d_knot[3] = coeff[3] * drot_dq[1] * dexp_dt[3] + 2 * dcoeff[3] * Eigen::Matrix3d::Identity();
        }
    } else {
        t_delta_scale[0] = t_delta[0] * coeff[0];
        t_delta_scale[1] = t_delta[1] * coeff[1];
        t_delta_scale[2] = t_delta[2] * coeff[2];
        t_delta_scale[3] = t_delta[3] * coeff[3];
        q_delta_scale[0] = ExpQuatHalf(t_delta_scale[0]);
        q_delta_scale[1] = ExpQuatHalf(t_delta_scale[1]);
        q_delta_scale[2] = ExpQuatHalf(t_delta_scale[2]);
        q_delta_scale[3] = ExpQuatHalf(t_delta_scale[3]);
        if (q_out) {
            q_itps[0] = cp0 * q_delta_scale[0];
            q_itps[1] = q_itps[0] * q_delta_scale[1];
            q_itps[2] = q_itps[1] * q_delta_scale[2];
            q_itps[3] = q_itps[2] * q_delta_scale[3];
            q_itps[3].normalize();
            *q_out = q_itps[3];
        }
        if (w_out) {
            baseCoeffsWithTime<1>(p, u);
            dcoeff = inv_dt * cumulative_blending_matrix * p;

            w_itps[0].setZero();
            w_itps[1] = 2 * dcoeff[1] * t_delta[1];
            w_itps[2] = q_delta_scale[2].inverse() * w_itps[1] + 2 * dcoeff[2] * t_delta[2];
            w_itps[3] = q_delta_scale[3].inverse() * w_itps[2] + 2 * dcoeff[3] * t_delta[3];
            *w_out = w_itps[3];
        }
    }
}

Eigen::Matrix4d SplineState::computeBaseCoefficients() {
    Eigen::Matrix4d base_coeff;
    base_coeff.setZero();
    base_coeff.row(0).setOnes();
    int order = 3;
    for (int n = 1; n < 4; n++) {
        for (int i = 3 - order; i < 4; i++) {
            base_coeff(n, i) = (order - 3 + i) * base_coeff(n - 1, i);
        }
        order--;
    }
    return base_coeff;
}

}  // namespace faster_lio

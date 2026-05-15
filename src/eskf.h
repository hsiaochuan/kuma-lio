#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>

#include "so3_math.h"
#include "state_point.h"

namespace faster_lio {

struct ImuInput {
    Vec3 acc = Vec3::Zero();
    Vec3 gyro = Vec3::Zero();
};

struct LidarObservation {
    Eigen::MatrixXd H;
    Eigen::VectorXd r;
    bool valid = false;
};

class IESKF {
   public:
    using NoiseMatrix = Eigen::Matrix<double, 12, 12>;
    static void Predict(double dt, const NoiseMatrix &Q, const ImuInput &input, StatePoint& state) {
        if (dt == 0.0) {
            return;
        }

        const Vec3 omega = input.gyro - state.bias_g;
        const Vec3 acc = input.acc - state.bias_a;
        const Mat3 R = state.rot.toRotationMatrix();
        const Vec3 acc_world = R * acc + state.gravity;

        state.pos += state.vel_end * dt + 0.5 * acc_world * dt * dt;
        state.vel_end += acc_world * dt;
        state.rot = (state.rot * ExpQuat(omega * dt)).normalized();

        StatePoint::MatrixN F = StatePoint::MatrixN::Identity();
        Eigen::Matrix<double, StatePoint::STATE_DOF, 12> G =
            Eigen::Matrix<double, StatePoint::STATE_DOF, 12>::Zero();

        F.block<3, 3>(StatePoint::ROT, StatePoint::ROT) = ExpMat(omega * (-dt));
        F.block<3, 3>(StatePoint::ROT, StatePoint::BIAS_G) = -Mat3::Identity() * dt;
        F.block<3, 3>(StatePoint::POS, StatePoint::VEL) = Mat3::Identity() * dt;
        F.block<3, 3>(StatePoint::VEL, StatePoint::ROT) = -R * Hat(acc) * dt;
        F.block<3, 3>(StatePoint::VEL, StatePoint::BIAS_A) = -R * dt;

        // Mat32 B = state_.gravity.B();
        // Mat3 g_x = Hat(state_.gravity.vec_);
        // F.block<3, 2>(State::VEL, State::GRAVITY) = g_x * B * dt;
        // F.block<2, 2>(State::GRAVITY, State::GRAVITY) = -(B.transpose() * g_x * g_x * B) * (1. / (GRAVITY_NORM * GRAVITY_NORM));
        F.block<3,3>(StatePoint::VEL, StatePoint::GRAVITY) = Mat3::Identity() * dt;


        G.block<3, 3>(StatePoint::ROT, 0) = -Mat3::Identity() * dt;
        G.block<3, 3>(StatePoint::VEL, 3) = -R * dt;
        G.block<3, 3>(StatePoint::BIAS_G, 6) = -Mat3::Identity() * dt;
        G.block<3, 3>(StatePoint::BIAS_A, 9) = -Mat3::Identity() * dt;

        const StatePoint::MatrixN Qd = G * Q * G.transpose();
        state.cov = F * state.cov * F.transpose() + Qd;
    }

    static bool IterativeUpdate(
        const std::function<bool(const StatePoint &, bool, LidarObservation &)> &build_obs,
        double measure_noise,
        int max_iter, StatePoint& state) {
        StatePoint old_state = state;

        bool if_converge, if_stop = false;
        for (int iter = 0; iter < max_iter; ++iter) {
            LidarObservation obs;
            const bool recompute = true;
            if (!build_obs(state, recompute, obs)) {
                LOG(WARNING) << "Failed to build observation!";
                return false;
            }
            if (!obs.valid || obs.H.rows() == 0) {
                LOG(WARNING) << "Invalid observation!";
                return false;
            }

            if_stop = false;
            if_converge = false;
            const Mat &H = obs.H;
            const Vec &r = obs.r;

            const Mat HTRinv = H.transpose() * (1. / measure_noise);
            const Mat K = (HTRinv * H + state.cov.inverse()).inverse() * HTRinv;
            StatePoint::VectorN vec = old_state - state;
            StatePoint::VectorN solution = K * (r - H * vec ) + vec;
            state += solution;

            auto rot_add = solution.block<3, 1>(StatePoint::ROT, 0);
            auto pos_add = solution.block<3, 1>(StatePoint::POS, 0);
            if ((rot_add.norm() * 57.3 < 0.01) && (pos_add.norm() * 100 < 0.015)) {
                if_converge = true;
            }

            if (iter == max_iter - 1) {
                static StatePoint::MatrixN state_iden_mat = StatePoint::MatrixN::Identity();
                state.cov = (state_iden_mat - K * H) * state.cov;
                if_stop = true;
            }
            if (if_stop) break;
        }
        return true;
    }
};

}  // namespace faster_lio


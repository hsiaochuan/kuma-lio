//
// Created by hsiaochuan on 2026/09/03.
//

#ifndef FASTER_LIO_GNSSPSRDOPPFACTOR_H
#define FASTER_LIO_GNSSPSRDOPPFACTOR_H

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "so3_math.h"
#include "rtklib_.h"

#include <Eigen/Dense>
#include <cmath>

namespace faster_lio {
class GnssPsrFactor : public gtsam::NoiseModelFactor4<gtsam::Vector3, gtsam::Vector4, gtsam::Vector3, gtsam::Rot3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    Eigen::Vector3d le_arm_local{Eigen::Vector3d::Zero()};
    Eigen::Vector3d sv_pos{Eigen::Vector3d::Zero()};
    double svdt{0.0};
    double ion_delay{0.0};
    double tro_delay{0.0};
    double noise_var{1.0};
    double psr_measure{0.0};
    int sys_idx{0};


    GnssPsrFactor(gtsam::Key j1, gtsam::Key j2, gtsam::Key j3, gtsam::Key j4,
                  const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor4<gtsam::Vector3, gtsam::Vector4, gtsam::Vector3, gtsam::Rot3>(model, j1, j2, j3, j4){}

    ~GnssPsrFactor() override = default;

    gtsam::Vector evaluateError(const gtsam::Vector3& pos, const gtsam::Vector4& dt, const gtsam::Vector3& ext_p,
                                const gtsam::Rot3& ext_R, boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none,
                                boost::optional<gtsam::Matrix&> H4 = boost::none) const override {
        const Eigen::Vector3d ref_ecef = ext_p;
        const Eigen::Vector3d local_pos = le_arm_local + pos;

        const Eigen::Matrix3d R_ecef_local = ext_R.matrix();
        const Eigen::Vector3d P_ecef = R_ecef_local * local_pos + ref_ecef;

        double e[3] = {0.0, 0.0, 0.0};
        const double range = geodist(sv_pos.data(), P_ecef.data(), e);
        if (range <= 0.0) {
            if (H1) *H1 = gtsam::Matrix::Zero(1, 3);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 4);
            if (H3) *H3 = gtsam::Matrix::Zero(1, 3);
            if (H4) *H4 = gtsam::Matrix::Zero(1, 3);
            return gtsam::Vector1::Zero();
        }
        const Eigen::Vector3d rcv2sat_unit(e[0], e[1], e[2]);

        const double pr_weight = 1.0 / std::sqrt(std::max(0.01, noise_var));

        const double psr_estimated = range + dt[sys_idx] - CLIGHT * svdt + ion_delay + tro_delay;

        gtsam::Vector1 residual;
        residual[0] = (psr_estimated - psr_measure) * pr_weight;

        if (!H1 && !H2 && !H3 && !H4) return residual;
        const Eigen::RowVector3d sagnac_p(-OMGE / CLIGHT * sv_pos(1), OMGE / CLIGHT * sv_pos(0), 0.0);
        const Eigen::RowVector3d d_psr_d_P = -rcv2sat_unit.transpose() + sagnac_p;

        if (H1) {
            *H1 = gtsam::Matrix::Zero(1, 3);
            H1->block<1, 3>(0, 0) = d_psr_d_P * R_ecef_local * pr_weight;
        }
        if (H2) {
            *H2 = gtsam::Matrix::Zero(1, 4);
            (*H2)(0, sys_idx) = pr_weight;
        }
        if (H3) {
            *H3 = gtsam::Matrix::Zero(1, 3);
            H3->block<1, 3>(0, 0) = d_psr_d_P * pr_weight;
        }
        if (H4) {
            const Eigen::Matrix3d d_P = -R_ecef_local * Hat(local_pos);
            *H4 = gtsam::Matrix::Zero(1, 3);
            H4->block<1, 3>(0, 0) = d_psr_d_P * d_P * pr_weight;
        }
        return residual;
    }

};

}  // namespace faster_lio

#endif  // FASTER_LIO_GNSSPSRDOPPFACTOR_H

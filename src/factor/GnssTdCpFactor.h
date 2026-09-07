//
// Created by hsiaochuan on 2026/09/03.
//

#ifndef FASTER_LIO_GNSSTDCPFACTOR_H
#define FASTER_LIO_GNSSTDCPFACTOR_H

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "rtklib_.h"

#include <Eigen/Dense>

namespace faster_lio {

class GnssTdCpFactor : public gtsam::NoiseModelFactor6<gtsam::Vector3, gtsam::Vector3, gtsam::Vector4, gtsam::Vector4,
                                                       gtsam::Vector3, gtsam::Rot3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Params {
        Eigen::Vector3d le_arm_local{Eigen::Vector3d::Zero()};  ///< IMU->antenna lever arm, local frame
        Eigen::Vector3d sv_pos_i{Eigen::Vector3d::Zero()};     ///< satellite position at epoch i, ECEF (m)
        Eigen::Vector3d sv_pos_j{Eigen::Vector3d::Zero()};     ///< satellite position at epoch j, ECEF (m)
        double cp_measured{0.0};                               ///< time-differenced carrier phase (m)
        double cp_weight{1.0};                                 ///< residual weight (1/sigma)
        int sys_idx{0};                                        ///< index into `dt` for this constellation
    };

    GnssTdCpFactor(gtsam::Key j1, gtsam::Key j2, gtsam::Key j3, gtsam::Key j4, gtsam::Key j5, gtsam::Key j6,
                   const Params& p, const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor6<gtsam::Vector3, gtsam::Vector3, gtsam::Vector4, gtsam::Vector4, gtsam::Vector3,
                                   gtsam::Rot3>(model, j1, j2, j3, j4, j5, j6),
          p_(p) {}

    ~GnssTdCpFactor() override = default;

    gtsam::Vector evaluateError(const gtsam::Vector3& pos_i, const gtsam::Vector3& pos_j,
                                const gtsam::Vector4& dt_i, const gtsam::Vector4& dt_j, const gtsam::Vector3& ext_p,
                                const gtsam::Rot3& ext_R, boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none,
                                boost::optional<gtsam::Matrix&> H4 = boost::none,
                                boost::optional<gtsam::Matrix&> H5 = boost::none,
                                boost::optional<gtsam::Matrix&> H6 = boost::none) const override {
        const Eigen::Vector3d ref_ecef = ext_p;
        const Eigen::Vector3d local_pos_i = p_.le_arm_local + pos_i;
        const Eigen::Vector3d local_pos_j = p_.le_arm_local + pos_j;

        const Eigen::Matrix3d R_ecef_local = ext_R.matrix();
        const Eigen::Vector3d P_ecef_i = R_ecef_local * local_pos_i + ref_ecef;
        const Eigen::Vector3d P_ecef_j = R_ecef_local * local_pos_j + ref_ecef;

        double e_i[3] = {0.0, 0.0, 0.0}, e_j[3] = {0.0, 0.0, 0.0};
        const double range_i = geodist(p_.sv_pos_i.data(), P_ecef_i.data(), e_i);
        const double range_j = geodist(p_.sv_pos_j.data(), P_ecef_j.data(), e_j);
        if (range_i <= 0.0 || range_j <= 0.0) {
            if (H1) *H1 = gtsam::Matrix::Zero(1, 3);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 3);
            if (H3) *H3 = gtsam::Matrix::Zero(1, 4);
            if (H4) *H4 = gtsam::Matrix::Zero(1, 4);
            if (H5) *H5 = gtsam::Matrix::Zero(1, 3);
            if (H6) *H6 = gtsam::Matrix::Zero(1, 3);
            return gtsam::Vector1::Zero();
        }
        const Eigen::Vector3d rcv2sat_unit_i(e_i[0], e_i[1], e_i[2]);
        const Eigen::Vector3d rcv2sat_unit_j(e_j[0], e_j[1], e_j[2]);

        gtsam::Vector1 residual;
        residual[0] =
            (range_j - range_i + dt_j[p_.sys_idx] - dt_i[p_.sys_idx] - p_.cp_measured) * p_.cp_weight;

        if (!H1 && !H2 && !H3 && !H4 && !H5 && !H6) return residual;

        // d(range)/d(receiver ECEF), with the Sagnac term geodist() added differentiated too.
        const Eigen::RowVector3d d_range_d_P_i =
            -rcv2sat_unit_i.transpose() +
            Eigen::RowVector3d(-OMGE / CLIGHT * p_.sv_pos_i(1), OMGE / CLIGHT * p_.sv_pos_i(0), 0.0);
        const Eigen::RowVector3d d_range_d_P_j =
            -rcv2sat_unit_j.transpose() +
            Eigen::RowVector3d(-OMGE / CLIGHT * p_.sv_pos_j(1), OMGE / CLIGHT * p_.sv_pos_j(0), 0.0);

        if (H1) {
            *H1 = gtsam::Matrix::Zero(1, 3);
            H1->block<1, 3>(0, 0) = -d_range_d_P_i * R_ecef_local * p_.cp_weight;
        }
        if (H2) {
            *H2 = gtsam::Matrix::Zero(1, 3);
            H2->block<1, 3>(0, 0) = d_range_d_P_j * R_ecef_local * p_.cp_weight;
        }
        if (H3) {
            *H3 = gtsam::Matrix::Zero(1, 4);
            (*H3)(0, p_.sys_idx) = -p_.cp_weight;
        }
        if (H4) {
            *H4 = gtsam::Matrix::Zero(1, 4);
            (*H4)(0, p_.sys_idx) = p_.cp_weight;
        }
        if (H5) {
            // The anchor shifts both epochs alike, so only the difference of the two rows survives.
            *H5 = gtsam::Matrix::Zero(1, 3);
            H5->block<1, 3>(0, 0) = (d_range_d_P_j - d_range_d_P_i) * p_.cp_weight;
        }
        if (H6) {
            // gtsam::Rot3 uses a right perturbation R*Exp(d), so d(R*v)/dd = -R*skew(v).
            *H6 = gtsam::Matrix::Zero(1, 3);
            H6->block<1, 3>(0, 0) = (d_range_d_P_i * R_ecef_local * Skew(local_pos_i) -
                                     d_range_d_P_j * R_ecef_local * Skew(local_pos_j)) *
                                    p_.cp_weight;
        }
        return residual;
    }

    const Params& params() const { return p_; }

   private:
    static Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
        return m;
    }

    Params p_;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_GNSSTDCPFACTOR_H

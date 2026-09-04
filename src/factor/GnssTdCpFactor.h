//
// Created by hsiaochuan on 2026/09/03.
//

#ifndef FASTER_LIO_GNSSTDCPFACTOR_H
#define FASTER_LIO_GNSSTDCPFACTOR_H

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Dense>

namespace faster_lio {

class GnssTdCpFactor : public gtsam::NoiseModelFactor6<gtsam::Vector3, gtsam::Vector3, gtsam::Vector4, gtsam::Vector4,
                                                       gtsam::Vector3, gtsam::Rot3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Params {
        Eigen::Vector3d Tex_imu_r_i{Eigen::Vector3d::Zero()};  ///< IMU->antenna lever arm at epoch i, local frame
        Eigen::Vector3d Tex_imu_r_j{Eigen::Vector3d::Zero()};  ///< IMU->antenna lever arm at epoch j, local frame
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
        const Eigen::Vector3d local_pos_i = p_.Tex_imu_r_i + pos_i;
        const Eigen::Vector3d local_pos_j = p_.Tex_imu_r_j + pos_j;

        const Eigen::Matrix3d R_ecef_local = ext_R.matrix();
        const Eigen::Vector3d P_ecef_i = R_ecef_local * local_pos_i + ref_ecef;
        const Eigen::Vector3d P_ecef_j = R_ecef_local * local_pos_j + ref_ecef;

        const Eigen::Vector3d rcv2sat_ecef_i = p_.sv_pos_i - P_ecef_i;
        const Eigen::Vector3d rcv2sat_ecef_j = p_.sv_pos_j - P_ecef_j;
        const Eigen::Vector3d rcv2sat_unit_i = rcv2sat_ecef_i.normalized();
        const Eigen::Vector3d rcv2sat_unit_j = rcv2sat_ecef_j.normalized();

        gtsam::Vector1 residual;
        residual[0] = (rcv2sat_ecef_j.norm() - rcv2sat_ecef_i.norm() + dt_j[p_.sys_idx] - dt_i[p_.sys_idx] -
                       p_.cp_measured) *
                      p_.cp_weight;

        if (H1) {
            *H1 = gtsam::Matrix::Zero(1, 3);
            H1->block<1, 3>(0, 0) = rcv2sat_unit_i.transpose() * R_ecef_local * p_.cp_weight;
        }
        if (H2) {
            *H2 = gtsam::Matrix::Zero(1, 3);
            H2->block<1, 3>(0, 0) = -rcv2sat_unit_j.transpose() * R_ecef_local * p_.cp_weight;
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
            // The anchor shifts both epochs alike, so only the difference of the line-of-sight survives.
            *H5 = gtsam::Matrix::Zero(1, 3);
            H5->block<1, 3>(0, 0) = (rcv2sat_unit_i - rcv2sat_unit_j).transpose() * p_.cp_weight;
        }
        if (H6) {
            // gtsam::Rot3 uses a right perturbation R*Exp(d), so d(R*v)/dd = -R*skew(v).
            *H6 = gtsam::Matrix::Zero(1, 3);
            H6->block<1, 3>(0, 0) = rcv2sat_unit_j.transpose() * (R_ecef_local * Skew(local_pos_j)) * p_.cp_weight -
                                    rcv2sat_unit_i.transpose() * (R_ecef_local * Skew(local_pos_i)) * p_.cp_weight;
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

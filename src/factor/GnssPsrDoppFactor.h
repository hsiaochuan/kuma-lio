//
// Created by hsiaochuan on 2026/09/03.
//

#ifndef FASTER_LIO_GNSSPSRDOPPFACTOR_H
#define FASTER_LIO_GNSSPSRDOPPFACTOR_H

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "rtklib_.h"

#include <Eigen/Dense>
#include <cmath>


namespace faster_lio {
/// Pseudorange residual, built from exactly the pieces RTKLIB's rescode() uses: geodist()
/// for the Sagnac-corrected geometric range and the line of sight, ionocorr() and tropcorr()
/// for the atmosphere.  The measurement itself has already been through prange(), so the code
/// bias and the group delay are gone from `psr_measure` and no TGD term appears below.
class GnssPsrDoppFactor
    : public gtsam::NoiseModelFactor4<gtsam::Vector3, gtsam::Vector4, gtsam::Vector3, gtsam::Rot3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Params {
        Eigen::Vector3d Tex_imu_r{Eigen::Vector3d::Zero()};  ///< IMU->antenna lever arm, local frame
        Eigen::Vector3d sv_pos{Eigen::Vector3d::Zero()};     ///< satellite position, ECEF (m)
        double svdt{0.0};                                    ///< satellite clock bias (s)
        double pr_uura{1.0};                                 ///< pseudorange sigma (m)
        double relative_sqrt_info{1.0};                      ///< global scale on the residual
        double freq{FREQL1};                                 ///< carrier frequency of this signal (Hz)
        double psr_measure{0.0};                             ///< prange() output, code bias corrected (m)
        const nav_t* nav{nullptr};                           ///< iono models / TEC grid for ionocorr()
        gtime_t time{};                                      ///< observation epoch (GPST)
        int sat{0};                                          ///< RTKLIB satellite number
        int ionoopt{IONOOPT_BRDC};                           ///< prcopt.ionoopt handed to ionocorr()
        int tropopt{TROPOPT_SAAS};                           ///< prcopt.tropopt handed to tropcorr()
        int sys_idx{0};                                      ///< index into `dt` for this constellation
    };

    GnssPsrDoppFactor(gtsam::Key j1, gtsam::Key j2, gtsam::Key j3, gtsam::Key j4, const Params& p,
                      const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor4<gtsam::Vector3, gtsam::Vector4, gtsam::Vector3, gtsam::Rot3>(model, j1, j2, j3, j4),
          p_(p) {}

    ~GnssPsrDoppFactor() override = default;

    gtsam::Vector evaluateError(const gtsam::Vector3& pos, const gtsam::Vector4& dt, const gtsam::Vector3& ext_p,
                                const gtsam::Rot3& ext_R, boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none,
                                boost::optional<gtsam::Matrix&> H4 = boost::none) const override {
        const Eigen::Vector3d ref_ecef = ext_p;
        const Eigen::Vector3d local_pos = p_.Tex_imu_r + pos;

        const Eigen::Matrix3d R_ecef_local = ext_R.matrix();
        const Eigen::Vector3d P_ecef = R_ecef_local * local_pos + ref_ecef;

        // --- geometry: geodist() already folds the Sagnac (earth rotation) correction into
        //     the range it returns, and hands back the receiver->satellite unit vector.
        double e[3] = {0.0, 0.0, 0.0};
        const double range = geodist(p_.sv_pos.data(), P_ecef.data(), e);
        if (range <= 0.0) {
            // Bogus satellite position; the epoch was screened, so this only happens if the
            // state ran away.  Contribute nothing rather than a NaN.
            if (H1) *H1 = gtsam::Matrix::Zero(1, 3);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 4);
            if (H3) *H3 = gtsam::Matrix::Zero(1, 3);
            if (H4) *H4 = gtsam::Matrix::Zero(1, 3);
            return gtsam::Vector1::Zero();
        }
        const Eigen::Vector3d rcv2sat_unit(e[0], e[1], e[2]);

        // --- atmosphere, through RTKLIB's own dispatchers so that ionoopt/tropopt mean here
        //     exactly what they mean to postpos().
        double rcv_pos[3] = {0.0, 0.0, 0.0};  // {lat, lon, h} in rad/rad/m
        double azel[2] = {0.0, PI / 2.0};
        double ion_delay = 0.0, vion = 0.0, tro_delay = 0.0, vtrp = 0.0;
        ecef2pos(P_ecef.data(), rcv_pos);
        satazel(rcv_pos, e, azel);
        if (p_.nav) {
            ionocorr(p_.time, p_.nav, p_.sat, rcv_pos, azel, p_.ionoopt, &ion_delay, &vion);
            ion_delay *= (FREQL1 / p_.freq) * (FREQL1 / p_.freq);  // ionocorr() reports the L1 delay
            tropcorr(p_.time, p_.nav, rcv_pos, azel, p_.tropopt, &tro_delay, &vtrp);
        }

        // The model variances vion/vtrp are deliberately not folded into the weights: the
        // elevation they come from changes with the state, and the noise model has to stay
        // constant for the m-estimator below it to mean anything.
        const double sin_el = std::sin(azel[1]);
        const double pr_weight = sin_el * sin_el / p_.pr_uura * p_.relative_sqrt_info;

        // --- predicted measurement, mirroring rescode() ------------------------------
        const double psr_estimated = range + dt[p_.sys_idx] - CLIGHT * p_.svdt + ion_delay + tro_delay;

        gtsam::Vector1 residual;
        residual[0] = (psr_estimated - p_.psr_measure) * pr_weight;

        if (!H1 && !H2 && !H3 && !H4) return residual;

        // --- jacobians.  The atmospheric delays and the elevation weights also depend on the
        //     receiver position; rescode() ignores that and so does this.
        //     The Sagnac term geodist() adds differentiates w.r.t. the receiver position.
        const Eigen::RowVector3d sagnac_p(-OMGE / CLIGHT * p_.sv_pos(1), OMGE / CLIGHT * p_.sv_pos(0), 0.0);
        const Eigen::RowVector3d d_psr_d_P = -rcv2sat_unit.transpose() + sagnac_p;

        if (H1) {
            *H1 = gtsam::Matrix::Zero(1, 3);
            H1->block<1, 3>(0, 0) = d_psr_d_P * R_ecef_local * pr_weight;
        }
        if (H2) {
            *H2 = gtsam::Matrix::Zero(1, 4);
            (*H2)(0, p_.sys_idx) = pr_weight;
        }
        if (H3) {
            *H3 = gtsam::Matrix::Zero(1, 3);
            H3->block<1, 3>(0, 0) = d_psr_d_P * pr_weight;
        }
        if (H4) {
            // gtsam::Rot3 uses a right perturbation R*Exp(d), so d(R*v)/dd = -R*skew(v).
            const Eigen::Matrix3d d_P = -R_ecef_local * Skew(local_pos);
            *H4 = gtsam::Matrix::Zero(1, 3);
            H4->block<1, 3>(0, 0) = d_psr_d_P * d_P * pr_weight;
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

#endif  // FASTER_LIO_GNSSPSRDOPPFACTOR_H

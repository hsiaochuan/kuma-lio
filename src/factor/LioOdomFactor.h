//
// Created by hsiaochuan on 2026/09/03.
//

#ifndef FASTER_LIO_LIOODOMFACTOR_H
#define FASTER_LIO_LIOODOMFACTOR_H

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Dense>

namespace faster_lio {

/// Ties two consecutive position nodes to the increment the lidar-inertial front end
/// reported between the same two epochs.  Both states live in the LIO local frame, so
/// the increment is a plain difference and no rotation enters the residual.
class LioOdomFactor : public gtsam::NoiseModelFactor2<gtsam::Vector3, gtsam::Vector3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LioOdomFactor(gtsam::Key i, gtsam::Key j, const Eigen::Vector3d& delta_p, const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Vector3, gtsam::Vector3>(model, i, j), delta_p_(delta_p) {}

    ~LioOdomFactor() override = default;

    gtsam::Vector evaluateError(const gtsam::Vector3& pos_i, const gtsam::Vector3& pos_j,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        if (H1) *H1 = -gtsam::Matrix::Identity(3, 3);
        if (H2) *H2 = gtsam::Matrix::Identity(3, 3);

        return gtsam::Vector3(pos_j - pos_i - delta_p_);
    }

   private:
    Eigen::Vector3d delta_p_;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LIOODOMFACTOR_H

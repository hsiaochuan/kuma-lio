//
// Created by hsiaochuan on 2026/09/03.
//

#ifndef FASTER_LIO_GNSSCLOCKFACTOR_H
#define FASTER_LIO_GNSSCLOCKFACTOR_H

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace faster_lio {

/// Propagates the four per-constellation receiver clock biases with the common drift:
/// dt_j - dt_i = 0.5 * (ddt_i + ddt_j) * delta_t.  Units are metres and metres/second.
class GnssDtDdtFactor
    : public gtsam::NoiseModelFactor4<gtsam::Vector4, gtsam::Vector4, gtsam::Vector1, gtsam::Vector1> {
   public:
    GnssDtDdtFactor(gtsam::Key i1, gtsam::Key j1, gtsam::Key i2, gtsam::Key j2, double delta_t,
                    const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor4<gtsam::Vector4, gtsam::Vector4, gtsam::Vector1, gtsam::Vector1>(model, i1, j1, i2,
                                                                                                   j2),
          delta_t_(delta_t) {}

    ~GnssDtDdtFactor() override = default;

    gtsam::Vector evaluateError(const gtsam::Vector4& dt_i, const gtsam::Vector4& dt_j, const gtsam::Vector1& ddt_i,
                                const gtsam::Vector1& ddt_j, boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none,
                                boost::optional<gtsam::Matrix&> H4 = boost::none) const override {
        if (H1) *H1 = -gtsam::Matrix::Identity(4, 4);
        if (H2) *H2 = gtsam::Matrix::Identity(4, 4);
        if (H3) *H3 = gtsam::Matrix::Constant(4, 1, -0.5 * delta_t_);
        if (H4) *H4 = gtsam::Matrix::Constant(4, 1, -0.5 * delta_t_);

        gtsam::Vector4 residual;
        const double propagated = 0.5 * (ddt_i[0] + ddt_j[0]) * delta_t_;
        for (int i = 0; i < 4; ++i) {
            residual[i] = dt_j[i] - dt_i[i] - propagated;
        }
        return residual;
    }

   private:
    double delta_t_;
};

/// Random-walk on the receiver clock drift.  `ddt` is only seen through the bias
/// propagation above, so this keeps it from wandering between epochs.
class GnssDdtSmoothFactor : public gtsam::NoiseModelFactor2<gtsam::Vector1, gtsam::Vector1> {
   public:
    GnssDdtSmoothFactor(gtsam::Key i, gtsam::Key j, const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactor2<gtsam::Vector1, gtsam::Vector1>(model, i, j) {}

    ~GnssDdtSmoothFactor() override = default;

    gtsam::Vector evaluateError(const gtsam::Vector1& ddt_i, const gtsam::Vector1& ddt_j,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        if (H1) *H1 = -gtsam::Matrix::Identity(1, 1);
        if (H2) *H2 = gtsam::Matrix::Identity(1, 1);
        return gtsam::Vector1(ddt_j[0] - ddt_i[0]);
    }
};

}  // namespace faster_lio

#endif  // FASTER_LIO_GNSSCLOCKFACTOR_H

//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_FISHEYE_CAMERA_H
#define PINHOLE_FISHEYE_CAMERA_H

#include "cameras.h"

namespace faster_lio {

/**
 * @brief Pinhole camera with equidistant fisheye distortion (Kannala–Brandt).
 *
 * \f[
 *   \theta_d = \theta + K_1\theta^3 + K_2\theta^5 + K_3\theta^7 + K_4\theta^9
 * \f]
 * where \f$\theta = \arctan(r)\f$ and \f$r = \|p\|\f$.
 *
 * Parameters: [ f, cx, cy, K1, K2, K3, K4 ]
 */
class PinholeFisheyeCamera : public CameraBase {
   public:
    PinholeFisheyeCamera(unsigned int w  = 0,
                         unsigned int h  = 0,
                         double       fx  = 0.0,
                         double       fy  = 0.0,
                         double       cx = 0.0,
                         double       cy = 0.0,
                         double       k1 = 0.0,
                         double       k2 = 0.0,
                         double       k3 = 0.0,
                         double       k4 = 0.0)
        : CameraBase(w, h), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          params_({k1, k2, k3, k4}) {}

    ~PinholeFisheyeCamera() override = default;

    std::unique_ptr<CameraBase> clone() const override {
        return std::make_unique<PinholeFisheyeCamera>(*this);
    }

    CAMERA_MODEL getType() const override { return PINHOLE_FISHEYE; }

    // -- Accessors -----------------------------------------------------------
    double      focal()           const { return fx_ / 2.0 + fy_ / 2.0; }
    common::V2D principal_point() const { return {cx_, cy_}; }

    // -- Coordinate transforms -----------------------------------------------
    common::V2D cam2ima(const common::V2D& p) const override {
        return focal() * p + common::V2D{cx_, cy_};
    }
    common::V2D ima2cam(const common::V2D& p) const override {
        return (p - common::V2D{cx_, cy_}) / focal();
    }

    // -- Distortion ----------------------------------------------------------
    bool have_disto() const override { return true; }

    common::V2D add_disto(const common::V2D& p) const override {
        constexpr double kEps = 1e-8;
        const double k1 = params_[0], k2 = params_[1],
                     k3 = params_[2], k4 = params_[3];
        const double r       = std::hypot(p(0), p(1));
        const double theta   = std::atan(r);
        const double th2     = theta * theta;
        const double theta_d = theta * (1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))));
        const double cdist   = (r > kEps) ? theta_d / r : 1.0;
        return p * cdist;
    }

    common::V2D remove_disto(const common::V2D& p) const override {
        constexpr double kEps = 1e-8;
        const double theta_d = std::hypot(p(0), p(1));
        double scale = 1.0;
        if (theta_d > kEps) {
            double theta = theta_d;
            for (int i = 0; i < 10; ++i) {
                const double th2 = theta * theta;
                theta = theta_d / (1.0 + th2 * (params_[0]
                                        + th2 * (params_[1]
                                        + th2 * (params_[2]
                                        + th2 *  params_[3]))));
            }
            scale = std::tan(theta) / theta_d;
        }
        return p * scale;
    }

    common::V2D get_ud_pixel(const common::V2D& p) const override {
        return cam2ima(remove_disto(ima2cam(p)));
    }
    common::V2D get_d_pixel(const common::V2D& p) const override {
        return cam2ima(add_disto(ima2cam(p)));
    }

    // -- Parameters ----------------------------------------------------------
    std::vector<double> getParams() const override {
        return {fx_, fy_, cx_, cy_,
                params_[0], params_[1], params_[2], params_[3]};
    }
    bool updateFromParams(const std::vector<double>& params) override {
        if (params.size() != 8) return false;
        *this = PinholeFisheyeCamera(w_, h_,
                                     params[0], params[1], params[2],params[3],
                                     params[4], params[5], params[6], params[7]);
        return true;
    }

   private:
    double fx_;   ///< Focal length (pixels)
    double fy_;
    double cx_;  ///< Principal-point x
    double cy_;  ///< Principal-point y
    std::vector<double> params_;  ///< [K1, K2, K3, K4]
};

}  // namespace faster_lio

#endif  // PINHOLE_FISHEYE_CAMERA_H
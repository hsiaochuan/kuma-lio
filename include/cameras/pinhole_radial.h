//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_RADIAL_H
#define PINHOLE_RADIAL_H

#include "camera_base.h"

namespace faster_lio {

/**
 * @brief Pinhole camera with radial + tangential (Brown–Conrady) distortion.
 *
 * \f[
 *   x_d = x_u (1 + K_1 r^2 + K_2 r^4 + K_3 r^6)
 *         + \bigl(T_2(r^2 + 2x_u^2) + 2T_1 x_u y_u\bigr)
 * \f]
 * \f[
 *   y_d = y_u (1 + K_1 r^2 + K_2 r^4 + K_3 r^6)
 *         + \bigl(T_1(r^2 + 2y_u^2) + 2T_2 x_u y_u\bigr)
 * \f]
 *
 * Parameters: [ f, cx, cy, K1, K2, K3, P1, P2 ]
 */
class PinholeRadialCamera : public CameraBase {
   public:
    PinholeRadialCamera(unsigned int w  = 0,
                        unsigned int h  = 0,
                        double       fx  = 0.0,
                        double       fy  = 0.0,
                        double       cx = 0.0,
                        double       cy = 0.0,
                        double       k1 = 0.0,
                        double       k2 = 0.0,
                        double       k3 = 0.0,
                        double       p1 = 0.0,
                        double       p2 = 0.0)
        : CameraBase(w, h), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          params_({k1, k2, k3, p1, p2}) {}

    ~PinholeRadialCamera() override = default;

    std::unique_ptr<CameraBase> clone() const override {
        return std::make_unique<PinholeRadialCamera>(*this);
    }

    CAMERA_MODEL getType() const override { return PINHOLE_RADIAL; }

    // -- Accessors -----------------------------------------------------------
    double      focal()          const { return fx_ * 0.5 + fy_ * 0.5; }
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
        return p + distoFunction(params_, p);
    }

    /**
     * @brief Iterative undistortion (Heikkilä 2000).
     */
    common::V2D remove_disto(const common::V2D& p) const override {
        constexpr double kEps = 1e-10;
        common::V2D p_u = p;
        common::V2D d   = distoFunction(params_, p_u);
        while ((p_u + d - p).template lpNorm<1>() > kEps) {
            p_u = p - d;
            d   = distoFunction(params_, p_u);
        }
        return p_u;
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
                params_[0], params_[1], params_[2],   // K1, K2, K3
                params_[3], params_[4]};               // P1, P2
    }
    bool updateFromParams(const std::vector<double>& params) override {
        if (params.size() != 9) return false;
        *this = PinholeRadialCamera(w_, h_,
                                    params[0], params[1], params[2], params[3],
                                    params[4], params[5],params[6],
                                    params[7], params[8]);
        return true;
    }

   private:
    double fx_;   ///< Focal length (pixels)
    double fy_;
    double cx_;  ///< Principal-point x
    double cy_;  ///< Principal-point y
    std::vector<double> params_;  ///< [K1, K2, K3, T1, T2]

    static common::V2D distoFunction(const std::vector<double>& params,
                                     const common::V2D& p) {
        const double k1 = params[0], k2 = params[1], k3 = params[2];
        const double p1 = params[3], p2 = params[4];
        const double r2     = p(0) * p(0) + p(1) * p(1);
        const double r4     = r2 * r2;
        const double r6     = r4 * r2;
        const double k_diff = k1 * r2 + k2 * r4 + k3 * r6;
        const double tx     = p2 * (r2 + 2.0 * p(0) * p(0)) + 2.0 * p1 * p(0) * p(1);
        const double ty     = p1 * (r2 + 2.0 * p(1) * p(1)) + 2.0 * p2 * p(0) * p(1);
        return {p(0) * k_diff + tx, p(1) * k_diff + ty};
    }
};

}  // namespace faster_lio

#endif  // PINHOLE_RADIAL_H
//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_FISHEYE_CAMERA_H
#define PINHOLE_FISHEYE_CAMERA_H

#include "camera_base.h"

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
class PinholeFisheyeCamera : public CamModel {
   public:
    PinholeFisheyeCamera(unsigned int w = 0, unsigned int h = 0, double fx = 0.0, double fy = 0.0, double cx = 0.0,
                         double cy = 0.0, double k1 = 0.0, double k2 = 0.0, double k3 = 0.0, double k4 = 0.0)
        : CamModel(w, h), fx_(fx), fy_(fy), cx_(cx), cy_(cy), k1_(k1), k2_(k2), k3_(k3), k4_(k4) {}

    ~PinholeFisheyeCamera() override = default;

    CAMERA_MODEL get_type() const override { return PINHOLE_FISHEYE; }
    double focal() const { return fx_ * 0.5 + fy_ * 0.5; }
    Vec2 principal_point() const { return {cx_, cy_}; }

    Vec2 cam2ima(const Vec2& p) const override { return focal() * p + Vec2{cx_, cy_}; }
    Vec2 ima2cam(const Vec2& p) const override { return (p - Vec2{cx_, cy_}) / focal(); }

    Vec2 add_disto(const Vec2& p) const override {
        constexpr double kEps = 1e-8;
        const double r = std::hypot(p(0), p(1));
        const double theta = std::atan(r);
        const double th2 = theta * theta;
        const double theta_d = theta * (1.0 + th2 * (k1_ + th2 * (k2_ + th2 * (k3_ + th2 * k4_))));
        const double cdist = (r > kEps) ? theta_d / r : 1.0;
        return p * cdist;
    }

    Vec2 remove_disto(const Vec2& p) const override {
        constexpr double kEps = 1e-8;
        const double theta_d = std::hypot(p(0), p(1));
        double scale = 1.0;
        if (theta_d > kEps) {
            double theta = theta_d;
            for (int i = 0; i < 10; ++i) {
                const double th2 = theta * theta;
                theta = theta_d / (1.0 + th2 * (k1_ + th2 * (k2_ + th2 * (k3_ + th2 * k4_))));
            }
            scale = std::tan(theta) / theta_d;
        }
        return p * scale;
    }

    Vec2 get_ud_pixel(const Vec2& p) const override { return cam2ima(remove_disto(ima2cam(p))); }

    Vec2 project(const Vec3& X) const override { return cam2ima(add_disto(X.hnormalized())); }
    Eigen::Vector3d bearing(const Eigen::Vector2d& ima_point) const override {
        return ima2cam(get_ud_pixel(ima_point)).homogeneous().normalized();
    }

    // -- Parameters ----------------------------------------------------------
    std::vector<double> get_params() const override { return {fx_, fy_, cx_, cy_, k1_, k2_, k3_, k4_}; }
    bool update_params(const std::vector<double>& params) override {
        CHECK(params.size() == 8) << "Expected 8 parameters for PinholeFisheyeCamera, got " << params.size();
        *this = PinholeFisheyeCamera(w_, h_, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
        return true;
    }

   private:
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    double k1_;
    double k2_;
    double k3_;
    double k4_;
};

}  // namespace faster_lio

#endif  // PINHOLE_FISHEYE_CAMERA_H
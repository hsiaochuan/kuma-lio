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
class PinholeRadialCamera : public CamModel {
   public:
    PinholeRadialCamera(unsigned int w = 0, unsigned int h = 0, double fx = 0.0, double fy = 0.0, double cx = 0.0,
                        double cy = 0.0, double k1 = 0.0, double k2 = 0.0, double k3 = 0.0, double p1 = 0.0,
                        double p2 = 0.0)
        : CamModel(w, h), fx_(fx), fy_(fy), cx_(cx), cy_(cy), k1_(k1), k2_(k2), k3_(k3), p1_(p1), p2_(p2) {}

    ~PinholeRadialCamera() override = default;

    CAMERA_MODEL get_type() const override { return PINHOLE_RADIAL; }

    double focal() const { return fx_ * 0.5 + fy_ * 0.5; }
    Vec2 principal_point() const { return {cx_, cy_}; }

    // -- Coordinate transforms -----------------------------------------------
    Vec2 cam2ima(const Vec2& p) const override { return focal() * p + Vec2{cx_, cy_}; }
    Vec2 ima2cam(const Vec2& p) const override { return (p - Vec2{cx_, cy_}) / focal(); }

    Vec2 add_disto(const Vec2& p) const override { return p + distoFunction(k1_, k2_, k3_, p1_, p2_, p); }
    Vec2 remove_disto(const Vec2& p) const override {
        constexpr double kEps = 1e-10;
        Vec2 p_u = p;
        Vec2 d = distoFunction(k1_, k2_, k3_, p1_, p2_, p_u);
        while ((p_u + d - p).template lpNorm<1>() > kEps) {
            p_u = p - d;
            d = distoFunction(k1_, k2_, k3_, p1_, p2_, p_u);
        }
        return p_u;
    }
    Vec2 get_ud_pixel(const Vec2& p) const override { return cam2ima(remove_disto(ima2cam(p))); }

    Vec2 project(const Vec3& X) const override { return cam2ima(add_disto(X.hnormalized())); }
    Eigen::Vector3d bearing(const Eigen::Vector2d& ima_point) const override {
        return ima2cam(get_ud_pixel(ima_point)).homogeneous().normalized();
    }
    std::vector<double> get_params() const override { return {fx_, fy_, cx_, cy_, k1_, k2_, k3_, p1_, p2_}; }
    bool update_params(const std::vector<double>& params) override {
        CHECK(params.size() == 9) << "Expected 9 parameters for PinholeRadialCamera, got " << params.size();
        *this = PinholeRadialCamera(w_, h_, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                                    params[7], params[8]);
        return true;
    }

    double fx_;
    double fy_;
    double cx_;
    double cy_;

    double k1_;
    double k2_;
    double k3_;
    double p1_;
    double p2_;

    static Vec2 distoFunction(const double& k1, const double& k2, const double& k3, const double& p1, const double& p2,
                              const Vec2& p) {
        const double r2 = p(0) * p(0) + p(1) * p(1);
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double k_diff = k1 * r2 + k2 * r4 + k3 * r6;
        const double tx = p2 * (r2 + 2.0 * p(0) * p(0)) + 2.0 * p1 * p(0) * p(1);
        const double ty = p1 * (r2 + 2.0 * p(1) * p(1)) + 2.0 * p2 * p(0) * p(1);
        return {p(0) * k_diff + tx, p(1) * k_diff + ty};
    }
};

}  // namespace faster_lio

#endif  // PINHOLE_RADIAL_H
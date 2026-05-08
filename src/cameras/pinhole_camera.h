//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_CAMERA_H
#define PINHOLE_CAMERA_H

#include "camera_base.h"

namespace faster_lio {
/**
 * @brief Ideal pinhole camera – no distortion.
 *
 * \f[ K = \begin{pmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{pmatrix} \f]
 *
 * Parameters: [ f, cx, cy ]
 */
class PinholeCamera : public CamModel {
   public:
    PinholeCamera(unsigned int w = 0, unsigned int h = 0, double fx = 0.0, double fy = 0.0, double cx = 0.0,
                  double cy = 0.0)
        : CamModel(w, h), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    ~PinholeCamera() override = default;

    CAMERA_MODEL get_type() const override { return PINHOLE; }

    double focal() const { return fx_ * 0.5 + fy_ * 0.5; }
    Vec2 principal_point() const { return {cx_, cy_}; }

    Vec2 cam2ima(const Vec2 &p) const override { return focal() * p + principal_point(); }
    Vec2 ima2cam(const Vec2 &p) const override { return (p - principal_point()) / focal(); }

    Vec2 add_disto(const Vec2 &p) const override { return p; }
    Vec2 remove_disto(const Vec2 &p) const override { return p; }
    Vec2 get_ud_pixel(const Vec2 &p) const override { return p; }

    Vec2 project(const Vec3 &X) const override { return cam2ima(add_disto(X.hnormalized())); }
    Eigen::Vector3d bearing(const Eigen::Vector2d &ima_point) const override {
        return ima2cam(get_ud_pixel(ima_point)).homogeneous().normalized();
    }

    // -- Parameters ----------------------------------------------------------
    std::vector<double> get_params() const override { return {fx_, fy_, cx_, cy_}; }

    bool update_params(const std::vector<double> &params) override {
        CHECK(params.size() == 4) << "PinholeCamera requires 4 parameters: [fx, fy, cx, cy]";
        *this = PinholeCamera(w_, h_, params[0], params[1], params[2], params[3]);
        return true;
    }

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};
}  // namespace faster_lio

#endif  // PINHOLE_CAMERA_H

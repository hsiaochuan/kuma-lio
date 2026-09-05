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
 * \f[ K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix} \f]
 *
 * Parameters: [ fx, fy, cx, cy ]
 */
class PinholeCamera : public CamModel {
   public:
    using Ptr = std::shared_ptr<PinholeCamera>;
    PinholeCamera(unsigned int w = 0, unsigned int h = 0, double fx = 0.0, double fy = 0.0, double cx = 0.0,
                  double cy = 0.0)
        : CamModel(w, h), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    ~PinholeCamera() override = default;

    CAMERA_MODEL get_type() const override { return PINHOLE; }

    Vec2 cam2ima(const Vec2 &p) const override { return {fx_ * p.x() + cx_, fy_ * p.y() + cy_}; }
    Vec2 ima2cam(const Vec2 &p) const override { return {(p.x() - cx_) / fx_, (p.y() - cy_) / fy_}; }

    Vec2 add_disto(const Vec2 &p) const override { return p; }
    Vec2 remove_disto(const Vec2 &p) const override { return p; }
    Vec2 get_ud_pixel(const Vec2 &p) const override { return p; }

    Vec2 project(const Vec3 &X) const override { return cam2ima(add_disto(X.hnormalized())); }
    Eigen::Vector3d bearing(const Eigen::Vector2d &ima_point) const override {
        return ima2cam(get_ud_pixel(ima_point)).homogeneous().normalized();
    }

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};
}  // namespace faster_lio

#endif  // PINHOLE_CAMERA_H

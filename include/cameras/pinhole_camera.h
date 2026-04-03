//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_CAMERA_H
#define PINHOLE_CAMERA_H

#include "cameras.h"

namespace faster_lio {
/**
 * @brief Ideal pinhole camera – no distortion.
 *
 * \f[ K = \begin{pmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{pmatrix} \f]
 *
 * Parameters: [ f, cx, cy ]
 */
class PinholeCamera : public CameraBase {
   public:
    PinholeCamera(unsigned int w = 0, unsigned int h = 0, double fx = 0.0, double fy = 0.0, double cx = 0.0,
                  double cy = 0.0)
        : CameraBase(w, h) {
        K_ << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
        Kinv_ = K_.inverse();
    }

    PinholeCamera(unsigned int w, unsigned int h, const common::M3D &K) : CameraBase(w, h), K_(K) {
        K_(0, 0) = K_(1, 1) = (K(0, 0) + K(1, 1)) / 2.0;
        Kinv_ = K_.inverse();
    }

    ~PinholeCamera() override = default;

    std::unique_ptr<CameraBase> clone() const override { return std::make_unique<PinholeCamera>(*this); }

    CAMERA_MODEL getType() const override { return PINHOLE; }

    // -- Accessors -----------------------------------------------------------
    const common::M3D &K() const { return K_; }
    const common::M3D &Kinv() const { return Kinv_; }
    double focal() const { return K_(0, 0) / 2.0 + K_(1, 1) / 2.0; }
    common::V2D principal_point() const { return {K_(0, 2), K_(1, 2)}; }

    // -- Coordinate transforms -----------------------------------------------
    common::V2D cam2ima(const common::V2D &p) const override { return focal() * p + principal_point(); }

    common::V2D ima2cam(const common::V2D &p) const override { return (p - principal_point()) / focal(); }

    // -- Distortion (none) ---------------------------------------------------
    bool have_disto() const override { return false; }
    common::V2D add_disto(const common::V2D &p) const override { return p; }
    common::V2D remove_disto(const common::V2D &p) const override { return p; }
    common::V2D get_ud_pixel(const common::V2D &p) const override { return p; }
    common::V2D get_d_pixel(const common::V2D &p) const override { return p; }

    // -- Parameters ----------------------------------------------------------
    std::vector<double> getParams() const override { return {K_(0, 0), K_(1, 1), K_(0, 2), K_(1, 2)}; }

    bool updateFromParams(const std::vector<double> &params) override {
        if (params.size() != 4) return false;
        *this = PinholeCamera(w_, h_, params[0], params[1], params[2], params[3]);
        return true;
    }

   private:
    common::M3D K_;
    common::M3D Kinv_;
};
}  // namespace faster_lio

#endif  // PINHOLE_CAMERA_H

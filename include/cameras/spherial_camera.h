//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef SPHERICAL_CAMERA_H
#define SPHERICAL_CAMERA_H

#include "camera_base.h"

namespace faster_lio {

/**
 * @brief Full-spherical (equirectangular) camera model.
 *
 * No focal length or distortion. The unit sphere maps to the image plane via:
 * \f[
 *   \text{lon} = \text{atan2}(X_x, X_z), \quad
 *   \text{lat} = \text{atan2}(-X_y,\, \sqrt{X_x^2 + X_z^2})
 * \f]
 * \f[
 *   u = \frac{\text{lon}}{2\pi} \cdot S + \frac{w}{2}, \quad
 *   v = \frac{-\text{lat}}{2\pi} \cdot S + \frac{h}{2}, \quad
 *   S = \max(w, h)
 * \f]
 *
 * Parameters: none (empty vector).
 */
class SphericalCamera : public CamModel {
   public:
    explicit SphericalCamera(unsigned int w = 0, unsigned int h = 0)
        : CamModel(w, h) {
    }

    ~SphericalCamera() override = default;

    CAMERA_MODEL getType() const override { return SPHERICAL; }

    // -- Projection (overrides default perspective-divide path) --------------
    common::V2D project(const common::V3D& X,
                        bool /*ignore_distortion*/ = false) const override {
        const double lon = std::atan2(X.x(), X.z());
        const double lat = std::atan2(-X.y(), std::hypot(X.x(), X.z()));
        return cam2ima({lon / (2.0 * M_PI), -lat / (2.0 * M_PI)});
    }

    // -- Coordinate transforms -----------------------------------------------
    common::V2D cam2ima(const common::V2D& p) const override {
        const double S = static_cast<double>(std::max(w_, h_));
        return {p.x() * S + w_ / 2.0,
                p.y() * S + h_ / 2.0};
    }
    common::V2D ima2cam(const common::V2D& p) const override {
        const double S = static_cast<double>(std::max(w_, h_));
        return {(p.x() - w_ / 2.0) / S,
                (p.y() - h_ / 2.0) / S};
    }

    // -- Distortion (none) ---------------------------------------------------
    bool        have_disto()                        const override { return false; }
    common::V2D add_disto   (const common::V2D& p) const override { return p; }
    common::V2D remove_disto(const common::V2D& p) const override { return p; }
    common::V2D get_ud_pixel(const common::V2D& p) const override { return p; }

    // -- Parameters ----------------------------------------------------------
    std::vector<double> getParams() const override { return {}; }
    bool updateFromParams(const std::vector<double>& /*params*/) override { return true; }
};

}  // namespace faster_lio

#endif  // SPHERICAL_CAMERA_H
//
// Refactored by hsiaochuan on 2026/03/30.
//

#ifndef CAMERAS_H
#define CAMERAS_H

#include <glog/logging.h>

#include "common_lib.h"

namespace faster_lio {

enum CAMERA_MODEL {
    PINHOLE = 0,          ///< Ideal pinhole, no distortion
    PINHOLE_RADIAL = 1,   ///< Pinhole + radial(K1,K2,K3) + tangential(T1,T2)
    PINHOLE_FISHEYE = 2,  ///< Pinhole + fisheye polynomial (K1,K2,K3,K4)
    SPHERICAL = 3,        ///< Full spherical / equirectangular
};
inline CAMERA_MODEL ToCameraModel(const std::string& camera_model) {
    if (camera_model == "pinhole") {
        return PINHOLE;
    } else if (camera_model == "pinhole_radial") {
        return PINHOLE_RADIAL;
    } else if (camera_model == "pinhole_fisheye") {
        return PINHOLE_FISHEYE;
    } else if (camera_model == "spherical") {
        return SPHERICAL;
    } else {
        LOG(ERROR) << "Unknown camera type: " << camera_model;
        return PINHOLE;
    }
}
inline std::string CameraModelToString(CAMERA_MODEL camera_model) {
    switch (camera_model) {
        case PINHOLE:
            return "pinhole";
        case PINHOLE_RADIAL:
            return "pinhole_radial";
        case PINHOLE_FISHEYE:
            return "pinhole_fisheye";
        case SPHERICAL:
            return "spherical";
        default:
            throw std::runtime_error("Unknown camera type");
    }
}
/**
 * @brief Pure abstract base class for all camera models.
 *
 * Every concrete camera derives directly from this class (flat hierarchy).
 */
class CamModel {
   public:
    using Ptr = std::shared_ptr<CamModel>;
    explicit CamModel(unsigned int w = 0, unsigned int h = 0) : w_(w), h_(h) {}
    virtual ~CamModel() = default;
    virtual CAMERA_MODEL getType() const = 0;


    // -- Image dimensions ----------------------------------------------------
    unsigned int w() const { return w_; }
    unsigned int h() const { return h_; }

    // -- Projection ----------------------------------------------------------

    virtual common::V2D project(const common::V3D& X, bool ignore_distort = false) const {
        if (have_disto() && !ignore_distort)
            return cam2ima(add_disto(X.hnormalized()));
        return cam2ima(X.hnormalized());
    }

    bool valid(const Eigen::Vector2i & uv) const {
        if (uv.x() < 0 || uv.x() >= w() || uv.y() < 0 || uv.y() >= h()) return false;
        return true;
    }
    bool positive_z(const common::V3D& X) const { return X.z() > 0; }

    /// Residual: observed pixel minus projected pixel
    common::V2D residual(const common::V3D& X, const common::V2D& x, bool ignore_distortion = false) const {
        return x - project(X, ignore_distortion);
    }

    // -- Coordinate transforms (pure virtual) --------------------------------
    virtual common::V2D cam2ima(const common::V2D& p) const = 0;
    virtual common::V2D ima2cam(const common::V2D& p) const = 0;

    // -- Distortion (pure virtual) -------------------------------------------
    virtual bool have_disto() const = 0;
    virtual common::V2D add_disto(const common::V2D& p) const = 0;
    virtual common::V2D remove_disto(const common::V2D& p) const = 0;
    virtual common::V2D get_ud_pixel(const common::V2D& p) const = 0;


    // -- Parameters (pure virtual) -------------------------------------------
    virtual std::vector<double> getParams() const = 0;
    virtual bool updateFromParams(const std::vector<double>&) = 0;

    unsigned int w_;
    unsigned int h_;
};

inline bool IsPinhole(CAMERA_MODEL m) { return m <= PINHOLE_FISHEYE; }
inline bool IsSpherical(CAMERA_MODEL m) { return m == SPHERICAL; }
inline bool IsDistorted(CAMERA_MODEL m) {
    if (m == PINHOLE || m == SPHERICAL) return false;
    return true;
}
}  // namespace faster_lio

#endif  // CAMERAS_H
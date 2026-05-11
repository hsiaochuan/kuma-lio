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
    virtual CAMERA_MODEL get_type() const = 0;

    // -- Image dimensions ----------------------------------------------------
    unsigned int w() const { return w_; }
    unsigned int h() const { return h_; }

    // -- Projection ----------------------------------------------------------

    template <typename Scalar>
    bool valid(const Eigen::Matrix<Scalar,2,1>& uv) const {
        if (uv.x() < 0 || uv.x() >= w() || uv.y() < 0 || uv.y() >= h()) return false;
        return true;
    }
    bool positive_z(const Vec3& X) const { return X.z() > 0; }
    std::optional<Vec2> project_and_valid(const Vec3& X) {
        if (!positive_z(X))
            return std::nullopt;
        Vec2 p_im = project(X);
        if (valid(p_im))
            return p_im;
        else
            return std::nullopt;
    }
    virtual Vec2 cam2ima(const Vec2& p) const = 0;
    virtual Vec2 ima2cam(const Vec2& p) const = 0;

    virtual Vec2 add_disto(const Vec2& p) const = 0;
    virtual Vec2 remove_disto(const Vec2& p) const = 0;
    virtual Vec2 get_ud_pixel(const Vec2& p) const = 0;

    virtual Eigen::Vector3d bearing(const Eigen::Vector2d& ima_point) const = 0;
    virtual Vec2 project(const Vec3& X) const = 0;

    virtual std::vector<double> get_params() const = 0;
    virtual bool update_params(const std::vector<double>&) = 0;

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
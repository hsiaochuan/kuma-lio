//
// FOV model (Devernay & Faugeras, "Straight lines have to be straight", 2001), after COLMAP
// src/colmap/sensor/models/fov.h.  Distortion and undistortion are both closed form.
//

#ifndef CAMERAS_FOV_H
#define CAMERAS_FOV_H

#include "camera_base.h"

namespace faster_lio {

/// FOV: fx, fy, cx, cy, omega.  r_d = atan(2 r tan(omega / 2)) / omega.
class FOVCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kFOV;
    static constexpr const char* kModelName = "FOV";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, omega";
    static constexpr size_t kNumParams = 5;
    FOVCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const Vec2 d = Distort(params_[4], X.hnormalized());
        *uv = Vec2(f1 * d.x() + c1, f2 * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *xy = Undistort(params_[4], Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2));
        return true;
    }

    /// The distorted point itself (COLMAP FOVCameraModel::Distortion), with its Taylor expansions near
    /// omega = 0 and r = 0.
    static Vec2 Distort(double omega, const Vec2& p) {
        constexpr double kEpsilon = 1e-4;  // chosen arbitrarily, as in COLMAP
        const double radius2 = p.squaredNorm();
        const double omega2 = omega * omega;
        double factor;
        if (omega2 < kEpsilon) {
            factor = omega2 / 12.0 - (omega2 * radius2) / 3.0 + 1.0;
        } else if (radius2 < kEpsilon) {
            const double tan_half_omega = std::tan(omega / 2.0);
            factor = (-2.0 * tan_half_omega * (4.0 * radius2 * tan_half_omega * tan_half_omega - 3.0)) / (3.0 * omega);
        } else {
            const double radius = std::sqrt(radius2);
            factor = std::atan(radius * 2.0 * std::tan(omega / 2.0)) / (radius * omega);
        }
        return p * factor;
    }
    /// The inverse of Distort (COLMAP FOVCameraModel::Undistortion).
    static Vec2 Undistort(double omega, const Vec2& p) {
        constexpr double kEpsilon = 1e-4;
        const double radius2 = p.squaredNorm();
        const double omega2 = omega * omega;
        double factor;
        if (omega2 < kEpsilon) {
            factor = (omega2 * radius2) / 3.0 - omega2 / 12.0 + 1.0;
        } else if (radius2 < kEpsilon) {
            factor = (omega * (omega * omega * radius2 + 3.0)) / (6.0 * std::tan(omega / 2.0));
        } else {
            const double radius = std::sqrt(radius2);
            factor = std::tan(radius * omega) / (radius * 2.0 * std::tan(omega / 2.0));
        }
        return p * factor;
    }
};

}  // namespace faster_lio

#endif  // CAMERAS_FOV_H

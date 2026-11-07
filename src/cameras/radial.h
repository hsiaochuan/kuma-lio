//
// Pinhole models with radial distortion, after COLMAP src/colmap/sensor/models/radial.h.
//

#ifndef CAMERAS_RADIAL_H
#define CAMERAS_RADIAL_H

#include "camera_base.h"

namespace faster_lio {

/// SIMPLE_RADIAL: f, cx, cy, k.  Distortion du = u k r^2.
class SimpleRadialCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kSimpleRadial;
    static constexpr const char* kModelName = "SIMPLE_RADIAL";
    static constexpr const char* kParamsInfo = "f, cx, cy, k";
    static constexpr size_t kNumParams = 4;
    SimpleRadialCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        const Vec2 p = X.hnormalized();
        const Vec2 d = p + Distortion(&params_[3], p);
        *uv = Vec2(f * d.x() + c1, f * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *xy = Vec2((uv.x() - c1) / f, (uv.y() - c2) / f);
        const double* extra = &params_[3];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, xy);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k = extra[0];
        const double radial = k * p.squaredNorm();
        return p * radial;
    }
};

/// RADIAL: f, cx, cy, k1, k2.  Distortion du = u (k1 r^2 + k2 r^4).
class RadialCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kRadial;
    static constexpr const char* kModelName = "RADIAL";
    static constexpr const char* kParamsInfo = "f, cx, cy, k1, k2";
    static constexpr size_t kNumParams = 5;
    RadialCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        const Vec2 p = X.hnormalized();
        const Vec2 d = p + Distortion(&params_[3], p);
        *uv = Vec2(f * d.x() + c1, f * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *xy = Vec2((uv.x() - c1) / f, (uv.y() - c2) / f);
        const double* extra = &params_[3];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, xy);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k1 = extra[0], k2 = extra[1];
        const double r2 = p.squaredNorm();
        const double radial = k1 * r2 + k2 * r2 * r2;
        return p * radial;
    }
};

}  // namespace faster_lio

#endif  // CAMERAS_RADIAL_H

//
// Equidistant fisheye models without distortion, after COLMAP src/colmap/sensor/models/fisheye.h.
//

#ifndef CAMERAS_FISHEYE_H
#define CAMERAS_FISHEYE_H

#include "camera_base.h"

namespace faster_lio {

/// SIMPLE_FISHEYE: f, cx, cy.  Equidistant: (uu, vv) = (u', v') theta / r, theta = atan(r), x = f uu + cx.
class SimpleFisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kSimpleFisheye;
    static constexpr const char* kModelName = "SIMPLE_FISHEYE";
    static constexpr const char* kParamsInfo = "f, cx, cy";
    static constexpr size_t kNumParams = 3;
    SimpleFisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        const Vec2 pp = camera_internal::FisheyeFromCam(X);
        *uv = Vec2(f * pp.x() + c1, f * pp.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *xy = camera_internal::NormalFromFisheye(Vec2((uv.x() - c1) / f, (uv.y() - c2) / f));
        return true;
    }
    /// Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *ray = camera_internal::CamRayFromFisheye(Vec2((uv.x() - c1) / f, (uv.y() - c2) / f));
        return true;
    }
};

/// FISHEYE: fx, fy, cx, cy.  Equidistant with two focal lengths.
class FisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kFisheye;
    static constexpr const char* kModelName = "FISHEYE";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy";
    static constexpr size_t kNumParams = 4;
    FisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const Vec2 pp = camera_internal::FisheyeFromCam(X);
        *uv = Vec2(f1 * pp.x() + c1, f2 * pp.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *xy = camera_internal::NormalFromFisheye(Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2));
        return true;
    }
    /// Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *ray = camera_internal::CamRayFromFisheye(Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2));
        return true;
    }
};

}  // namespace faster_lio

#endif  // CAMERAS_FISHEYE_H

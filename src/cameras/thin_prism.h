//
// Fisheye models with radial, tangential and thin-prism distortion, after COLMAP
// src/colmap/sensor/models/thin_prism.h.
//

#ifndef CAMERAS_THIN_PRISM_H
#define CAMERAS_THIN_PRISM_H

#include "camera_base.h"

namespace faster_lio {

/// THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1.  On the fisheye point (uu, vv):
///   du = uu (k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8) + 2 p1 uu vv + p2 (r^2 + 2 uu^2) + sx1 r^2, dv likewise with sy1
class ThinPrismFisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kThinPrismFisheye;
    static constexpr const char* kModelName = "THIN_PRISM_FISHEYE";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1";
    static constexpr size_t kNumParams = 12;
    ThinPrismFisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const Vec2 pp = camera_internal::FisheyeFromCam(X);
        const Vec2 d = pp + Distortion(&params_[4], pp);
        *uv = Vec2(f1 * d.x() + c1, f2 * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *xy = camera_internal::NormalFromFisheye(pp);
        return true;
    }
    /// Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *ray = camera_internal::CamRayFromFisheye(pp);
        return true;
    }
    /// The undistorted fisheye point (uu, vv) of a pixel.
    bool FisheyeFromImg(const Vec2& uv, Vec2* pp) const {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *pp = Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        const double* extra = &params_[4];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, pp);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k1 = extra[0], k2 = extra[1], p1 = extra[2], p2 = extra[3];
        const double k3 = extra[4], k4 = extra[5], sx1 = extra[6], sy1 = extra[7];
        const double u = p.x(), v = p.y();
        const double u2 = u * u, uv = u * v, v2 = v * v;
        const double r2 = u2 + v2, r4 = r2 * r2, r6 = r4 * r2, r8 = r6 * r2;
        const double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        return {u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2,
                v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2};
    }
};

/// RAD_TAN_THIN_PRISM_FISHEYE (Project Aria Fisheye624): fx, fy, cx, cy, k0..k5, p0, p1, s0..s3.
///   (x, y) = (uu, vv) (1 + sum_i k_i theta^(2 i + 2)), then tangential p0, p1 and thin prism s0..s3 on (x, y).
class RadTanThinPrismFisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kRadTanThinPrismFisheye;
    static constexpr const char* kModelName = "RAD_TAN_THIN_PRISM_FISHEYE";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3";
    static constexpr size_t kNumParams = 16;
    RadTanThinPrismFisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const Vec2 pp = camera_internal::FisheyeFromCam(X);
        const Vec2 d = pp + Distortion(&params_[4], pp);
        *uv = Vec2(f1 * d.x() + c1, f2 * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *xy = camera_internal::NormalFromFisheye(pp);
        return true;
    }
    /// Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *ray = camera_internal::CamRayFromFisheye(pp);
        return true;
    }
    /// The undistorted fisheye point (uu, vv) of a pixel.
    bool FisheyeFromImg(const Vec2& uv, Vec2* pp) const {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *pp = Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        const double* extra = &params_[4];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, pp);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        constexpr int kNumRadialParams = 6;
        const double p0 = extra[6], p1 = extra[7];
        const double s0 = extra[8], s1 = extra[9], s2 = extra[10], s3 = extra[11];
        const double theta2 = p.squaredNorm();
        double th_radial = 1.0, theta_power = 1.0;
        for (int i = 0; i < kNumRadialParams; ++i) {
            theta_power *= theta2;
            th_radial += extra[i] * theta_power;
        }
        const double x = th_radial * p.x(), y = th_radial * p.y();
        const double x2 = x * x, y2 = y * y, xy = x * y;
        const double r2 = x2 + y2, r4 = r2 * r2;
        const double dx_tang = 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2);
        const double dy_tang = 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2);
        const double dx_tp = s0 * r2 + s1 * r4;
        const double dy_tp = s2 * r2 + s3 * r4;
        return {x + dx_tang + dx_tp - p.x(), y + dy_tang + dy_tp - p.y()};
    }
};

}  // namespace faster_lio

#endif  // CAMERAS_THIN_PRISM_H

// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "camera_base.h"

namespace kuma_lio {

// OPENCV_FISHEYE (Kannala-Brandt): fx, fy, cx, cy, k1, k2, k3, k4.
//   theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
class OpenCVFisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kOpenCVFisheye;
    static constexpr const char* kModelName = "OPENCV_FISHEYE";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, k1, k2, k3, k4";
    static constexpr size_t kNumParams = 8;
    OpenCVFisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
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
    // Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *ray = camera_internal::CamRayFromFisheye(pp);
        return true;
    }
    // The undistorted fisheye point (uu, vv) of a pixel.
    bool FisheyeFromImg(const Vec2& uv, Vec2* pp) const {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *pp = Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        const double* extra = &params_[4];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, pp);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k1 = extra[0], k2 = extra[1], k3 = extra[2], k4 = extra[3];
        const double t2 = p.squaredNorm(), t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
        const double radial = k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
        return p * radial;
    }
};

// SIMPLE_RADIAL_FISHEYE: f, cx, cy, k.  theta_d = theta (1 + k theta^2).
class SimpleRadialFisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kSimpleRadialFisheye;
    static constexpr const char* kModelName = "SIMPLE_RADIAL_FISHEYE";
    static constexpr const char* kParamsInfo = "f, cx, cy, k";
    static constexpr size_t kNumParams = 4;
    SimpleRadialFisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        const Vec2 pp = camera_internal::FisheyeFromCam(X);
        const Vec2 d = pp + Distortion(&params_[3], pp);
        *uv = Vec2(f * d.x() + c1, f * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *xy = camera_internal::NormalFromFisheye(pp);
        return true;
    }
    // Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *ray = camera_internal::CamRayFromFisheye(pp);
        return true;
    }
    // The undistorted fisheye point (uu, vv) of a pixel.
    bool FisheyeFromImg(const Vec2& uv, Vec2* pp) const {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *pp = Vec2((uv.x() - c1) / f, (uv.y() - c2) / f);
        const double* extra = &params_[3];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, pp);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k = extra[0];
        return p * (k * p.squaredNorm());
    }
};

// RADIAL_FISHEYE: f, cx, cy, k1, k2.  theta_d = theta (1 + k1 theta^2 + k2 theta^4).
class RadialFisheyeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kRadialFisheye;
    static constexpr const char* kModelName = "RADIAL_FISHEYE";
    static constexpr const char* kParamsInfo = "f, cx, cy, k1, k2";
    static constexpr size_t kNumParams = 5;
    RadialFisheyeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        const Vec2 pp = camera_internal::FisheyeFromCam(X);
        const Vec2 d = pp + Distortion(&params_[3], pp);
        *uv = Vec2(f * d.x() + c1, f * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *xy = camera_internal::NormalFromFisheye(pp);
        return true;
    }
    // Unlike COLMAP, which goes through CamFromImg, also the rays at 90 degrees and beyond.
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        Vec2 pp;
        if (!FisheyeFromImg(uv, &pp)) return false;
        *ray = camera_internal::CamRayFromFisheye(pp);
        return true;
    }
    // The undistorted fisheye point (uu, vv) of a pixel.
    bool FisheyeFromImg(const Vec2& uv, Vec2* pp) const {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *pp = Vec2((uv.x() - c1) / f, (uv.y() - c2) / f);
        const double* extra = &params_[3];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, pp);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k1 = extra[0], k2 = extra[1];
        const double t2 = p.squaredNorm();
        return p * (k1 * t2 + k2 * t2 * t2);
    }
};

}  // namespace kuma_lio

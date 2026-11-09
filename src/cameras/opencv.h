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

// OPENCV: fx, fy, cx, cy, k1, k2, p1, p2.
//   du = u (k1 r^2 + k2 r^4) + 2 p1 u v + p2 (r^2 + 2 u^2)
//   dv = v (k1 r^2 + k2 r^4) + 2 p2 u v + p1 (r^2 + 2 v^2)
class OpenCVCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kOpenCV;
    static constexpr const char* kModelName = "OPENCV";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, k1, k2, p1, p2";
    static constexpr size_t kNumParams = 8;
    OpenCVCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const Vec2 p = X.hnormalized();
        const Vec2 d = p + Distortion(&params_[4], p);
        *uv = Vec2(f1 * d.x() + c1, f2 * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *xy = Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        const double* extra = &params_[4];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, xy);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k1 = extra[0], k2 = extra[1], p1 = extra[2], p2 = extra[3];
        const double u = p.x(), v = p.y();
        const double u2 = u * u, uv = u * v, v2 = v * v;
        const double r2 = u2 + v2;
        const double radial = k1 * r2 + k2 * r2 * r2;
        return {u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2), v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2)};
    }
};

// FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6.  The rational radial term
//   (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)
// plus the tangential terms of OPENCV.
class FullOpenCVCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kFullOpenCV;
    static constexpr const char* kModelName = "FULL_OPENCV";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";
    static constexpr size_t kNumParams = 12;
    FullOpenCVCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const Vec2 p = X.hnormalized();
        const Vec2 d = p + Distortion(&params_[4], p);
        *uv = Vec2(f1 * d.x() + c1, f2 * d.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *xy = Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        const double* extra = &params_[4];
        return camera_internal::IterativeUndistortion([extra](const Vec2& p) { return Distortion(extra, p); }, xy);
    }
    static Vec2 Distortion(const double* extra, const Vec2& p) {
        const double k1 = extra[0], k2 = extra[1], p1 = extra[2], p2 = extra[3];
        const double k3 = extra[4], k4 = extra[5], k5 = extra[6], k6 = extra[7];
        const double u = p.x(), v = p.y();
        const double u2 = u * u, uv = u * v, v2 = v * v;
        const double r2 = u2 + v2, r4 = r2 * r2, r6 = r4 * r2;
        const double radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
        return {u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) - u,
                v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) - v};
    }
};

}  // namespace kuma_lio

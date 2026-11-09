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

namespace camera_internal {
// Normalized pixel of the ray (u, v, w): r p with r = 2 / (w + sqrt(w^2 - 4 k rho^2)), the root of
// rho k r^2 - w r + rho = 0.  No cheirality test, as in COLMAP.  False if the discriminant is negative.
inline bool DivisionFromCam(double k, const Vec3& X, Vec2* p) {
    const double rho2 = X.x() * X.x() + X.y() * X.y();
    const double disc_sq = X.z() * X.z() - 4.0 * rho2 * k;
    if (disc_sq < 0) return false;
    const double r = 2.0 / (X.z() + std::sqrt(disc_sq));
    *p = Vec2(r * X.x(), r * X.y());
    return true;
}
}  // namespace camera_internal

// SIMPLE_DIVISION: f, cx, cy, k.
class SimpleDivisionCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kSimpleDivision;
    static constexpr const char* kModelName = "SIMPLE_DIVISION";
    static constexpr const char* kParamsInfo = "f, cx, cy, k";
    static constexpr size_t kNumParams = 4;
    SimpleDivisionCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool /*check_cheirality*/ = true) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        Vec2 p;
        if (!camera_internal::DivisionFromCam(params_[3], X, &p)) return false;
        *uv = Vec2(f * p.x() + c1, f * p.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2], k = params_[3];
        const Vec2 p((uv.x() - c1) / f, (uv.y() - c2) / f);
        *xy = p / (1.0 + k * p.squaredNorm());
        return true;
    }
};

// DIVISION: fx, fy, cx, cy, k.
class DivisionCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kDivision;
    static constexpr const char* kModelName = "DIVISION";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, k";
    static constexpr size_t kNumParams = 5;
    DivisionCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool /*check_cheirality*/ = true) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        Vec2 p;
        if (!camera_internal::DivisionFromCam(params_[4], X, &p)) return false;
        *uv = Vec2(f1 * p.x() + c1, f2 * p.y() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3], k = params_[4];
        const Vec2 p((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        *xy = p / (1.0 + k * p.squaredNorm());
        return true;
    }
};

}  // namespace kuma_lio

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

// SIMPLE_PINHOLE: f, cx, cy.  x = f u / w + cx.
class SimplePinholeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kSimplePinhole;
    static constexpr const char* kModelName = "SIMPLE_PINHOLE";
    static constexpr const char* kParamsInfo = "f, cx, cy";
    static constexpr size_t kNumParams = 3;
    SimplePinholeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *uv = Vec2(f * X.x() / X.z() + c1, f * X.y() / X.z() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f = params_[0], c1 = params_[1], c2 = params_[2];
        *xy = Vec2((uv.x() - c1) / f, (uv.y() - c2) / f);
        return true;
    }
};

// PINHOLE: fx, fy, cx, cy.
class PinholeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kPinhole;
    static constexpr const char* kModelName = "PINHOLE";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy";
    static constexpr size_t kNumParams = 4;
    PinholeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *uv = Vec2(f1 * X.x() / X.z() + c1, f2 * X.y() / X.z() + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        *xy = Vec2((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        return true;
    }
};

// SKEWED_PINHOLE: fx, fy, cx, cy, s.  x = fx u' + s v' + cx, y = fy v' + cy with (u', v') = (u, v) / w.
class SkewedPinholeCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kSkewedPinhole;
    static constexpr const char* kModelName = "SKEWED_PINHOLE";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, s";
    static constexpr size_t kNumParams = 5;
    SkewedPinholeCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3], s = params_[4];
        const double uu = X.x() / X.z(), vv = X.y() / X.z();
        *uv = Vec2(f1 * uu + s * vv + c1, f2 * vv + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3], s = params_[4];
        const double v = (uv.y() - c2) / f2;
        *xy = Vec2((uv.x() - c1 - s * v) / f1, v);
        return true;
    }
};

}  // namespace kuma_lio

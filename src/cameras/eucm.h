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

// EUCM: fx, fy, cx, cy, alpha, beta.  (u, v) / (alpha rho + (1 - alpha) w), rho = sqrt(beta (u^2 + v^2) + w^2);
// alpha in [0, 1], beta > 0.
class EUCMCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kEUCM;
    static constexpr const char* kModelName = "EUCM";
    static constexpr const char* kParamsInfo = "fx, fy, cx, cy, alpha, beta";
    static constexpr size_t kNumParams = 6;
    EUCMCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const override {
        if (!camera_internal::HasProjectableDepth(X.z(), check_cheirality)) return false;
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const double alpha = params_[4], beta = params_[5];
        const double rho2 = beta * (X.x() * X.x() + X.y() * X.y()) + X.z() * X.z();
        if (rho2 < 0) return false;
        const double den = alpha * std::sqrt(rho2) + (1.0 - alpha) * X.z();
        if (!camera_internal::HasProjectableDepth(den, check_cheirality)) return false;
        *uv = Vec2(f1 * X.x() / den + c1, f2 * X.y() / den + c2);
        return true;
    }
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        const double f1 = params_[0], f2 = params_[1], c1 = params_[2], c2 = params_[3];
        const double alpha = params_[4], beta = params_[5];
        const Vec2 p((uv.x() - c1) / f1, (uv.y() - c2) / f2);
        const double r2 = p.squaredNorm();
        const double gamma = 1.0 - alpha;
        const double radicand = 1.0 - (alpha - gamma) * beta * r2;
        if (radicand < 0) return false;
        const double helper_den = alpha * std::sqrt(radicand) + gamma;
        if (helper_den < std::numeric_limits<double>::epsilon()) return false;
        const double helper = (1.0 - alpha * alpha * beta * r2) / helper_den;
        if (helper < std::numeric_limits<double>::epsilon()) return false;
        *xy = p / helper;
        return true;
    }
};

}  // namespace kuma_lio

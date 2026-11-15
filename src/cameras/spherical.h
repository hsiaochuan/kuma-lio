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

// EQUIRECTANGULAR: w, h (the image size; no focal length).
//   azimuth theta = atan2(u, w) in (-pi, pi], from +Z (forward), +X is +pi/2
//   elevation phi = atan2(-v, sqrt(u^2 + w^2)) in [-pi/2, pi/2], -Y (up) is +pi/2
//   x = (theta / 2 pi + 1/2) W,  y = (1/2 - phi / pi) H
// Every non-zero direction projects, behind the camera too.
class EquirectangularCamera : public CamModel {
   public:
    static constexpr CameraModelId kModelId = CameraModelId::kEquirectangular;
    static constexpr const char* kModelName = "EQUIRECTANGULAR";
    static constexpr const char* kParamsInfo = "w, h";
    static constexpr size_t kNumParams = 2;
    EquirectangularCamera(unsigned int w, unsigned int h, std::vector<double> params)
        : CamModel(kModelId, kModelName, kParamsInfo, kNumParams, w, h, std::move(params)) {}

    bool ImgFromCam(const Vec3& X, Vec2* uv, bool /*check_cheirality*/ = true) const override {
        const double width = params_[0], height = params_[1];
        const double horizontal = std::hypot(X.x(), X.z());
        if (horizontal + std::abs(X.y()) < std::numeric_limits<double>::epsilon()) return false;  // zero vector
        const double theta = std::atan2(X.x(), X.z());
        const double phi = std::atan2(-X.y(), horizontal);
        *uv = Vec2((theta / (2.0 * EIGEN_PI) + 0.5) * width, (0.5 - phi / EIGEN_PI) * height);
        return true;
    }
    // The normalized plane point of a pixel, only for rays in front of the camera (Z > 0); CamRayFromImg
    // covers the whole sphere.
    bool CamFromImg(const Vec2& uv, Vec2* xy) const override {
        Vec3 ray;
        CamRayFromImg(uv, &ray);
        if (ray.z() <= std::numeric_limits<double>::epsilon()) return false;
        *xy = ray.hnormalized();
        return true;
    }
    bool CamRayFromImg(const Vec2& uv, Vec3* ray) const override {
        const double width = params_[0], height = params_[1];
        const double theta = 2.0 * EIGEN_PI * (uv.x() / width - 0.5);
        const double phi = EIGEN_PI * (0.5 - uv.y() / height);
        const double cos_phi = std::cos(phi);
        *ray = Vec3(cos_phi * std::sin(theta), -std::sin(phi), cos_phi * std::cos(theta));
        return true;
    }
};

}  // namespace kuma_lio

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

#include <glog/logging.h>

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "eigen_type.h"

namespace kuma_lio {

// COLMAP's CameraModelId, same values, so a model id read from a COLMAP database means the same here.
enum class CameraModelId {
    kInvalid = -1,
    kSimplePinhole = 0,
    kPinhole = 1,
    kSimpleRadial = 2,
    kRadial = 3,
    kOpenCV = 4,
    kOpenCVFisheye = 5,
    kFullOpenCV = 6,
    kFOV = 7,
    kSimpleRadialFisheye = 8,
    kRadialFisheye = 9,
    kThinPrismFisheye = 10,
    kRadTanThinPrismFisheye = 11,
    kSimpleDivision = 12,
    kDivision = 13,
    kSimpleFisheye = 14,
    kFisheye = 15,
    kEUCM = 16,
    kEquirectangular = 17,
    kSkewedPinhole = 18,
};

/**
 * @brief Abstract base of every camera model.  Each model derives directly from it (one level of inheritance).
 *
 * The parameters are kept in COLMAP's order (cameras.txt), see params_info().  A model implements
 *   ImgFromCam: camera frame point (u, v, w) -> pixel (x, y)
 *   CamFromImg: pixel -> normalized image plane (u / w, v / w)
 * and may override CamRayFromImg (pixel -> unit ray), which by default goes through CamFromImg and so only
 * covers rays in front of the camera.
 */
class CamModel {
   public:
    using Ptr = std::shared_ptr<CamModel>;
    virtual ~CamModel() = default;

    // -- Description ---------------------------------------------------------
    CameraModelId model_id() const { return model_id_; }
    const std::string& model_name() const { return model_name_; }
    const std::string& params_info() const { return params_info_; }  // e.g. "fx, fy, cx, cy, k1, k2, p1, p2"
    const std::vector<double>& params() const { return params_; }
    unsigned int w() const { return w_; }
    unsigned int h() const { return h_; }

    // -- Projection, implemented by each model -------------------------------
    // Pixel of the camera frame point X; false if X cannot be projected (behind the camera with
    // `check_cheirality`, or outside the model's domain).
    virtual bool ImgFromCam(const Vec3& X, Vec2* uv, bool check_cheirality = true) const = 0;
    // Normalized image plane point (X / Z, Y / Z) of a pixel; false if the pixel has none.
    virtual bool CamFromImg(const Vec2& uv, Vec2* xy) const = 0;
    // Unit ray of a pixel in the camera frame; false if the pixel has none.
    virtual bool CamRayFromImg(const Vec2& uv, Vec3* ray) const {
        Vec2 xy;
        if (!CamFromImg(uv, &xy)) return false;
        *ray = xy.homogeneous().normalized();
        return true;
    }

    // -- Convenience ---------------------------------------------------------
    // Pixel of X, NaN if it cannot be projected.
    Vec2 project(const Vec3& X) const {
        Vec2 uv;
        return ImgFromCam(X, &uv) ? uv : Vec2::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    // Unit ray of a pixel, NaN if it has none.
    Vec3 bearing(const Vec2& uv) const {
        Vec3 ray;
        return CamRayFromImg(uv, &ray) ? ray : Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    // Inside the image, `border` pixels away from its edges.
    bool valid(const Vec2& uv, int border = 0) const {
        return std::isfinite(uv.x()) && std::isfinite(uv.y()) && uv.x() >= border && uv.x() < double(w_) - border &&
               uv.y() >= border && uv.y() < double(h_) - border;
    }
    std::optional<Vec2> project_and_valid(const Vec3& X, int border = 0) const {
        Vec2 uv;
        if (!ImgFromCam(X, &uv) || !valid(uv, border)) return std::nullopt;
        return uv;
    }

   protected:
    CamModel(CameraModelId model_id, std::string model_name, std::string params_info, size_t num_params,
             unsigned int w, unsigned int h, std::vector<double> params)
        : model_id_(model_id),
          model_name_(std::move(model_name)),
          params_info_(std::move(params_info)),
          w_(w),
          h_(h),
          params_(std::move(params)) {
        CHECK_EQ(params_.size(), num_params) << model_name_ << " takes " << num_params << " parameters: "
                                             << params_info_;
    }

    CameraModelId model_id_;
    std::string model_name_;
    std::string params_info_;
    unsigned int w_;
    unsigned int h_;
    std::vector<double> params_;  // COLMAP order
};

inline bool IsSpherical(CameraModelId m) { return m == CameraModelId::kEquirectangular; }
// The models that project through the fisheye angle theta = atan(r) (COLMAP PERSPECTIVE_FISHEYE_CAMERA_MODEL_CASES).
inline bool IsFisheye(CameraModelId m) {
    return m == CameraModelId::kSimpleRadialFisheye || m == CameraModelId::kRadialFisheye ||
           m == CameraModelId::kOpenCVFisheye || m == CameraModelId::kThinPrismFisheye ||
           m == CameraModelId::kRadTanThinPrismFisheye || m == CameraModelId::kSimpleFisheye ||
           m == CameraModelId::kFisheye;
}

namespace camera_internal {

// COLMAP HasProjectableDepth.
inline bool HasProjectableDepth(double w, bool check_cheirality) {
    return check_cheirality ? w >= std::numeric_limits<double>::epsilon()
                            : std::abs(w) >= std::numeric_limits<double>::epsilon();
}

// COLMAP BasePerspectiveFisheyeCameraModel::FisheyeFromNormal: (u, v) * theta / r with theta = atan(r).
inline Vec2 FisheyeFromNormal(const Vec2& p) {
    const double r = p.norm();
    if (r > std::numeric_limits<double>::epsilon()) return p * (std::atan(r) / r);
    return p;
}

// FisheyeFromNormal of X / Z, written with theta = atan2(|xy|, z): the same for z > 0, and it also holds for
// rays at 90 degrees and beyond, which the normalized plane cannot represent.
inline Vec2 FisheyeFromCam(const Vec3& X) {
    const double r = X.head<2>().norm();
    if (r > std::numeric_limits<double>::epsilon()) return X.head<2>() * (std::atan2(r, X.z()) / r);
    return X.head<2>() / X.z();
}

// Unit ray of a fisheye point (uu, vv): theta = |(uu, vv)| off the optical axis, for theta up to pi.
inline Vec3 CamRayFromFisheye(const Vec2& pp) {
    const double theta = pp.norm();
    if (theta > std::numeric_limits<double>::epsilon()) {
        const Vec2 xy = pp * (std::sin(theta) / theta);
        return Vec3(xy.x(), xy.y(), std::cos(theta));
    }
    return pp.homogeneous().normalized();
}

// COLMAP BasePerspectiveFisheyeCameraModel::NormalFromFisheye: the inverse, (uu, vv) * tan(theta) / theta.
inline Vec2 NormalFromFisheye(const Vec2& pp) {
    const double theta = pp.norm();
    const double theta_cos_theta = theta * std::cos(theta);
    if (theta_cos_theta > std::numeric_limits<double>::epsilon()) return pp * (std::sin(theta) / theta_cos_theta);
    return pp;
}

/**
 * @brief COLMAP BasePerspectiveCameraModel::IterativeUndistortion: solves x + D(x) = x0 for x by Newton steps
 * inside a trust region.  `distortion(x)` returns D(x).  COLMAP differentiates D with ceres Jets; here the
 * 2x2 Jacobian is a central difference.  On return *p is the last iterate; false if it did not converge.
 */
template <typename Distortion>
bool IterativeUndistortion(const Distortion& distortion, Vec2* p) {
    constexpr int kNumIterations = 100;
    constexpr double kMinStepSquaredNorm = 1e-10;
    constexpr double kRelStepRadius = 0.1;  // trust region: |step| <= max(|x| * kRelStepRadius, kStepRadius)
    constexpr double kStepRadius = 0.1;

    const Vec2 x0 = *p;
    Vec2 x = *p;
    for (int i = 0; i < kNumIterations; ++i) {
        const Vec2 dx = distortion(x);
        Eigen::Matrix2d J;
        for (int k = 0; k < 2; ++k) {
            const double step = 1e-6 * std::max(1.0, std::abs(x[k]));
            Vec2 xp = x, xm = x;
            xp[k] += step;
            xm[k] -= step;
            J.col(k) = (distortion(xp) - distortion(xm)) / (2.0 * step);
        }
        J += Eigen::Matrix2d::Identity();

        Vec2 step_x = J.partialPivLu().solve(x + dx - x0);
        const double radius_sqr =
            std::max(x.squaredNorm() * kRelStepRadius * kRelStepRadius, kStepRadius * kStepRadius);
        const double step_x_norm_sqr = step_x.squaredNorm();
        if (step_x_norm_sqr > radius_sqr) step_x *= std::sqrt(radius_sqr / step_x_norm_sqr);
        x -= step_x;
        if (step_x.squaredNorm() < kMinStepSquaredNorm) {
            *p = x;
            return true;
        }
    }
    *p = x;
    return false;
}

}  // namespace camera_internal
}  // namespace kuma_lio

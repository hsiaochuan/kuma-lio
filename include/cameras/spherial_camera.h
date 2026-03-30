//
// Created by hsiaochuan on 2026/03/30.
//

#ifndef SPHERIAL_CAMERA_H
#define SPHERIAL_CAMERA_H
#include "cameras.h"
namespace faster_lio {
class Intrinsic_Spherical : public IntrinsicBase {
    using class_type = Intrinsic_Spherical;

   public:
    /**
     * @brief Constructor
     * @param w Width of the image plane
     * @param h Height of the image plane
     */
    Intrinsic_Spherical(unsigned int w = 0, unsigned int h = 0) : IntrinsicBase(w, h) {}

    ~Intrinsic_Spherical() override = default;

    /**
     * @brief Tell from which type the embed camera is
     * @retval CAMERA_SPHERICAL
     */
    CAMERA_INTRINSIC getType() const override { return CAMERA_SPHERICAL; }

    /**
     * @brief Data wrapper for non linear optimization (get data)
     * @return an empty vector of parameter since a spherical camera does not have any intrinsic parameter
     */
    std::vector<double> getParams() const override { return {}; }

    /**
     * @brief Data wrapper for non linear optimization (update from data)
     * @param params List of params used to update this intrinsic
     * @retval true if update is correct
     * @retval false if there was an error during update
     */
    bool updateFromParams(const std::vector<double>& params) override { return true; }

    /**
     * @brief Transform a point from the camera plane to the image plane
     * @param p Camera plane point
     * @return Point on image plane
     */
    common::V2D cam2ima(const common::V2D& p) const override {
        const double size(std::max(w(), h()));
        return {p.x() * size + w() / 2.0, p.y() * size + h() / 2.0};
    }

    /**
     * @brief Transform a point from the image plane to the camera plane
     * @param p Image plane point
     * @return camera plane point
     */
    common::V2D ima2cam(const common::V2D& p) const override {
        const double size(std::max(w(), h()));
        return {(p.x() - w() / 2.0) / size, (p.y() - h() / 2.0) / size};
    }

    /**
     * @brief Compute projection of a 3D point into the image plane
     * (Apply disto (if any) and Intrinsics)
     * @param pt3D 3D-point to project on image plane
     * @return Projected (2D) point on image plane
     */
    common::V2D project(const common::V3D& X, const bool ignore_distortion = false) const override {
        const double lon = std::atan2(X.x(), X.z());  // Horizontal normalization of the  X-Z component
        const double lat = std::atan2(-X.y(), std::hypot(X.x(), X.z()));  // Tilt angle
        // denormalization (angle to pixel value)
        return cam2ima({lon / (2 * M_PI), -lat / (2 * M_PI)});
    }

    /**
     * @brief Does the camera model handle a distortion field?
     * @retval false
     */
    virtual bool have_disto() const override { return false; }

    /**
     * @brief Add the distortion field to a point (that is in normalized camera frame)
     * @param p Point before distortion computation (in normalized camera frame)
     * @return the initial point p (spherical camera does not have distortion field)
     */
    virtual common::V2D add_disto(const common::V2D& p) const override { return p; }

    /**
     * @brief Remove the distortion to a camera point (that is in normalized camera frame)
     * @param p Point with distortion
     * @return the initial point p (spherical camera does not have distortion field)
     */
    virtual common::V2D remove_disto(const common::V2D& p) const override { return p; }

    /**
     * @brief Return the un-distorted pixel (with removed distortion)
     * @param p Input distorted pixel
     * @return Point without distortion
     */
    virtual common::V2D get_ud_pixel(const common::V2D& p) const override { return p; }

    /**
     * @brief Return the distorted pixel (with added distortion)
     * @param p Input pixel
     * @return Distorted pixel
     */
    virtual common::V2D get_d_pixel(const common::V2D& p) const override { return p; }
};
}  // namespace faster_lio
#endif  // SPHERIAL_CAMERA_H

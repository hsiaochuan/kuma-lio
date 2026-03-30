//
// Created by hsiaochuan on 2026/03/30.
//

#ifndef CAMERAS_H
#define CAMERAS_H
#include "common_lib.h"
namespace faster_lio {
enum CAMERA_INTRINSIC {
    PINHOLE_CAMERA = 0,      // No distortion
    PINHOLE_CAMERA_RADIAL3,  // radial distortion K1,K2,K3
    PINHOLE_CAMERA_BROWN,
    PINHOLE_CAMERA_FISHEYE,  // a simple Fish-eye distortion model with 4 distortion coefficients
    CAMERA_SPHERICAL,
};

static inline bool IsPinhole(CAMERA_INTRINSIC intrinsic) { return intrinsic <= PINHOLE_CAMERA_FISHEYE; }

static inline bool IsSpherical(CAMERA_INTRINSIC intrinsic) { return intrinsic == CAMERA_SPHERICAL; }
/**
 * @brief Base class used to store common intrinsics parameters
 */
struct IntrinsicBase {
    /// Width of image
    unsigned int w_;
    /// Height of image
    unsigned int h_;

    /**
     * @brief Constructor
     * @param w Width of the image
     * @param h Height of the image
     */
    IntrinsicBase(unsigned int w = 0, unsigned int h = 0) : w_(w), h_(h) {}

    /**
     * @brief Destructor
     */
    virtual ~IntrinsicBase() = default;

    /**
     * @brief Get width of the image
     * @return width of the image
     */
    unsigned int w() const { return w_; }

    /**
     * @brief Get height of the image
     * @return height of the image
     */
    unsigned int h() const { return h_; }

    /**
     * @brief Compute projection of a 3D point into the image plane
     * (Apply disto (if any) and Intrinsics)
     * @param X 3D-point to project on image plane
     * @return Projected (2D) point on image plane
     */
    virtual common::V2D project(const common::V3D& X, const bool ignore_distortion = false) const {
        if (this->have_disto() && !ignore_distortion)  // apply disto & intrinsics
        {
            return this->cam2ima(this->add_disto(X.hnormalized()));
        } else  // apply intrinsics
        {
            return this->cam2ima(X.hnormalized());
        }
    }

    /**
     * @brief Compute the residual between the 3D projected point and an image observation
     * @param X 3d point to project on camera plane
     * @param x image observation
     * @brief Relative 2d distance between projected and observed points
     */
    common::V2D residual(const common::V3D& X, const common::V2D& x, const bool ignore_distortion = false) const {
        const common::V2D proj = this->project(X, ignore_distortion);
        return x - proj;
    }

    // --
    // Virtual members
    // --

    /**
     * @brief Tell from which type the embed camera is
     * @return Corresponding intrinsic
     */
    virtual CAMERA_INTRINSIC getType() const = 0;

    /**
     * @brief Data wrapper for non linear optimization (get data)
     * @return vector of parameter of this intrinsic
     */
    virtual std::vector<double> getParams() const = 0;

    /**
     * @brief Data wrapper for non linear optimization (update from data)
     * @param params List of params used to update this intrinsic
     * @retval true if update is correct
     * @retval false if there was an error during update
     */
    virtual bool updateFromParams(const std::vector<double>& params) = 0;

    /**
     * @brief Transform a point from the camera plane to the image plane
     * @param p Camera plane point
     * @return Point on image plane
     */
    virtual common::V2D cam2ima(const common::V2D& p) const = 0;

    /**
     * @brief Transform a point from the image plane to the camera plane
     * @param p Image plane point
     * @return camera plane point
     */
    virtual common::V2D ima2cam(const common::V2D& p) const = 0;

    /**
     * @brief Does the camera model handle a distortion field?
     * @retval true if intrinsic holds distortion
     * @retval false if intrinsic does not hold distortion
     */
    virtual bool have_disto() const = 0;

    /**
     * @brief Add the distortion field to a point (that is in normalized camera frame)
     * @param p Point before distortion computation (in normalized camera frame)
     * @return point with distortion
     */
    virtual common::V2D add_disto(const common::V2D& p) const = 0;

    /**
     * @brief Remove the distortion to a camera point (that is in normalized camera frame)
     * @param p Point with distortion
     * @return Point without distortion
     */
    virtual common::V2D remove_disto(const common::V2D& p) const = 0;

    /**
     * @brief Return the un-distorted pixel (with removed distortion)
     * @param p Input distorted pixel
     * @return Point without distortion
     */
    virtual common::V2D get_ud_pixel(const common::V2D& p) const = 0;

    /**
     * @brief Return the distorted pixel (with added distortion)
     * @param p Input pixel
     * @return Distorted pixel
     */
    virtual common::V2D get_d_pixel(const common::V2D& p) const = 0;
};
}  // namespace faster_lio

#endif  // CAMERAS_H

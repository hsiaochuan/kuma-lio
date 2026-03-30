//
// Created by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_CAMERA_H
#define PINHOLE_CAMERA_H
#include "cameras.h"

namespace faster_lio{
class Pinhole_Intrinsic : public IntrinsicBase {
    using class_type = Pinhole_Intrinsic;

   protected:
    /// Intrinsic matrix : Focal & principal point are embed into the calibration matrix K
    common::M3D K_;

    /// Inverse of intrinsic matrix
    common::M3D Kinv_;

   public:
    /**
     * @brief Constructor
     * @param w Width of the image plane
     * @param h Height of the image plane
     * @param focal_length_pix Focal length (in pixel) of the camera
     * @param ppx Principal point on x-axis
     * @param ppy Principal point on y-axis
     */
    Pinhole_Intrinsic(unsigned int w = 0, unsigned int h = 0, double focal_length_pix = 0.0, double ppx = 0.0,
                      double ppy = 0.0)
        : IntrinsicBase(w, h) {
        K_ << focal_length_pix, 0., ppx, 0., focal_length_pix, ppy, 0., 0., 1.;
        Kinv_ = K_.inverse();
    }

    /**
     * @brief Constructor
     * @param w Width of the image plane
     * @param h Height of the image plane
     * @param K Intrinsic Matrix (3x3) {f,0,ppx; 0,f,ppy; 0,0,1}
     */
    Pinhole_Intrinsic(unsigned int w, unsigned int h, const common::M3D& K) : IntrinsicBase(w, h), K_(K) {
        K_(0, 0) = K_(1, 1) = (K(0, 0) + K(1, 1)) / 2.0;
        Kinv_ = K_.inverse();
    }

    /**
     * @brief Destructor
     */
    ~Pinhole_Intrinsic() override = default;

    /**
     * @brief Get type of the intrinsic
     * @retval PINHOLE_CAMERA
     */
    CAMERA_INTRINSIC getType() const override { return PINHOLE_CAMERA; }

    /**
     * @brief Get the intrinsic matrix
     * @return 3x3 intrinsic matrix
     */
    const common::M3D& K() const { return K_; }

    /**
     * @brief Get the inverse of the intrinsic matrix
     * @return Inverse of intrinsic matrix
     */
    const common::M3D& Kinv() const { return Kinv_; }

    /**
     * @brief Return the value of the focal in pixels
     * @return Focal of the camera (in pixel)
     */
    inline double focal() const { return K_(0, 0); }

    /**
     * @brief Get principal point of the camera
     * @return Principal point of the camera
     */
    inline common::V2D principal_point() const { return {K_(0, 2), K_(1, 2)}; }

    /**
     * @brief Transform a point from the camera plane to the image plane
     * @param p Camera plane point
     * @return Point on image plane
     */
    common::V2D cam2ima(const common::V2D& p) const override { return focal() * p + principal_point(); }

    /**
     * @brief Transform a point from the image plane to the camera plane
     * @param p Image plane point
     * @return camera plane point
     */
    common::V2D ima2cam(const common::V2D& p) const override { return (p - principal_point()) / focal(); }

    /**
     * @brief Does the camera model handle a distortion field?
     * @retval false if intrinsic does not hold distortion
     */
    bool have_disto() const override { return false; }

    /**
     * @brief Add the distortion field to a point (that is in normalized camera frame)
     * @param p Point before distortion computation (in normalized camera frame)
     * @return point with distortion
     */
    common::V2D add_disto(const common::V2D& p) const override { return p; }

    /**
     * @brief Remove the distortion to a camera point (that is in normalized camera frame)
     * @param p Point with distortion
     * @return Point without distortion
     */
    common::V2D remove_disto(const common::V2D& p) const override { return p; }

    /**
     * @brief Data wrapper for non linear optimization (get data)
     * @return vector of parameter of this intrinsic
     */
    std::vector<double> getParams() const override { return {K_(0, 0), K_(0, 2), K_(1, 2)}; }

    /**
     * @brief Data wrapper for non linear optimization (update from data)
     * @param params List of params used to update this intrinsic
     * @retval true if update is correct
     * @retval false if there was an error during update
     */
    bool updateFromParams(const std::vector<double>& params) override {
        if (params.size() == 3) {
            *this = Pinhole_Intrinsic(w_, h_, params[0], params[1], params[2]);
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief Return the un-distorted pixel (with removed distortion)
     * @param p Input distorted pixel
     * @return Point without distortion
     */
    common::V2D get_ud_pixel(const common::V2D& p) const override { return p; }

    /**
     * @brief Return the distorted pixel (with added distortion)
     * @param p Input pixel
     * @return Distorted pixel
     */
    common::V2D get_d_pixel(const common::V2D& p) const override { return p; }
};
}

#endif //PINHOLE_CAMERA_H

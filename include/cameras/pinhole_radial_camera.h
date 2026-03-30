//
// Created by hsiaochuan on 2026/03/30.
//

#ifndef PINHOLE_RADIAL_CAMERA_H
#define PINHOLE_RADIAL_CAMERA_H
#include "pinhole_camera.h"
namespace faster_lio {
template <class Disto_Functor>
double bisection_Radius_Solve(const std::vector<double>& params,  // radial distortion parameters
                              double r2,                          // targeted radius
                              Disto_Functor& functor,
                              double epsilon = 1e-10  // criteria to stop the bisection
) {
    // Guess plausible upper and lower bound
    double lowerbound = r2, upbound = r2;
    while (functor(params, lowerbound) > r2) {
        lowerbound /= 1.05;
    }
    while (functor(params, upbound) < r2) {
        upbound *= 1.05;
    }

    // Perform a bisection until epsilon accuracy is not reached
    while (epsilon < upbound - lowerbound) {
        const double mid = .5 * (lowerbound + upbound);
        if (functor(params, mid) > r2) {
            upbound = mid;
        } else {
            lowerbound = mid;
        }
    }
    return .5 * (lowerbound + upbound);
}
/**
 * @brief Implement a Pinhole camera with a 3 radial distortion coefficients.
 * \f$ x_d = x_u (1 + K_1 r^2 + K_2 r^4 + K_3 r^6) \f$
 */
class Pinhole_Intrinsic_Radial_K3 : public Pinhole_Intrinsic {
    using class_type = Pinhole_Intrinsic_Radial_K3;

   protected:
    // center of distortion is applied by the Intrinsics class
    /// K1, K2, K3
    std::vector<double> params_;

   public:
    /**
     * @brief Constructor
     * @param w Width of image
     * @param h Height of image
     * @param focal Focal (in pixel) of the camera
     * @param ppx Principal point on X-Axis
     * @param ppy Principal point on Y-Axis
     * @param k1 First radial distortion coefficient
     * @param k2 Second radial distortion coefficient
     * @param k3 Third radial distortion coefficient
     */
    Pinhole_Intrinsic_Radial_K3(int w = 0, int h = 0, double focal = 0.0, double ppx = 0, double ppy = 0,
                                double k1 = 0.0, double k2 = 0.0, double k3 = 0.0)
        : Pinhole_Intrinsic(w, h, focal, ppx, ppy), params_({k1, k2, k3}) {}

    ~Pinhole_Intrinsic_Radial_K3() override = default;

    /**
     * @brief Tell from which type the embed camera is
     * @retval PINHOLE_CAMERA_RADIAL3
     */
    CAMERA_INTRINSIC getType() const override { return PINHOLE_CAMERA_RADIAL3; }

    /**
     * @brief Does the camera model handle a distortion field?
     * @retval true
     */
    bool have_disto() const override { return true; }

    /**
     * @brief Add the distortion field to a point (that is in normalized camera frame)
     * @param p Point before distortion computation (in normalized camera frame)
     * @return point with distortion
     */
    common::V2D add_disto(const common::V2D& p) const override {
        const double &k1 = params_[0], &k2 = params_[1], &k3 = params_[2];

        const double r2 = p(0) * p(0) + p(1) * p(1);
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r_coeff = (1. + k1 * r2 + k2 * r4 + k3 * r6);

        return (p * r_coeff);
    }

    /**
     * @brief Remove the distortion to a camera point (that is in normalized camera frame)
     * @param p Point with distortion
     * @return Point without distortion
     */
    common::V2D remove_disto(const common::V2D& p) const override {
        // Compute the radius from which the point p comes from thanks to a bisection
        // Minimize disto(radius(p')^2) == actual Squared(radius(p))

        const double r2 = p(0) * p(0) + p(1) * p(1);
        const double radius = (r2 == 0) ?  // 1. : ::sqrt(bisectionSolve(_params, r2) / r2);
                                  1.
                                        : std::sqrt(bisection_Radius_Solve(params_, r2, distoFunctor) / r2);
        return radius * p;
    }

    /**
     * @brief Data wrapper for non linear optimization (get data)
     * @return vector of parameter of this intrinsic
     */
    std::vector<double> getParams() const override {
        std::vector<double> params = Pinhole_Intrinsic::getParams();
        params.insert(params.end(), std::begin(params_), std::end(params_));
        return params;
    }

    /**
     * @brief Data wrapper for non linear optimization (update from data)
     * @param params List of params used to update this intrinsic
     * @retval true if update is correct
     * @retval false if there was an error during update
     */
    bool updateFromParams(const std::vector<double>& params) override {
        if (params.size() == 6) {
            *this = Pinhole_Intrinsic_Radial_K3(w_, h_, params[0], params[1], params[2],  // focal, ppx, ppy
                                                params[3], params[4], params[5]);         // K1, K2, K3
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
    common::V2D get_ud_pixel(const common::V2D& p) const override { return cam2ima(remove_disto(ima2cam(p))); }

    /**
     * @brief Return the distorted pixel (with added distortion)
     * @param p Input pixel
     * @return Distorted pixel
     */
    common::V2D get_d_pixel(const common::V2D& p) const override { return cam2ima(add_disto(ima2cam(p))); }

   private:
    /**
     * @brief Functor to solve Square(disto(radius(p'))) = r^2
     * @param params List of the radial factors {k1, k2, k3}
     * @param r2 square distance (relative to center)
     * @return distance
     */
    static inline double distoFunctor(const std::vector<double>& params, double r2) {
        const double &k1 = params[0], &k2 = params[1], &k3 = params[2];
        double r = 1. + r2 * (k1 + r2 * (k2 + r2 * k3));
        return r2 * r * r;
    }
};
}
#endif //PINHOLE_RADIAL_CAMERA_H

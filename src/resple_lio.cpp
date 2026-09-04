//
// Created by hsiaochuan on 2026/08/23.
//
#include "resple_lio.h"
const Eigen::Matrix4d SplineState::base_coefficients =
    SplineState::computeBaseCoefficients();
const Eigen::Matrix4d SplineState::blending_matrix =
    SplineState::computeBlendingMatrix();
const Eigen::Matrix4d SplineState::cumulative_blending_matrix =
    SplineState::computeBlendingMatrix<true>();
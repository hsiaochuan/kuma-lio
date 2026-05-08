#pragma once
#include <Eigen/Eigen>
#include "so3_math.h"
namespace faster_lio {
struct State {
  using Ptr = std::shared_ptr<State>;
  enum {
    STATE_DOF = 18,
  };
  using MatrixN = Eigen::Matrix<double, STATE_DOF, STATE_DOF>;
  using VectorN = Eigen::Matrix<double, STATE_DOF, 1>;
  enum {
    ROT = 0,
    POS = 3,
    VEL = 6,
    BIAS_G = 9,
    BIAS_A = 12,
    GRAVITY = 15,
  };
  State() {
    timestamp = -1.0;
    this->rot_end = Eigen::Quaterniond::Identity();
    this->pos_end = common::V3D::Zero();
    this->vel_end = common::V3D::Zero();
    this->bias_g =  common::V3D::Zero();
    this->bias_a =  common::V3D::Zero();
    this->gravity = common::V3D::Zero();

    // init cov
    this->cov = Eigen::Matrix<double, STATE_DOF, STATE_DOF>::Identity() * 0.01;
    this->cov.block<3, 3>(ROT, ROT) = common::M3D::Identity() * 1e-3;
    this->cov.block<3, 3>(POS, POS) = common::M3D::Identity() * 1e-12;
    this->cov.block<3, 3>(VEL, VEL) = common::M3D::Identity() * 1e-12;
    this->cov.block<3, 3>(BIAS_G, BIAS_G) = common::M3D::Identity() * 1e-3;
    this->cov.block<3, 3>(BIAS_A, BIAS_A) = common::M3D::Identity() * 1e-3;
    this->cov.block<3, 3>(GRAVITY, GRAVITY) =
        common::M3D::Identity() * 1e-3;
  };

  State &operator=(const State &b) = default;

  State operator+(const Eigen::Matrix<double, STATE_DOF, 1> &state_add) {
    State a;
    a.rot_end = this->rot_end * ExpQuat(state_add.block<3, 1>(ROT, 0));
    a.pos_end = this->pos_end + state_add.block<3, 1>(POS, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(VEL, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(BIAS_G, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(BIAS_A, 0);
    a.gravity = this->gravity + state_add.block<3, 1>(GRAVITY, 0);
    a.cov = this->cov;
    a.rot_end.normalize();
    return a;
  };

  State &operator+=(
      const Eigen::Matrix<double, STATE_DOF, 1> &state_add) {
    this->rot_end = this->rot_end * ExpQuat(state_add.block<3, 1>(ROT, 0));
    this->pos_end += state_add.block<3, 1>(POS, 0);
    this->vel_end += state_add.block<3, 1>(VEL, 0);
    this->bias_g += state_add.block<3, 1>(BIAS_G, 0);
    this->bias_a += state_add.block<3, 1>(BIAS_A, 0);
    this->gravity += state_add.block<3, 1>(GRAVITY, 0);
    this->rot_end.normalize();
    return *this;
  };

  Eigen::Matrix<double, STATE_DOF, 1> operator-(const State &b) {
    Eigen::Matrix<double, STATE_DOF, 1> a;
    a.block<3, 1>(ROT, 0) = LogQuat(b.rot_end.conjugate() * this->rot_end);
    a.block<3, 1>(POS, 0) = this->pos_end - b.pos_end;
    a.block<3, 1>(VEL, 0) = this->vel_end - b.vel_end;
    a.block<3, 1>(BIAS_G, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(BIAS_A, 0) = this->bias_a - b.bias_a;
    a.block<3, 1>(GRAVITY, 0) = this->gravity - b.gravity;
    return a;
  };


  double timestamp;
  Eigen::Quaterniond rot_end;
  common::V3D pos_end;
  common::V3D vel_end;
  common::V3D bias_g;
  common::V3D bias_a;
  common::V3D gravity;
  Eigen::Matrix<double, STATE_DOF, STATE_DOF> cov;  // states covariance
};
}  // namespace open_livo
#pragma once
#include <Eigen/Eigen>
#include <memory>
#include "common_lib.h"
#include "so3_math.h"
#include "s2.h"
namespace faster_lio {
struct StatePoint {
  using Ptr = std::shared_ptr<StatePoint>;
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
  StatePoint() {
    timestamp = -1.0;
    this->rot = Eigen::Quaterniond::Identity();
    this->pos = Vec3::Zero();
    this->vel_end = Vec3::Zero();
    this->bias_g = Vec3::Zero();
    this->bias_a = Vec3::Zero();
    this->gravity = Vec3::UnitZ() * GRAVITY_NORM;

    // init cov
    this->cov = Eigen::Matrix<double, STATE_DOF, STATE_DOF>::Identity() * 0.01;
    this->cov.block<3, 3>(ROT, ROT) = Mat3::Identity() * 1e-3;
    this->cov.block<3, 3>(POS, POS) = Mat3::Identity() * 1e-12;
    this->cov.block<3, 3>(VEL, VEL) = Mat3::Identity() * 1e-12;
    this->cov.block<3, 3>(BIAS_G, BIAS_G) = Mat3::Identity() * 1e-3;
    this->cov.block<3, 3>(BIAS_A, BIAS_A) = Mat3::Identity() * 1e-3;
    this->cov.block<3, 3>(GRAVITY, GRAVITY) =
        Mat3::Identity() * 1e-3;
  };

  StatePoint &operator=(const StatePoint &b) = default;

  StatePoint operator+(const Eigen::Matrix<double, STATE_DOF, 1> &state_add) {
    StatePoint a;
    a.rot = this->rot * ExpQuat(state_add.block<3, 1>(ROT, 0));
    a.pos = this->pos + state_add.block<3, 1>(POS, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(VEL, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(BIAS_G, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(BIAS_A, 0);
    a.gravity = this->gravity + state_add.block<3, 1>(GRAVITY, 0);
    a.cov = this->cov;
    a.rot.normalize();
    a.gravity = a.gravity.normalized() * GRAVITY_NORM;
    return a;
  };

  StatePoint &operator+=(
      const Eigen::Matrix<double, STATE_DOF, 1> &state_add) {
    this->rot = this->rot * ExpQuat(state_add.block<3, 1>(ROT, 0));
    this->pos += state_add.block<3, 1>(POS, 0);
    this->vel_end += state_add.block<3, 1>(VEL, 0);
    this->bias_g += state_add.block<3, 1>(BIAS_G, 0);
    this->bias_a += state_add.block<3, 1>(BIAS_A, 0);
    this->gravity += state_add.block<3, 1>(GRAVITY, 0);
    this->rot.normalize();
    this->gravity = this->gravity.normalized() * GRAVITY_NORM;
    return *this;
  };

  Eigen::Matrix<double, STATE_DOF, 1> operator-(const StatePoint &b) {
    Eigen::Matrix<double, STATE_DOF, 1> a;
    a.block<3, 1>(ROT, 0) = LogQuat(b.rot.conjugate() * this->rot);
    a.block<3, 1>(POS, 0) = this->pos - b.pos;
    a.block<3, 1>(VEL, 0) = this->vel_end - b.vel_end;
    a.block<3, 1>(BIAS_G, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(BIAS_A, 0) = this->bias_a - b.bias_a;
    a.block<3, 1>(GRAVITY, 0) = this->gravity - b.gravity;
    return a;
  };


  double timestamp;
  Eigen::Quaterniond rot;
  Vec3 pos;
  Vec3 vel_end;
  Vec3 bias_g;
  Vec3 bias_a;
  Vec3 gravity;
  Eigen::Matrix<double, STATE_DOF, STATE_DOF> cov;  // states covariance
};
}  // namespace faster_lio

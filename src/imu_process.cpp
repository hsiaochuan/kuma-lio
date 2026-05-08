#include "imu_processing.hpp"
#include "utils.h"
using namespace faster_lio;
ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true) {
    init_iter_num_ = 1;
    Q_ = process_noise_cov();
    cov_acc_ = Vec3(0.1, 0.1, 0.1);
    cov_gyr_ = Vec3(0.1, 0.1, 0.1);
    cov_bias_gyr_ = Vec3(0.0001, 0.0001, 0.0001);
    cov_bias_acc_ = Vec3(0.0001, 0.0001, 0.0001);
    mean_acc_ = Vec3(0, 0, -1.0);
    mean_gyr_ = Vec3(0, 0, 0);
    angvel_last_ = Vec3::Zero();
    Lidar_T_wrt_IMU_ = Vec3::Zero();
    Lidar_R_wrt_IMU_ = Mat3::Identity();
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
    mean_acc_ = Vec3(0, 0, -1.0);
    mean_gyr_ = Vec3(0, 0, 0);
    angvel_last_ = Vec3::Zero();
    imu_need_init_ = true;
    init_iter_num_ = 1;
    v_imu_.clear();
    imu_poses_.clear();
    cur_pcl_un_.reset(new PointCloud());
}

void ImuProcess::SetExtrinsic(const Vec3 &transl, const Mat3 &rot) {
    Lidar_T_wrt_IMU_ = transl;
    Lidar_R_wrt_IMU_ = rot;
}

void ImuProcess::SetGyrCov(const Vec3 &scaler) { cov_gyr_scale_ = scaler; }

void ImuProcess::SetAccCov(const Vec3 &scaler) { cov_acc_scale_ = scaler; }

void ImuProcess::SetGyrBiasCov(const Vec3 &b_g) { cov_bias_gyr_ = b_g; }

void ImuProcess::SetAccBiasCov(const Vec3 &b_a) { cov_bias_acc_ = b_a; }

void ImuProcess::IMUInit(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         int &N) {
    /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity_ **/

    Vec3 cur_acc, cur_gyr;

    if (b_first_frame_) {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu_.front().linear_acceleration;
        const auto &gyr_acc = meas.imu_.front().angular_velocity;
        mean_acc_ = imu_acc;
        mean_gyr_ = gyr_acc;
    }

    for (const auto &imu : meas.imu_) {
        const auto &imu_acc = imu.linear_acceleration;
        const auto &gyr_acc = imu.angular_velocity;
        cur_acc = imu_acc;
        cur_gyr = gyr_acc;

        mean_acc_ += (cur_acc - mean_acc_) / N;
        mean_gyr_ += (cur_gyr - mean_gyr_) / N;

        cov_acc_ =
            cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
        cov_gyr_ =
            cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

        N++;
    }
    state_ikfom init_state = kf_state.get_x();
    init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);

    init_state.bg = mean_gyr_;
    init_state.t_il = Lidar_T_wrt_IMU_;
    init_state.R_il = Lidar_R_wrt_IMU_;
    kf_state.change_x(init_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu_.back();
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                              PointCloud::Ptr distort_points, PointCloud& undistort_points) {
    /*** add the imu_ of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu_;
    v_imu.push_front(last_imu_);
    const double &imu_end_time = v_imu.back().timestamp;
    const double &pcl_end_time = meas.lidar_end_time_;

    /*** sort point clouds by offset time ***/
    undistort_points = *distort_points;
    std::sort(undistort_points.points.begin(), undistort_points.points.end(), [](const faster_lio::Point &x, const faster_lio::Point &y) { return (x.timestamp < y.timestamp); });

    /*** Initialize IMU pose ***/
    state_ikfom imu_state = kf_state.get_x();
    imu_poses_.clear();
    imu_poses_.emplace_back(last_lidar_end_time_, imu_state.pos, imu_state.vel, acc_s_last_, imu_state.rot.toRotationMatrix(), angvel_last_);

    /*** forward propagation at each imu_ point ***/
    Vec3 omega_i, acc_avr, acc_i, vel_i, pos_i;
    Mat3 R_i;

    double dt = 0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail.timestamp < last_lidar_end_time_) {
            continue;
        }

        omega_i = 0.5 * (head.angular_velocity + tail.angular_velocity);
        acc_avr =  0.5 * (head.linear_acceleration + tail.linear_acceleration);

        acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();  // - state_inout.ba;

        if (head.timestamp < last_lidar_end_time_) {
            dt = tail.timestamp - last_lidar_end_time_;
        } else {
            dt = tail.timestamp - head.timestamp;
        }

        in.acc = acc_avr;
        in.gyro = omega_i;
        Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
        Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
        Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        kf_state.predict(dt, Q_, in);

        /* save the poses at each IMU measurements */
        imu_state = kf_state.get_x();
        angvel_last_ = omega_i - imu_state.bg;
        acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba);
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav[i];
        }

        imu_poses_.emplace_back(tail.timestamp, imu_state.pos, imu_state.vel, acc_s_last_, imu_state.rot.toRotationMatrix(), angvel_last_);
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q_, in);

    imu_state = kf_state.get_x();
    last_imu_ = meas.imu_.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    if (undistort_points.points.empty()) {
        return;
    }
    auto it_k = undistort_points.points.end() - 1;
    Pose3 extrin_il(imu_state.R_il, imu_state.t_il);
    Pose3 extrin_li = extrin_il.GetInverse();
    Pose3 body_world_end = Pose3(imu_state.rot, imu_state.pos).GetInverse();
    for (auto imu_i = imu_poses_.end() - 1; imu_i != imu_poses_.begin(); imu_i--) {
        auto head = imu_i - 1;
        auto tail = imu_i;
        R_i = head->rot;
        vel_i = head->vel;
        pos_i = head->pos;
        acc_i = tail->acc;
        omega_i = tail->gyr;

        for (; it_k->timestamp > head->offset_time; it_k--) {
            dt = it_k->timestamp - head->offset_time;


            Mat3 R_k(R_i * ExpMat(omega_i * dt));
            Vec3 pk(it_k->x, it_k->y, it_k->z);
            Vec3 pos_k = pos_i + vel_i * dt + 0.5 * acc_i * dt * dt;
            Vec3 p_compensate = body_world_end * (R_k * pk + pos_k);

            // save Undistorted points and their rotation
            it_k->x = p_compensate(0);
            it_k->y = p_compensate(1);
            it_k->z = p_compensate(2);

            if (it_k == undistort_points.points.begin()) {
                break;
            }
        }
    }
}

void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloud::Ptr distort_points, PointCloud& undistort_points) {
    if (meas.imu_.empty()) {
        return;
    }

    CHECK(meas.lidar_ != nullptr);

    if (imu_need_init_) {
        /// The very first lidar frame
        IMUInit(meas, kf_state, init_iter_num_);

        imu_need_init_ = true;

        last_imu_ = meas.imu_.back();

        state_ikfom imu_state = kf_state.get_x();
        if (init_iter_num_ > MAX_INI_COUNT) {
            cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
            imu_need_init_ = false;

            cov_acc_ = cov_acc_scale_;
            cov_gyr_ = cov_gyr_scale_;
            LOG(INFO) << "IMU Initial Done";
        }

        return;
    }

    Timer::Evaluate([&, this]() { UndistortPcl(meas, kf_state, distort_points, undistort_points); }, "Undistort Pcl");
}
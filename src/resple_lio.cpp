//
// Created by hsiaochuan on 2026/08/23.
//
#include "resple_lio.h"

#include <ros/param.h>

#include "rtklib_func.h"
namespace faster_lio {

void RESPLE_LIO::Init() {
    const double dt_s = param->dt_ns * 1e-9;
    // set P0
    cov_rcp.setZero();
    cov_rcp.topLeftCorner<CP_SIZE, CP_SIZE>() =
        param->cov_P0 * dt_s * dt_s * Eigen::Matrix<double, CP_SIZE, CP_SIZE>::Identity();
    cov_rcp.block<3, 3>(BA_OFFSET, BA_OFFSET) = param->cov_ba.asDiagonal();
    cov_rcp.block<3, 3>(BG_OFFSET, BG_OFFSET) = param->cov_bg.asDiagonal();

    // set Q
    double cov_sys_pos = param->std_sys_pos * param->std_sys_pos * dt_s * dt_s;
    double cov_sys_ort = param->std_sys_ort * param->std_sys_ort * dt_s * dt_s;
    Eigen::Matrix<double, XSIZE, XSIZE> Q = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
    Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
    Q_block_old.topLeftCorner<3, 3>() = param->cov_RCP_pos_old * cov_sys_pos * Eigen::Matrix3d::Identity();
    Q_block_old.bottomRightCorner<3, 3>() = param->cov_RCP_ort_old * cov_sys_ort * Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
    Q_block_new.topLeftCorner<3, 3>() = param->cov_RCP_pos_new * cov_sys_pos * Eigen::Matrix3d::Identity();
    Q_block_new.bottomRightCorner<3, 3>() = param->cov_RCP_ort_new * cov_sys_ort * Eigen::Matrix3d::Identity();
    Q.topLeftCorner<6, 6>() = Q_block_old;
    Q.block<6, 6>(6, 6) = Q_block_old;
    Q.block<6, 6>(12, 12) = Q_block_old;
    Q.block<6, 6>(18, 18) = Q_block_new;
    cov_sys = Q;

    // A
    a_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
    Eigen::Matrix<double, 6, 6> matblock = Eigen::Matrix<double, 6, 6>::Zero();
    matblock.topLeftCorner<3, 3>().setIdentity();
    matblock.bottomRightCorner<3, 3>().setIdentity();
    a_mat.block(0, 6, 6, 6) = matblock;
    a_mat.block(6, 12, 6, 6) = matblock;
    a_mat.block(12, 18, 6, 6) = matblock;
    a_mat.block(18, 0, 3, 3) = -Eigen::Matrix3d::Identity();
    a_mat.block(18, 12, 3, 3) = 2 * Eigen::Matrix3d::Identity();
    a_mat.block(21, 9, 3, 3) = Eigen::Matrix3d::Identity();
    a_mat.block(BA_OFFSET, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
    a_mat.block(BG_OFFSET, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();

    // GNSS states are frozen at a placeholder covariance until InitGnssState()
    cov_rcp.block<GNSS_SIZE, GNSS_SIZE>(CLK_OFFSET, CLK_OFFSET).setIdentity();
    a_mat.block<GNSS_SIZE, GNSS_SIZE>(CLK_OFFSET, CLK_OFFSET).setIdentity();

    param->lba_.num_threads = param->num_threads;
    static_detector = StaticDetector(param->zupt_);
    lba = std::make_unique<LocalBA>(param->lba_);
    LOG(INFO) << "local BA map shared with the odometry: voxel " << param->lba_.voxel_size << " window "
              << param->lba_.win_size;
}
// receiver clock index; QZSS keeps GPS time
static int SysIdx(int sys) {
    if (sys == SYS_GPS || sys == SYS_QZS) return 0;
    if (sys == SYS_GLO) return 1;
    if (sys == SYS_GAL) return 2;
    if (sys == SYS_CMP) return 3;
    return -1;
}

static bool Alignment(Mat3& R_ecef_enu, Vec3& t_ecef_enu, const std::vector<Vec3>& locals,
                      const std::vector<Vec3>& ecefs, double min_travel) {
    CHECK(locals.size() == ecefs.size());
    int N = locals.size();
    if (N < 3) return false;
    // the yaw needs real horizontal spread; summed path length is fooled by jitter on a vertical climb
    double spread = 0.0;
    for (int i = 1; i < N; ++i) {
        spread = std::max(spread, (locals[i] - locals[0]).head<2>().norm());
    }
    if (spread < min_travel) return false;
    Vec3 mean_l(Vec3::Zero());
    Vec3 mean_e(Vec3::Zero());
    for (int i = 0; i < N; ++i) {
        mean_l += locals[i];
        mean_e += ecefs[i];
    }
    mean_l /= N;
    mean_e /= N;
    double anc_llh[3];
    ecef2pos(mean_e.data(), anc_llh);
    Eigen::Matrix3d R_enu_ecef;
    xyz2enu(anc_llh, R_enu_ecef.data());
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const Vec3 s = locals[i] - mean_l;
        const Vec3 d = R_enu_ecef * (ecefs[i] - mean_e);
        num += s.x() * d.y() - s.y() * d.x();
        den += s.x() * d.x() + s.y() * d.y();
    }
    const double yaw = std::atan2(num, den);
    const Mat3 Rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    R_ecef_enu = R_enu_ecef.transpose() * Rz;
    t_ecef_enu = mean_e - R_ecef_enu * mean_l;
    return true;
}
void RESPLE_LIO::ProcessMeasurement(MeasureGroup& measures) {
    Timer stage_timer;
    for (const Imu& imu : measures.imu_) {
        lba->AddImu(imu.time_ns, imu.angular_velocity, imu.linear_acceleration);
        static_detector.Add(imu);
    }
    if (measures.lidar_->empty()) {
        LOG(INFO) << "No LiDAR data, skip process measurement";
        return;
    }
    // transform point
    PointCloud::Ptr points_body(new PointCloud);
    pcl::transformPointCloud(*measures.lidar_, *points_body, param->extrin_il_.Mat4d(), true);

    // downsample point
    if (param->ds_scan_voxel > 0) {
        pcl::UniformSampling<Point> scan_sampler;
        scan_sampler.setRadiusSearch(param->ds_scan_voxel);
        scan_sampler.setInputCloud(points_body);
        scan_sampler.filter(*points_body);
        std::sort(points_body->points.begin(), points_body->points.end(),
                  [](const Point& p1, const Point& p2) { return p1.GetTimeNs() < p2.GetTimeNs(); });
    }

    sol_t sol{};
    bool gnss_epoch_ok = false;
    tdcp_meas.clear();
    psr_meas.clear();
    if (measures.obs_id > 0) {
        gnss_epoch_ok = ProcessGnssEpoch(measures.obs_id, measures.obs_time_ns, sol);
        if (!tdcp_meas.empty())
            PrepareTdCpEpoch();
    }

    // add to imu
    imu_meas.clear();
    imu_meas.resize(measures.imu_.size());
    for (size_t i = 0; i < measures.imu_.size(); ++i) {
        imu_meas[i] =
            ImuData(measures.imu_[i].time_ns, measures.imu_[i].angular_velocity, measures.imu_[i].linear_acceleration);
    }

    // add to lidar points
    pt_meas.clear();
    pt_meas.resize(points_body->size());
    for (int i = 0; i < points_body->size(); ++i) {
        pt_meas[i].point_body = points_body->at(i).getVector3fMap().cast<double>();
        pt_meas[i].point = points_body->at(i);
        pt_meas[i].time_ns = points_body->at(i).GetTimeNs();
        pt_meas[i].pt_wt = param->w_pt;
        pt_meas[i].body_cov = BodyPointCov(pt_meas[i].point_body, param->lba_.range_err, param->lba_.angle_err);
    }

    // lidar-inertial update
    IterativeUpdate();

    if (param->zupt_.enable) {
        const bool was_static = is_static;
        is_static = static_detector.Quiet() && TrajectoryStill();
        if (is_static) ZeroVelocityUpdate();
        if (is_static != was_static)
            LOG(INFO) << "ZUPT: " << (is_static ? "at rest" : "moving") << " from " << NS_TO_S(spl.maxTimeNs()) << " s";
    }
    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; ++i) {
        PointData& pt_data = pt_meas[i];
        Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
        Eigen::Quaterniond interp_q;
        spl.itpQuaternion(pt_data.time_ns, &interp_q);
        pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
    }
    const double t_update = stage_timer.ElapsedMiniSeconds(true);
    UpdateLocalBA(measures);
    LOG(INFO) << "Stage time [ms]: update " << t_update << " lba " << stage_timer.ElapsedMiniSeconds()
              << " raw points " << measures.lidar_->size();

    if (gnss_epoch_ok && !if_GNSS_aligned) {
        // align the antenna, not the IMU, with the SPP fix
        Eigen::Isometry3d pose = spl.Interpolate(measures.obs_time_ns);
        Vec3 local = pose * param->level_arm;
        Vec3 ecef(sol.rr[0], sol.rr[1], sol.rr[2]);
        if (align_locals.empty() || (align_locals.back() - local).norm() > 0.3) {
            align_locals.push_back(local);
            align_ecefs.push_back(ecef);
            if (Alignment(R_ecef_enu_init, t_ecef_enu, align_locals, align_ecefs, param->ali_min_travel)) {
                InitGnssState(sol);
            }
        }
    }
}

bool RESPLE_LIO::ProcessGnssEpoch(int ei, int64_t local_time, sol_t& sol) {
    int n = 0;
    while (ei + n < obss.n && std::abs(timediff(obss.data[ei].time, obss.data[ei + n].time)) < DTTOL) ++n;

    char err_msg[128];
    if (!pntpos(obss.data + ei, n, &nav, &param->prcopt, &sol, nullptr, nullptr, err_msg)) {
        LOG(INFO) << "SPP failed: " << err_msg;
        return false;
    }

    // satellite pos
    std::vector<double> rs(6 * n, 0.0), dts(2 * n, 0.0), svar(n, 0.0);
    std::vector<int> svh(n, 0);
    satposs(sol.time, obss.data + ei, n, &nav, param->prcopt.sateph, rs.data(), dts.data(), svar.data(), svh.data());
    double rcv_llh[3];
    ecef2pos(sol.rr, rcv_llh);



    for (int si = 0; si < n; ++si) {
        int oi = ei + si;
        const obsd_t& obs = obss.data[oi];
        // filter
        if (satexclude(obs.sat, svar[si], svh[si], &param->prcopt)) continue;
        const int sys = satsys(obs.sat, nullptr);
        const int sys_idx = SysIdx(sys);
        if (sys_idx < 0) continue;
        double azel[2] = {0.0, PI / 2.0};
        double e[3];
        if (geodist(&rs[si * 6], sol.rr, e) < 0.0) continue;
        if (satazel(rcv_llh, e, azel) < param->prcopt.elmin) continue;
        if (testsnr(0, 0, azel[1], obs.SNR[0], &param->prcopt.snrmask)) continue;

        const double freq = sat2freq(obs.sat, obs.code[0], &nav);
        if (freq <= 0.0) continue;
        const double lambda = CLIGHT / freq;
        const double phase_var_cur = rtklib::varerr_phase(&param->prcopt, &obs, azel[1], sys);

        // manage the sat track; phase and Doppler are signed, so only zero means missing
        bool psr_valid = obs.P[0] > 0.0;
        bool cp_valid = obs.L[0] != 0.0;
        bool dopp_valid = obs.D[0] != 0.0;
        bool obs_valid = psr_valid && cp_valid && dopp_valid;

        bool arc_continu = true;
        do {
            if (!obs_valid) {
                arc_continu = false;
                break;
            }
            if (sat_tracks_[obs.sat].obs_ids.empty()) {
                arc_continu = false;
                break;
            }
            if (sat_tracks_[obs.sat].obs_ids.size() >= 20) {
                arc_continu = false;
                break;
            }
            int last_oi = sat_tracks_[obs.sat].obs_ids.back();
            obsd_t& last_obs = obss.data[last_oi];
            if ((obs.LLI[0] & LLI_SLIP) || ((obs.LLI[0] & LLI_HALFC) != (last_obs.LLI[0] & LLI_HALFC))) {
                arc_continu = false;
                break;
            }
            // code change
            if (last_obs.code[0] != obs.code[0]) {
                arc_continu = false;
                break;
            }

            // time diff
            double dt = timediff(obs.time, last_obs.time);
            if (dt <= 0.0 || dt > param->max_gap) {
                arc_continu = false;
                break;
            }

            // geometry-free
            if (param->prcopt.thresslip > 0.0 && last_obs.L[1] > 0.0 && obs.L[1] > 0.0) {
                const double freq2 = sat2freq(obs.sat, obs.code[1], &nav);
                if (freq2 > 0.0) {
                    const double lambda2 = CLIGHT / freq2;
                    double last_gf = last_obs.L[0] * lambda - last_obs.L[1] * lambda2;
                    double cur_gf = obs.L[0] * lambda - obs.L[1] * lambda2;
                    if (std::abs(last_gf - cur_gf) > param->prcopt.thresslip) {
                        arc_continu = false;
                        break;
                    }
                }
            }

            // doppler-phase
            const double dpt = (-0.5) * lambda * (obs.D[0] + last_obs.D[0]) * dt;
            const double dcp = obs.L[0] * lambda - last_obs.L[0] * lambda;
            if (std::abs(dcp - dpt) > param->prcopt.thresdop) {
                arc_continu = false;
                break;
            }
        } while (false);

        Eigen::Vector3d sv(rs[si * 6], rs[si * 6 + 1], rs[si * 6 + 2]);
        if (arc_continu) {
            sat_tracks_[obs.sat].obs_ids.push_back(oi);
            sat_tracks_[obs.sat].sv_poss.push_back(sv);
            sat_tracks_[obs.sat].local_times.push_back(local_time);
            sat_tracks_[obs.sat].sv_dts.push_back(dts[si * 2 + 0]);
            sat_tracks_[obs.sat].phase_vars.push_back(phase_var_cur);
        } else {
            // clear the track
            sat_tracks_[obs.sat].obs_ids.clear();
            sat_tracks_[obs.sat].sv_poss.clear();
            sat_tracks_[obs.sat].local_times.clear();
            sat_tracks_[obs.sat].sv_dts.clear();
            sat_tracks_[obs.sat].phase_vars.clear();
            // insert the one observation
            sat_tracks_[obs.sat].obs_ids.push_back(oi);
            sat_tracks_[obs.sat].sv_poss.push_back(sv);
            sat_tracks_[obs.sat].local_times.push_back(local_time);
            sat_tracks_[obs.sat].sv_dts.push_back(dts[si * 2 + 0]);
            sat_tracks_[obs.sat].phase_vars.push_back(phase_var_cur);
        }

        SatTrack& track = sat_tracks_[obs.sat];
        const int N = static_cast<int>(track.obs_ids.size());
        if (if_GNSS_aligned && param->tdcp_enable && arc_continu && N >= 2) {
            const int i = N - 2;
            const obsd_t& anchor_obs = obss.data[track.obs_ids[i]];
            TimeDiffCarrierPhaseData tdcp_data;
            tdcp_data.time_ns = local_time;
            tdcp_data.anchor_time_ns = track.local_times[i];
            tdcp_data.sv_pos = sv;
            tdcp_data.sv_pos_anchor = track.sv_poss[i];
            tdcp_data.sv_dt = dts[si * 2 + 0];
            tdcp_data.sv_dt_anchor = track.sv_dts[i];
            tdcp_data.cp_diff = lambda * (obs.L[0] - anchor_obs.L[0]);
            tdcp_data.sys_idx = sys_idx;
            tdcp_data.var = track.phase_vars[i] + phase_var_cur;
            tdcp_data.if_valid = true;
            tdcp_meas.push_back(tdcp_data);
        }
    }  // for satellite

    return true;
}

void RESPLE_LIO::PrepareTdCpEpoch() {
    // Residual without the receiver clock step. The step is common to every system (the
    // inter-system biases cancel between epochs), so the median across satellites estimates it.
    const int num = static_cast<int>(tdcp_meas.size());
    std::vector<double> res(num), var_innov(num);
    Eigen::Matrix<double, 1, XSIZE> H_cur, H_anchor;
    for (int i = 0; i < num; ++i) {
        const TimeDiffCarrierPhaseData& data = tdcp_meas[i];
        const double range = AntennaRange(data.time_ns, data.sv_pos, &H_cur);
        const double range_anchor = AntennaRange(data.anchor_time_ns, data.sv_pos_anchor, &H_anchor);
        res[i] = data.cp_diff + CLIGHT * (data.sv_dt - data.sv_dt_anchor) - (range - range_anchor);
        // innovation variance without the clock column: the median below takes the clock step out
        const Eigen::Matrix<double, 1, XSIZE> h = H_cur - H_anchor;
        var_innov[i] = (h * cov_rcp * h.transpose())(0) + data.var;
    }
    std::vector<double> sorted = res;
    std::nth_element(sorted.begin(), sorted.begin() + num / 2, sorted.end());
    const double clk_step = sorted[num / 2];

    // hold the clocks of epoch k as the anchor and predict epoch k+1 from the median step
    clk_anchor = clk;
    clk.array() += clk_step;

    // LIVO-prior chi-square test (FAST-LIVGO eq. 14) on each clock-free innovation
    int num_valid = 0;
    double sq_sum = 0.0;
    for (int i = 0; i < num; ++i) {
        const double r = res[i] - clk_step;
        tdcp_meas[i].if_valid = r * r / var_innov[i] < param->tdcp_chi2 && std::abs(r) < param->tdcp_outlier;
        if (tdcp_meas[i].if_valid) {
            ++num_valid;
            sq_sum += r * r;
        }
    }
    if (num_valid < param->tdcp_min_sat) {
        for (auto& d : tdcp_meas) d.if_valid = false;
    }
    LOG(INFO) << "TDCP: " << num_valid << "/" << num << " valid, rms " << std::sqrt(sq_sum / std::max(num_valid, 1))
              << " m, clock step " << clk_step << " m, theta_ecef_enu " << theta_ecef_enu;
}

void RESPLE_LIO::InitGnssState(const sol_t& sol) {
    // pntpos reports the GPS clock and the GLO/GAL/BDS offsets from it
    clk(0) = CLIGHT * sol.dtr[0];
    clk(1) = CLIGHT * (sol.dtr[0] + sol.dtr[1]);
    clk(2) = CLIGHT * (sol.dtr[0] + sol.dtr[2]);
    clk(3) = CLIGHT * (sol.dtr[0] + sol.dtr[3]);
    clk_anchor = clk;
    theta_ecef_enu = 0.0;

    cov_rcp.middleRows<GNSS_SIZE>(CLK_OFFSET).setZero();
    cov_rcp.middleCols<GNSS_SIZE>(CLK_OFFSET).setZero();
    cov_rcp.block<4, 4>(CLK_OFFSET, CLK_OFFSET) =
        param->std_clk_init * param->std_clk_init * Eigen::Matrix4d::Identity();
    cov_rcp(ROT_OFFSET, ROT_OFFSET) = param->std_ext_yaw_init * param->std_ext_yaw_init;
    cov_rcp.block<3, 3>(TRANS_OFFSET, TRANS_OFFSET) =
        param->std_ext_pos_init * param->std_ext_pos_init * Eigen::Matrix3d::Identity();

    // one receiver oscillator drives all four clocks, plus a slow per-system bias
    const double dt_s = param->dt_ns * 1e-9;
    cov_sys.block<4, 4>(CLK_OFFSET, CLK_OFFSET) =
        (param->std_clk * param->std_clk * Eigen::Matrix4d::Ones() +
         param->std_isb * param->std_isb * Eigen::Matrix4d::Identity()) * dt_s;

    if_GNSS_aligned = true;
    LOG(INFO) << "GNSS aligned with " << align_locals.size() << " SPP fixes, t_ecef_enu " << t_ecef_enu.transpose()
              << ", clk " << clk.transpose();
}

Eigen::Matrix<double, XSIZE, 1> RESPLE_LIO::getState() {
    Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
    Eigen::Matrix<double, XSIZE, 1> state;
    state << cps_win, ba, bg, clk, theta_ecef_enu, t_ecef_enu;
    return state;
}

int RESPLE_LIO::Match() {
    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) schedule(dynamic) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; i++) {
        PointData& pt_data = pt_meas[i];
        pt_data.if_valid = false;
        // interp to point world
        Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
        Eigen::Quaterniond interp_q;
        spl.itpQuaternion(pt_data.time_ns, &interp_q);
        pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
        const Mat3 R = interp_q.toRotationMatrix();
        const Mat3 var_w = R * pt_data.body_cov * R.transpose() + param->lba_.pose_var * Mat3::Identity();
        PlaneHit hit;
        if (lba->Query(LbaPoint{pt_data.point_world, var_w}, &hit) &&
            pt_data.point_body.norm() > 81.0 * hit.signed_dist * hit.signed_dist) {
            pt_data.if_valid = true;
            pt_data.ppl_normal = hit.normal;
            pt_data.ppl_d = hit.d;
            pt_data.pt_wt = param->lba_.cov_en
                                ? param->lba_.r_base + hit.plane_sigma + hit.normal.dot(var_w * hit.normal)
                                : param->w_pt;
        }
    }  // for point

    int eff_count = 0;
    for (int i = 0; i < pt_meas.size(); i++) {
        if (pt_meas[i].if_valid) {
            eff_count++;
        }
    }
    return eff_count;
}

void RESPLE_LIO::IterativeUpdate() {
    const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
    Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
    bool converged = true;
    int num_eff = 0;
    int t = 0;
    const int max_iter = 5;
    const int n_iter = 1;
    const double eps = 0.1;
    Timer timer;
    for (int i = 0; i < max_iter; i++) {
        Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
        if (converged) {
            num_eff = Match();
        }
        if (num_eff > 0) {
            UpdateLiDARInertial(num_eff, rcp_prop, cov_prop);
        } else {
            LOG(INFO) << "No effective points";
            break;
        }
        converged = true;
        Eigen::Matrix<double, XSIZE, 1> state_af = getState();
        if ((state_af - rcpi).norm() > eps) {
            converged = false;
        } else {
            t++;
        }
        if (!t && i == max_iter - 2) {
            converged = true;
        }
        if (t > n_iter || (i == max_iter - 1)) {
            cov_rcp = (Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
            cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());
            break;
        }
    }  // iter
    double update_time = timer.ElapsedMiniSeconds();
    LOG(INFO) << "Points: " << pt_meas.size()
              << " Imu: " << imu_meas.size()
              << " Effect Points: " << num_eff
              << " GNSS " << tdcp_meas.size()
              << " Time: " << update_time;
}
double RESPLE_LIO::AntennaRange(int64_t t_ns, const Vec3& sv_pos, Eigen::Matrix<double, 1, XSIZE>* H) const {
    Eigen::Quaterniond q_itp;
    Jacobian43 J_ortdel;
    Jacobian J_pos;
    spl.itpQuaternion(t_ns, &q_itp, nullptr, H ? &J_ortdel : nullptr);
    const Vec3 p_itp = spl.itpPosition(t_ns, H ? &J_pos : nullptr);
    const Vec3 p_ant = q_itp * param->level_arm + p_itp;
    const Mat3 R = RotEcefEnu();
    const Vec3 p_ecef = R * p_ant + t_ecef_enu;
    double e[3];
    const double range = geodist(sv_pos.data(), p_ecef.data(), e);
    if (!H) return range;

    // d(range)/d(receiver ECEF), Sagnac term included
    const Eigen::RowVector3d d_range =
        -Eigen::Map<const Vec3>(e).transpose() + Eigen::RowVector3d(-OMGE / CLIGHT * sv_pos.y(), OMGE / CLIGHT * sv_pos.x(), 0.0);
    const Eigen::RowVector3d d_range_local = d_range * R;
    Eigen::Matrix<double, 3, 4> drot;
    dRot(param->level_arm, q_itp, drot);
    const Eigen::Matrix<double, 1, 4> tmp = d_range_local * drot;

    H->setZero();

    for (int i = 0; i < 4; i++) {
        H->block<1, 3>(0, i * 6) = d_range_local * J_pos.d_val_d_knot[i];
        H->block<1, 3>(0, i * 6 + 3) = tmp * J_ortdel.d_val_d_knot[i];
    }
    // R_init * Rz(yaw) * v, d/d(yaw) = R * (ez x v) since Rz(yaw) commutes with its own generator
    H->block<1, 1>(0, ROT_OFFSET) = d_range_local * Vec3::UnitZ().cross(p_ant);
    H->block<1, 3>(0, TRANS_OFFSET) = d_range;
    return range;
}

void RESPLE_LIO::prepTdCp(TimeDiffCarrierPhaseData& tdcp_data) const {
    if (!tdcp_data.if_valid) return;
    // lambda*dPhi + c*d(dts) = (range_{k+1} - range_k) + (clk_{k+1} - clk_k), clk_k held at the anchor
    Eigen::Matrix<double, 1, XSIZE> H_cur, H_anchor;
    const double range = AntennaRange(tdcp_data.time_ns, tdcp_data.sv_pos, &H_cur);
    const double range_anchor = AntennaRange(tdcp_data.anchor_time_ns, tdcp_data.sv_pos_anchor, &H_anchor);
    H_anchor.leftCols<24>().setZero();
    const int s = tdcp_data.sys_idx;
    const double pred = range - range_anchor + clk(s) - clk_anchor(s);
    tdcp_data.z = tdcp_data.cp_diff + CLIGHT * (tdcp_data.sv_dt - tdcp_data.sv_dt_anchor) - pred;
    tdcp_data.H = H_cur - H_anchor;
    tdcp_data.H(0, CLK_OFFSET + s) = 1.0;
}
void RESPLE_LIO::prepPsr(PsrData& psr_data) {}

void RESPLE_LIO::prepIMU(ImuData& imu_data) {
    Eigen::Quaterniond q_itp;
    Eigen::Vector3d rot_vel;
    Jacobian43 J_ortdel;
    Jacobian J_line_acc;
    Jacobian33 J_gyro;
    spl.itpQuaternion(imu_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro);
    Eigen::Vector3d a_w_no_g = spl.itpPosition<2>(imu_data.time_ns, &J_line_acc);
    Eigen::Vector3d a_w = a_w_no_g + gravity;
    Eigen::Matrix3d RT = q_itp.inverse().toRotationMatrix();
    Eigen::Matrix<double, 3, 4> drot;
    faster_lio::dRotInv(a_w, q_itp, drot);
    Eigen::Matrix<double, 6, XSIZE> Hi = Eigen::Matrix<double, 6, XSIZE>::Zero();
    // map knot Jacobians into the recursive-window columns; measurements in an
    // older segment touch a knot outside the window, whose column is dropped
    const int recur_st_id = spl.num_knot - 4;
    for (int i = 0; i < 4; i++) {
        int j = static_cast<int>(J_line_acc.start_idx) + i - recur_st_id;
        if (j < 0) continue;
        Hi.block(0, j * 6, 3, 3) = RT * J_line_acc.d_val_d_knot[i];
        Hi.block(0, j * 6 + 3, 3, 3) = drot * J_ortdel.d_val_d_knot[i];
        Hi.block(3, j * 6 + 3, 3, 3) = J_gyro.d_val_d_knot[i];
    }
    Hi.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
    Hi.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
    imu_data.imu_itp.head<3>() = RT * a_w + ba;
    imu_data.imu_itp.tail<3>() = rot_vel + bg;
    imu_data.H = Hi;

    Eigen::Matrix<double, 6, 1> imu;
    imu.head<3>() = imu_data.accel;
    imu.tail<3>() = imu_data.gyro;
    imu_data.imu_itp.head<3>() = RT * a_w + ba;
    imu_data.imu_itp.tail<3>() = rot_vel + bg;
    imu_data.z = imu - imu_data.imu_itp;

    for (int i = 0; i < 3; i++) {
        imu_data.if_acc_valid[i] = std::abs(imu_data.accel(i) - imu_data.imu_itp(i)) <= param->imu_acc_outlier;
        if (!imu_data.if_acc_valid[i]) {
            imu_data.H.row(i).setZero();
            imu_data.z(i) = 0;
        }
        imu_data.if_gyro_valid[i] = std::abs(imu_data.gyro(i) - imu_data.imu_itp(i + 3)) <= param->imu_gyro_outlier;
        if (!imu_data.if_gyro_valid[i]) {
            imu_data.H.row(i + 3).setZero();
            imu_data.z(i + 3) = 0;
        }
    }
}

void RESPLE_LIO::prepLiDAR(PointData& pt_data) const {
    if (pt_data.if_valid) {
        Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
        Eigen::Quaterniond q_itp;
        Jacobian43 J_ortdel;
        Jacobian J_pos;
        spl.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel);
        Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos);
        Eigen::Vector3d pt_w = q_itp * pt_data.point_body + p_itp;
        pt_data.z = pt_data.ppl_normal.dot(pt_w) + pt_data.ppl_d;

        Eigen::Matrix<double, 3, 4> drot;
        dRot(pt_data.point_body, q_itp, drot);
        Eigen::Matrix<double, 1, 4> tmp = pt_data.ppl_normal.transpose() * drot;
        for (int i = 0; i < 4; i++) {
            Hi.block(0, i * 6, 1, 3) = pt_data.ppl_normal.transpose() * J_pos.d_val_d_knot[i];
            Hi.block(0, i * 6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
        }
        pt_data.H = Hi.leftCols<24>();
    }
}

void RESPLE_LIO::propRCP() {
    Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
    Eigen::Matrix<double, 6, 1> cp_prop_pos = 2 * cps_win.block<6, 1>(12, 0) - cps_win.block<6, 1>(0, 0);
    Eigen::Vector3d delta = cps_win.segment<3>(9);
    spl.addOneStateKnot(cp_prop_pos.head<3>(), delta);
    cov_rcp = a_mat * cov_rcp * a_mat.transpose() + cov_sys;
}

void RESPLE_LIO::UpdateLiDARInertial(int eff_lidar_num, const Eigen::Matrix<double, XSIZE, 1>& x_prop,
                                     const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop) {
    Eigen::Matrix<double, 6, 1> cov_imu_inv;
    cov_imu_inv(0) = 1. / param->cov_acc(0);
    cov_imu_inv(1) = 1. / param->cov_acc(1);
    cov_imu_inv(2) = 1. / param->cov_acc(2);
    cov_imu_inv(3) = 1. / param->cov_gyro(0);
    cov_imu_inv(4) = 1. / param->cov_gyro(1);
    cov_imu_inv(5) = 1. / param->cov_gyro(2);

    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) schedule(dynamic) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; i++) {
        PointData& pt_data = pt_meas[i];
        prepLiDAR(pt_data);
    }
    for (size_t i = 0; i < imu_meas.size(); i++) {
        prepIMU(imu_meas[i]);
    }
    int num_tdcp = 0;
    for (int i = 0; i < tdcp_meas.size(); i++) {
        prepTdCp(tdcp_meas[i]);
        if (tdcp_meas[i].if_valid) ++num_tdcp;
    }
    for (int i = 0; i < psr_meas.size(); ++i) {
        prepPsr(psr_meas[i]);
    }
    int dim_meas = 6 * imu_meas.size() + eff_lidar_num + num_tdcp;
    Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(dim_meas, XSIZE);
    Eigen::Matrix<double, Eigen::Dynamic, 1> z(dim_meas, 1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(dim_meas, 1);
    H.setZero();
    z.setZero();
    mat_cov_inv.setZero();
    int idx_offset = 0;
    for (const PointData& pt_data : pt_meas) {
        if (!pt_data.if_valid) continue;
        double lid_cov = pt_data.H * cov_rcp.topLeftCorner<24, 24>() * pt_data.H.transpose() + pt_data.pt_wt;
        if (std::abs(pt_data.z) < param->ppl_thr || lid_cov < pt_data.pt_wt * param->coeff_cov) {
            z(idx_offset) = -pt_data.z;
            H.block(idx_offset, 0, 1, 24) = pt_data.H;
        }
        mat_cov_inv(idx_offset) = 1 / pt_data.pt_wt;
        idx_offset++;
    }
    for (const ImuData& imu_data : imu_meas) {
        z.segment<6>(idx_offset) = imu_data.z;
        H.block(idx_offset, 0, 6, XSIZE) = imu_data.H;
        mat_cov_inv.segment<6>(idx_offset) = cov_imu_inv;
        idx_offset += 6;
    }
    for (const TimeDiffCarrierPhaseData& tdcp_data : tdcp_meas) {
        if (!tdcp_data.if_valid) continue;
        z(idx_offset) = tdcp_data.z;
        H.row(idx_offset) = tdcp_data.H;
        mat_cov_inv(idx_offset) = 1.0 / tdcp_data.var;
        idx_offset++;
    }
    Update(z, mat_cov_inv, H, x_prop, P_prop);
}

bool RESPLE_LIO::TrajectoryStill() {
    const int64_t t1 = spl.maxTimeNs() - param->dt_ns;
    const int64_t t0 = t1 - S_TO_NS(param->zupt_.window);
    if (t0 < spl.start_t_ns) return false;
    const Eigen::Isometry3d a = spl.Interpolate(t0), b = spl.Interpolate(t1);
    return (b.translation() - a.translation()).norm() < param->zupt_.max_move &&
           Eigen::AngleAxisd(a.linear().transpose() * b.linear()).angle() < param->zupt_.max_turn;
}

void RESPLE_LIO::ZeroVelocityUpdate() {
    // rows: linear velocity, then angular rate, at the start of the newest knot interval
    const int64_t t_ns = spl.maxTimeNs() - param->dt_ns;
    Eigen::Matrix<double, 6, XSIZE> H = Eigen::Matrix<double, 6, XSIZE>::Zero();
    Eigen::Matrix<double, 6, 1> innov, R_inv;
    Eigen::Quaterniond q;
    Vec3 rot_vel;
    Jacobian43 J_ortdel;
    Jacobian33 J_gyro;
    Jacobian J_vel;
    spl.itpQuaternion(t_ns, &q, &rot_vel, &J_ortdel, &J_gyro);
    innov.head<3>() = -spl.itpPosition<1>(t_ns, &J_vel);
    innov.tail<3>() = -rot_vel;
    const int recur_st_id = spl.num_knot - 4;
    for (int i = 0; i < 4; ++i) {
        const int j = static_cast<int>(J_vel.start_idx) + i - recur_st_id;
        if (j < 0) continue;
        H.block<3, 3>(0, j * 6) = J_vel.d_val_d_knot[i] * Mat3::Identity();
        H.block<3, 3>(3, j * 6 + 3) = J_gyro.d_val_d_knot[i];
    }
    R_inv.head<3>().setConstant(1.0 / (param->zupt_.std_vel * param->zupt_.std_vel));
    R_inv.tail<3>().setConstant(1.0 / (param->zupt_.std_gyr * param->zupt_.std_gyr));

    const Eigen::Matrix<double, XSIZE, XSIZE> cov_prior = cov_rcp;
    const Eigen::Matrix<double, XSIZE, 1> x_prior = getState();
    Update<6>(innov, R_inv, H, x_prior, cov_prior);
    cov_rcp = (Eigen::Matrix<double, XSIZE, XSIZE>::Identity() - KH) * cov_prior;
    cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());
}

void RESPLE_LIO::UpdateLocalBA(const MeasureGroup& measures) {
    if (is_static) {
        // the scans of a sensor at rest add nothing to the map
        lba_points_.clear();
        lba_last_cut_ns_ = spl.maxTimeNs() - param->dt_ns;
        return;
    }
    for (const Point& pt : measures.lidar_->points) {
        const Vec3 p_body = param->extrin_il_ * Vec3(pt.getVector3fMap().cast<double>());
        Eigen::Quaterniond q;
        spl.itpQuaternion(pt.GetTimeNs(), &q);
        const Mat3 R = q.toRotationMatrix();
        LbaPoint pv;
        pv.p = R * p_body + spl.itpPosition(pt.GetTimeNs());
        pv.var = R * BodyPointCov(p_body, param->lba_.range_err, param->lba_.angle_err) * R.transpose();
        lba_points_.emplace_back(pt.GetTimeNs(), pv);
    }
    // The spline has no frames, so one closes every frame_dt at the start of this knot interval: the
    // points since the last cut, moved into the body frame at this instant.
    const int64_t t_ns = spl.maxTimeNs() - param->dt_ns;
    if (lba_last_cut_ns_ < 0) lba_last_cut_ns_ = t_ns;
    if (t_ns - lba_last_cut_ns_ < S_TO_NS(param->lba_.frame_dt)) return;
    lba_last_cut_ns_ = t_ns;
    const Eigen::Isometry3d pose = spl.Interpolate(t_ns);
    LbaState odom;
    odom.R = pose.linear();
    odom.p = pose.translation();
    odom.v = spl.itpPosition<1>(t_ns);
    odom.bg = bg;
    odom.ba = ba;
    std::vector<LbaPoint> points;
    while (!lba_points_.empty() && lba_points_.front().first <= t_ns) {
        const LbaPoint& pv = lba_points_.front().second;
        points.push_back(LbaPoint{odom.R.transpose() * (pv.p - odom.p),
                                  odom.R.transpose() * pv.var * odom.R});
        lba_points_.pop_front();
    }
    // local ba
    if (lba->AddFrame(t_ns, odom, points, gravity))
        // align the front end
        AlignToBA(t_ns, lba->Newest());
    lba->SetOdomPose(spl.Interpolate(t_ns));
}

void RESPLE_LIO::AlignToBA(int64_t t_ns, const LbaState& x) {
    // Rows: position, then the three body axes in the world frame.  The axes carry the rotation
    // with the Jacobian the LiDAR rows already use; their 9 rows hold 2 / sigma^2 per angle.
    Eigen::Matrix<double, 12, XSIZE> H = Eigen::Matrix<double, 12, XSIZE>::Zero();
    Eigen::Matrix<double, 12, 1> innov, R_inv;
    Eigen::Quaterniond q;
    Jacobian43 J_ortdel;
    Jacobian J_pos;
    spl.itpQuaternion(t_ns, &q, nullptr, &J_ortdel);
    const Vec3 p = spl.itpPosition(t_ns, &J_pos);
    const Mat3 R = q.toRotationMatrix();
    innov.head<3>() = x.p - p;
    const int recur_st_id = spl.num_knot - 4;
    for (int k = 0; k < 3; ++k) {
        innov.segment<3>(3 + 3 * k) = x.R.col(k) - R.col(k);
        Eigen::Matrix<double, 3, 4> drot;
        dRot(Vec3(Vec3::Unit(k)), q, drot);
        for (int i = 0; i < 4; ++i) {
            const int j = static_cast<int>(J_ortdel.start_idx) + i - recur_st_id;
            if (j < 0) continue;
            H.block<3, 3>(3 + 3 * k, j * 6 + 3) = drot * J_ortdel.d_val_d_knot[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        const int j = static_cast<int>(J_pos.start_idx) + i - recur_st_id;
        if (j < 0) continue;
        H.block<3, 3>(0, j * 6) = J_pos.d_val_d_knot[i] * Mat3::Identity();
    }
    R_inv.head<3>().setConstant(1.0 / (param->lba_.fb_std_pos * param->lba_.fb_std_pos));
    R_inv.tail<9>().setConstant(0.5 / (param->lba_.fb_std_ort * param->lba_.fb_std_ort));

    const Eigen::Matrix<double, XSIZE, XSIZE> cov_prior = cov_rcp;
    const Eigen::Matrix<double, XSIZE, 1> x_prior = getState();
    Update<12>(innov, R_inv, H, x_prior, cov_prior);
    if (!param->lba_.fb_keep_cov) {
        cov_rcp = (Eigen::Matrix<double, XSIZE, XSIZE>::Identity() - KH) * cov_prior;
        cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());
    }
    const Eigen::Isometry3d after = spl.Interpolate(t_ns);
    LOG(INFO) << "LBA align: offset " << innov.head<3>().norm() << " m " << innov.tail<9>().norm() / std::sqrt(2.0)
              << " rad, left " << (x.p - after.translation()).norm() << " m";
}

void RESPLE_LIO::UpdateState(const Eigen::Matrix<double, XSIZE, 1>& xupd) {
    Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
    for (int i = 0; i < 4; ++i) {
        int knot_id = spl.num_knot - 4 + i;
        CHECK(knot_id >= 0);
        spl.t_knots[knot_id] = cp_win.segment<3>(i * 6);
        spl.ort_delta[knot_id] = cp_win.segment<3>(i * 6 + 3);
        Eigen::Quaterniond q_del = faster_lio::ExpQuatHalf(spl.ort_delta[knot_id]);
        if (knot_id == 0) {
            spl.q_knots[knot_id] = spl.q_idle * q_del;
        } else {
            spl.q_knots[knot_id] = spl.q_knots[knot_id - 1] * q_del;
        }
    }
    ba = xupd.segment(BA_OFFSET, 3);
    bg = xupd.segment(BG_OFFSET, 3);
    clk = xupd.segment<4>(CLK_OFFSET);
    theta_ecef_enu = xupd(ROT_OFFSET);
    t_ecef_enu = xupd.segment<3>(TRANS_OFFSET);
}

}  // namespace faster_lio

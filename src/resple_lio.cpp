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

    // ikd tree
    kd_tree = std::make_shared<KD_TREE<faster_lio::Point>>();
    kd_tree->set_downsample_param(param->ds_lm_voxel);
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
    }

    IterativeUpdate();
    const int num_pt = static_cast<int>(pt_meas.size());
#pragma omp parallel for num_threads(param->num_threads) if (num_pt > kMinParallelPoints)
    for (int i = 0; i < num_pt; ++i) {
        PointData& pt_data = pt_meas[i];
        Vec3 interp_pos = spl.itpPosition(pt_data.time_ns);
        Eigen::Quaterniond interp_q;
        spl.itpQuaternion(pt_data.time_ns, &interp_q);
        pt_data.point_world = interp_q * pt_data.point_body + interp_pos;
    }
    accum_points.reserve(accum_points.size() + pt_meas.size());
    accum_nearest_points.reserve(accum_nearest_points.size() + pt_meas.size());
    for (int i = 0; i < pt_meas.size(); ++i) {
        faster_lio::Point point{};
        point.getVector3fMap() = pt_meas[i].point_world.cast<float>();
        point.SetTimeNs(pt_meas[i].time_ns);
        accum_points.push_back(point);
        accum_nearest_points.push_back(pt_meas[i].near_points);
    }
    if (accum_points.size() >= 2) {
        int64_t accum_points_interval = accum_points.back().GetTimeNs() - accum_points.front().GetTimeNs();
        if (accum_points_interval > S_TO_NS(0.1)) {
            mapIncremental();
            LaserMapFovSegment();
            LOG(INFO) << "Add Points: " << accum_points.size();
            accum_points.clear();
            accum_nearest_points.clear();
        }
    }

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
        const int sys_idx = SysIdx(satsys(obs.sat, nullptr));
        if (sys_idx < 0) continue;
        double azel[2] = {0.0, PI / 2.0};
        double e[3];
        if (geodist(&rs[si * 6], sol.rr, e) < 0.0) continue;
        if (satazel(rcv_llh, e, azel) < param->prcopt.elmin) continue;
        if (testsnr(0, 0, azel[1], obs.SNR[0], &param->prcopt.snrmask)) continue;

        const double freq = sat2freq(obs.sat, obs.code[0], &nav);
        if (freq <= 0.0) continue;
        const double lambda = CLIGHT / freq;

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
        } else {
            // clear the track
            sat_tracks_[obs.sat].obs_ids.clear();
            sat_tracks_[obs.sat].sv_poss.clear();
            sat_tracks_[obs.sat].local_times.clear();
            sat_tracks_[obs.sat].sv_dts.clear();
            // insert the one observation
            sat_tracks_[obs.sat].obs_ids.push_back(oi);
            sat_tracks_[obs.sat].sv_poss.push_back(sv);
            sat_tracks_[obs.sat].local_times.push_back(local_time);
            sat_tracks_[obs.sat].sv_dts.push_back(dts[si * 2 + 0]);
        }
        // TDCP against the previous epoch, only once the local frame is tied to ECEF
        SatTrack& track = sat_tracks_[obs.sat];
        const int N = static_cast<int>(track.obs_ids.size());
        if (if_GNSS_aligned && param->tdcp_enable && arc_continu && N >= 2) {
            for (int i = 0; i < N -1; ++i) {
                const obsd_t& anchor_obs = obss.data[track.obs_ids[N - 2]];
                const double sin_el = std::max(std::sin(azel[1]), 0.1);
                TimeDiffCarrierPhaseData tdcp_data;
                tdcp_data.time_ns = local_time;
                tdcp_data.anchor_time_ns = track.local_times[i];
                tdcp_data.sv_pos = sv;
                tdcp_data.sv_pos_anchor = track.sv_poss[i];
                tdcp_data.sv_dt = dts[si * 2 + 0];
                tdcp_data.sv_dt_anchor = track.sv_dts[i];
                tdcp_data.cp_diff = lambda * (obs.L[0] - anchor_obs.L[0]);
                tdcp_data.sys_idx = sys_idx;
                tdcp_data.var = param->tdcp_std * param->tdcp_std / (sin_el * sin_el);
                tdcp_data.if_valid = true;
                tdcp_meas.push_back(tdcp_data);
            }
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

Eigen::Matrix<double, RESPLE_LIO::XSIZE, 1> RESPLE_LIO::getState() {
    Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
    Eigen::Matrix<double, XSIZE, 1> state;
    state << cps_win, ba, bg, clk, theta_ecef_enu, t_ecef_enu;
    return state;
}

bool RESPLE_LIO::esti_plane(Vec4& pca_result, const Eigen::aligned_vector<faster_lio::Point>& point,
                            const double& threshold) {
    Eigen::Matrix<double, MATCH_POINT, 3> A;
    Eigen::Matrix<double, MATCH_POINT, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;
    for (int j = 0; j < MATCH_POINT; j++) {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }
    Eigen::Matrix<double, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
    double n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;
    for (int j = 0; j < MATCH_POINT; j++) {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) >
            threshold) {
            return false;
        }
    }
    return true;
}

int RESPLE_LIO::Match() {
    const float nn_radius = static_cast<float>(param->nn_search_radius);
    const float nn_max_sq = static_cast<float>(param->nn_search_radius * param->nn_search_radius);
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
        faster_lio::Point point_w{};
        point_w.getVector3fMap() = pt_data.point_world.cast<float>();
        // nearest search
        std::vector<float> point_search_dists;
        kd_tree->Nearest_Search(point_w, MATCH_POINT, pt_data.near_points, point_search_dists, nn_radius);
        if (static_cast<int>(pt_data.near_points.size()) >= MATCH_POINT &&
            point_search_dists[MATCH_POINT - 1] < nn_max_sq) {
            Vec4 abcd = Vec4::Zero();
            if (esti_plane(abcd, pt_data.near_points, param->plane_thresh)) {
                double pd2 = abcd.dot(pt_data.point_world.homogeneous());
                if (pt_data.point_body.norm() > 81.0 * pd2 * pd2) {
                    pt_data.if_valid = true;
                    pt_data.ppl_normal = abcd.head<3>().normalized();
                    pt_data.ppl_d = abcd(3);
                }
            }
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
        if ((t > n_iter) || (i == max_iter - 1)) {
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

void RESPLE_LIO::LaserMapFovSegment() {
    Vec3 pos_lidar = spl.itpPosition(spl.maxTimeNs() - 1);
    if (!localmap_box_initialized) {
        for (int i = 0; i < 3; i++) {
            localmap_box.vertex_min[i] = pos_lidar(i) - param->cube_len / 2.0;
            localmap_box.vertex_max[i] = pos_lidar(i) + param->cube_len / 2.0;
        }
        localmap_box_initialized = true;
        return;
    }
    const float mov_threshold = 1.5;
    const float det_range = 100.0;
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_lidar(i) - localmap_box.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_lidar(i) - localmap_box.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= mov_threshold * det_range || dist_to_map_edge[i][1] <= mov_threshold * det_range)
            need_move = true;
    }
    if (!need_move) return;
    std::vector<BoxPointType> cub_needrm;
    BoxPointType new_localmap_box = localmap_box, tmp_box;
    float mov_dist = std::max((param->cube_len - 2.0 * mov_threshold * det_range) * 0.5 * 0.9,
                              double(det_range * (mov_threshold - 1)));
    for (int i = 0; i < 3; i++) {
        tmp_box = localmap_box;
        if (dist_to_map_edge[i][0] <= mov_threshold * det_range) {
            new_localmap_box.vertex_max[i] -= mov_dist;
            new_localmap_box.vertex_min[i] -= mov_dist;
            tmp_box.vertex_min[i] = localmap_box.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_box);
        } else if (dist_to_map_edge[i][1] <= mov_threshold * det_range) {
            new_localmap_box.vertex_max[i] += mov_dist;
            new_localmap_box.vertex_min[i] += mov_dist;
            tmp_box.vertex_max[i] = localmap_box.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_box);
        }
    }
    localmap_box = new_localmap_box;
    if (!cub_needrm.empty()) kd_tree->Delete_Point_Boxes(cub_needrm);
}

void RESPLE_LIO::mapIncremental() {
    Eigen::aligned_vector<faster_lio::Point> PointToAdd;
    Eigen::aligned_vector<faster_lio::Point> PointNoNeedDownsample;
    int feats_down_size = accum_points.points.size();
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++) {
        const faster_lio::Point& point = accum_points.points[i];
        if (!accum_nearest_points[i].empty()) {
            Eigen::aligned_vector<faster_lio::Point>& points_near = accum_nearest_points[i];
            bool need_add = true;
            double ds_lm_voxel = param->ds_lm_voxel;
            faster_lio::Point mid_point;
            mid_point.x = floor(point.x / ds_lm_voxel) * ds_lm_voxel + 0.5 * ds_lm_voxel;
            mid_point.y = floor(point.y / ds_lm_voxel) * ds_lm_voxel + 0.5 * ds_lm_voxel;
            mid_point.z = floor(point.z / ds_lm_voxel) * ds_lm_voxel + 0.5 * ds_lm_voxel;
            if (fabs(points_near[0].x - mid_point.x) > 0.866 * ds_lm_voxel ||
                fabs(points_near[0].y - mid_point.y) > 0.866 * ds_lm_voxel ||
                fabs(points_near[0].z - mid_point.z) > 0.866 * ds_lm_voxel) {
                PointNoNeedDownsample.emplace_back(accum_points.points[i]);
                continue;
            }
            for (size_t readd_i = 0; readd_i < points_near.size(); readd_i++) {
                if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * ds_lm_voxel &&
                    fabs(points_near[readd_i].y - mid_point.y) < 0.5 * ds_lm_voxel &&
                    fabs(points_near[readd_i].z - mid_point.z) < 0.5 * ds_lm_voxel) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.emplace_back(point);
        } else {
            PointNoNeedDownsample.emplace_back(point);
        }
    }
    kd_tree->Add_Points(PointToAdd, true);
    kd_tree->Add_Points(PointNoNeedDownsample, false);
}

}  // namespace faster_lio

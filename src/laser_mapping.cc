#include <iomanip>
#include <map>
#include <sstream>

#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/filesystem.hpp>
#include "laser_mapping.h"

#include <pcl/io/ply_io.h>

#include "rtklib_.h"
#include "rtklib_func.h"
namespace fs = boost::filesystem;

int showmsg(const char* format, ...) { return 0; }
void settspan(gtime_t ts, gtime_t te) {}
void settime(gtime_t time) {}

namespace faster_lio {
namespace {

constexpr double kInitCovGyro = 0.1;
constexpr double kInitCovAcc = 0.1;
constexpr double kInitRandWalk = 1e-4;
constexpr double kInitImuCoef = 1e-4;


int SysIdx(int sys) {
    if (sys == SYS_GPS || sys == SYS_QZS) return 0;
    if (sys == SYS_GLO) return 1;
    if (sys == SYS_GAL) return 2;
    if (sys == SYS_CMP) return 3;
    return -1;
}

// One time-differenced carrier-phase observation
struct TdcpRow {
    int anchor_idx = -1;
    int cur_idx = -1;
    int sys_idx = 0;
    Vec3 u_enu = Vec3::Zero();
    double z = 0.0;
    double var = 1.0;
};

// The yaw-free TDCP factor
struct TdcpFactor {
    int i = -1;  // anchor state
    int j = -1;  // current state
    Eigen::Vector2d y = Eigen::Vector2d::Zero();      // (up displacement, horizontal displacement) [m]
    Eigen::Matrix2d info = Eigen::Matrix2d::Zero();   // Sigma_y^-1; row/col 1 is zero when the
                                                      // horizontal step is too short to have a direction
    int num_sat = 0;
};

struct State {
    std::int64_t time_ns = 0;
    Mat3 R = Mat3::Identity();
    Vec3 p = Vec3::Zero();
    Vec3 v = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
    std::vector<double> dts;
};

// Zero-order-hold IMU integration over [t_b, t_e): the sample at or before t_b covers the gap up
// to the next sample, so consecutive intervals tile time exactly. f(imu, seg_begin_ns, dt_s).
template <typename F>
void ForEachImuInterval(const std::vector<Imu>& imus, std::int64_t t_b, std::int64_t t_e, F&& f) {
    if (imus.empty() || t_e <= t_b) return;
    // find the first imu which bigger than the t_b
    auto it =
        std::upper_bound(imus.begin(), imus.end(), t_b, [](std::int64_t t, const Imu& imu) { return t < imu.time_ns; });
    // find the one before
    if (it != imus.begin()) --it;
    for (; it != imus.end() && it->time_ns < t_e; ++it) {
        std::int64_t head = std::max(it->time_ns, t_b);
        std::int64_t tail = (it + 1 == imus.end()) ? t_e : std::min((it + 1)->time_ns, t_e);
        if (tail > head) f(*it, head, NS_TO_S(tail - head));
    }
}

// Voxel-SLAM IMU_PRE
struct ImuPreintegration {
    double dt_sum = 0.0;
    Mat3 dR = Mat3::Identity();
    Vec3 dv = Vec3::Zero();
    Vec3 dp = Vec3::Zero();
    Vec3 bg_lin = Vec3::Zero(), ba_lin = Vec3::Zero();
    Mat3 dR_dbg = Mat3::Zero();
    Mat3 dv_dbg = Mat3::Zero(), dv_dba = Mat3::Zero();
    Mat3 dp_dbg = Mat3::Zero(), dp_dba = Mat3::Zero();
    Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Zero();  // [R, p, v, bg, ba]

    void reset(const Vec3& bg, const Vec3& ba) {
        dt_sum = 0.0;
        dR.setIdentity();
        dv.setZero();
        dp.setZero();
        bg_lin = bg;
        ba_lin = ba;
        dR_dbg.setZero();
        dv_dbg.setZero();
        dv_dba.setZero();
        dp_dbg.setZero();
        dp_dba.setZero();
        cov.setZero();
    }

    void integrate(const Vec3& gyro, const Vec3& acc, double dt) {
        if (dt <= 0.0) return;
        Vec3 w = gyro - bg_lin;
        Vec3 a = acc - ba_lin;
        Vec3 w_dt = w * dt;
        Mat3 dRk = ExpMat(w_dt);
        Mat3 Jr;
        rightJacobianSO3(w_dt, Jr);
        Mat3 a_hat = Hat(a);

        Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
        Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
        A.block<3, 3>(0, 0) = dRk.transpose();
        A.block<3, 3>(3, 0) = -0.5 * dt * dt * dR * a_hat;
        A.block<3, 3>(3, 6) = Mat3::Identity() * dt;
        A.block<3, 3>(6, 0) = -dt * dR * a_hat;
        B.block<3, 3>(0, 0) = Jr * dt;
        B.block<3, 3>(3, 3) = 0.5 * dt * dt * dR;
        B.block<3, 3>(6, 3) = dt * dR;
        Eigen::Matrix<double, 6, 1> noise_meas;
        noise_meas << kInitCovGyro, kInitCovGyro, kInitCovGyro, kInitCovAcc, kInitCovAcc, kInitCovAcc;
        cov.topLeftCorner<9, 9>() =
            A * cov.topLeftCorner<9, 9>() * A.transpose() + B * noise_meas.asDiagonal() * B.transpose();
        cov.bottomRightCorner<6, 6>().diagonal().array() += kInitRandWalk * dt;

        dp_dba += dv_dba * dt - 0.5 * dR * dt * dt;
        dp_dbg += dv_dbg * dt - 0.5 * dR * a_hat * dR_dbg * dt * dt;
        dv_dba += -dR * dt;
        dv_dbg += -dR * a_hat * dR_dbg * dt;
        dR_dbg = dRk.transpose() * dR_dbg - Jr * dt;

        dp += dv * dt + 0.5 * dR * a * dt * dt;
        dv += dR * a * dt;
        dR = dR * dRk;
        dt_sum += dt;
    }

    // IMU_PRE::give_evaluate_g. J columns: [state i (15) | state j (15) | gravity (3)], each state
    // ordered [R, p, v, bg, ba]
    double Evaluate(const State& si, const State& sj, const Vec3& g, Eigen::Matrix<double, 15, 33>* J,
                    Eigen::Matrix<double, 15, 1>* r_out, Eigen::Matrix<double, 15, 15>* cov_inv_out) const {
        const double T = dt_sum;
        Vec3 dbg = si.bg - bg_lin;
        Vec3 dba = si.ba - ba_lin;
        Vec3 dR_corr_vec = dR_dbg * dbg;
        Mat3 R_corr = dR * ExpMat(dR_corr_vec);
        Vec3 p_corr = dp + dp_dbg * dbg + dp_dba * dba;
        Vec3 v_corr = dv + dv_dbg * dbg + dv_dba * dba;

        Mat3 res_R = R_corr.transpose() * si.R.transpose() * sj.R;
        Vec3 exp_p = si.R.transpose() * (sj.p - si.p - si.v * T - 0.5 * T * T * g);
        Vec3 exp_v = si.R.transpose() * (sj.v - si.v - T * g);

        Eigen::Matrix<double, 15, 1> r;
        r << LogMat(res_R), exp_p - p_corr, exp_v - v_corr, sj.bg - si.bg, sj.ba - si.ba;
        Eigen::Matrix<double, 15, 15> cov_inv = cov.inverse();

        if (J) {
            Vec3 r_R = r.head<3>();
            Mat3 Jr_res, Jr_bg;
            rightJacobianSO3(r_R, Jr_res);
            rightJacobianSO3(dR_corr_vec, Jr_bg);
            Mat3 Jr_inv = Jr_res.inverse();
            J->setZero();
            J->block<3, 3>(0, 0) = -Jr_inv * sj.R.transpose() * si.R;
            J->block<3, 3>(0, 9) = -Jr_inv * res_R.transpose() * Jr_bg * dR_dbg;
            J->block<3, 3>(0, 15) = Jr_inv;

            J->block<3, 3>(3, 0) = Hat(exp_p);
            J->block<3, 3>(3, 3) = -si.R.transpose();
            J->block<3, 3>(3, 6) = -si.R.transpose() * T;
            J->block<3, 3>(3, 9) = -dp_dbg;
            J->block<3, 3>(3, 12) = -dp_dba;
            J->block<3, 3>(3, 18) = si.R.transpose();
            J->block<3, 3>(3, 30) = -0.5 * T * T * si.R.transpose();

            J->block<3, 3>(6, 0) = Hat(exp_v);
            J->block<3, 3>(6, 6) = -si.R.transpose();
            J->block<3, 3>(6, 9) = -dv_dbg;
            J->block<3, 3>(6, 12) = -dv_dba;
            J->block<3, 3>(6, 21) = si.R.transpose();
            J->block<3, 3>(6, 30) = -T * si.R.transpose();

            J->block<3, 3>(9, 9) = -Mat3::Identity();
            J->block<3, 3>(9, 24) = Mat3::Identity();
            J->block<3, 3>(12, 12) = -Mat3::Identity();
            J->block<3, 3>(12, 27) = Mat3::Identity();
        }
        if (r_out) *r_out = r;
        if (cov_inv_out) *cov_inv_out = cov_inv;
        return r.dot(cov_inv * r);
    }
};

struct LidarBAFactor {
    int win_size = 0;
    std::vector<std::vector<PointCluster>> pcrs;
    std::vector<Vec3> eig_values;
    std::vector<Mat3> eig_vectors;
    std::vector<PointCluster> pcr_adds;
    std::vector<double> coeffs;

    void clear() {
        pcrs.clear();
        eig_values.clear();
        eig_vectors.clear();
        pcr_adds.clear();
        coeffs.clear();
    }

    void build(const std::vector<State>& states, const std::vector<std::vector<Vec3>>& scan_points, double voxel_size,
               double ratio_thresh, double abs_thresh, int min_pts) {
        clear();
        win_size = (int)states.size();
        std::unordered_map<VOXEL_LOCATION, std::vector<PointCluster>> vox_cls_map;

        // push to voxel clusters
        for (int i = 0; i < win_size; i++) {
            for (const auto& p : scan_points[i]) {
                Vec3 pw = states[i].R * p + states[i].p;
                VOXEL_LOCATION vox = VOXEL_LOCATION(pw, voxel_size);
                auto it = vox_cls_map.find(vox);
                if (it == vox_cls_map.end()) {
                    std::vector<PointCluster> cls(win_size);
                    cls[i].Push(p);
                    vox_cls_map.emplace(vox, std::move(cls));
                } else {
                    it->second[i].Push(p);
                }
            }
        }

        // sum the cluster
        for (auto& vox_cls : vox_cls_map) {
            std::vector<PointCluster>& cls = vox_cls.second;
            PointCluster agg_cl;
            int total = 0;
            for (int i = 0; i < win_size; i++)
                if (cls[i].N > 0) {
                    PointCluster cl_tran;
                    cl_tran.Transform(cls[i], states[i].R, states[i].p);
                    agg_cl += cl_tran;
                    total += cls[i].N;
                }
            if (total < min_pts) continue;

            Eigen::SelfAdjointEigenSolver<Mat3> es(agg_cl.Cov());
            Vec3 evals = es.eigenvalues();
            if (evals(0) < abs_thresh && evals(0) < ratio_thresh * std::max(evals(1), 1e-9)) {
                pcrs.push_back(std::move(cls));
                eig_values.push_back(evals);
                eig_vectors.push_back(es.eigenvectors());
                pcr_adds.push_back(agg_cl);
                coeffs.push_back(1.0);
            }
        }
    }

    double eval_error(const std::vector<State>& states) {
        double residual = 0.0;
        for (size_t v = 0; v < pcrs.size(); v++) {
            PointCluster agg_cl;
            const std::vector<PointCluster>& cls = pcrs[v];
            for (int i = 0; i < win_size; i++)
                if (cls[i].N > 0) {
                    PointCluster cl_w;
                    cl_w.Transform(cls[i], states[i].R, states[i].p);
                    agg_cl += cl_w;
                }
            Eigen::SelfAdjointEigenSolver<Mat3> es(agg_cl.Cov());
            eig_values[v] = es.eigenvalues();
            eig_vectors[v] = es.eigenvectors();
            pcr_adds[v] = agg_cl;
            residual += coeffs[v] * eig_values[v](0);
        }
        return residual;
    }

    void eval_H(const std::vector<State>& states, MatX& H, VecX& b, int state_stride) const {
        std::vector<Vec3> viRiTuk(win_size);
        std::vector<Mat3> viRiTukukT(win_size);
        std::vector<Eigen::Matrix<double, 3, 6>> Auk(win_size);

        for (size_t a = 0; a < pcrs.size(); a++) {
            const std::vector<PointCluster>& sig_orig = pcrs[a];
            double coe = coeffs[a];

            const Vec3& lmbd = eig_values[a];
            const Mat3& U = eig_vectors[a];
            Vec3 u0 = U.col(0), u1 = U.col(1), u2 = U.col(2);
            const Vec3& uk = u0;
            Mat3 ukukT = uk * uk.transpose();

            Mat3 umumT = Mat3::Zero();
            umumT += 2.0 / (lmbd[0] - lmbd[1]) * u1 * u1.transpose();
            umumT += 2.0 / (lmbd[0] - lmbd[2]) * u2 * u2.transpose();

            int NN = pcr_adds[a].N;
            Vec3 vBar = pcr_adds[a].v / (double)NN;

            for (int i = 0; i < win_size; i++)
                if (sig_orig[i].N != 0) {
                    const Mat3& Pi = sig_orig[i].P;
                    const Vec3& vi = sig_orig[i].v;
                    const Mat3& Ri = states[i].R;
                    double ni = sig_orig[i].N;

                    Mat3 vihat = Hat(vi);
                    Vec3 RiTuk = Ri.transpose() * uk;
                    Mat3 RiTukhat = Hat(RiTuk);

                    Vec3 PiRiTuk = Pi * RiTuk;
                    viRiTuk[i] = vihat * RiTuk;
                    viRiTukukT[i] = viRiTuk[i] * uk.transpose();

                    Vec3 ti_v = states[i].p - vBar;
                    double ukTti_v = uk.dot(ti_v);

                    Mat3 combo1 = Hat(PiRiTuk) + vihat * ukTti_v;
                    Vec3 combo2 = Ri * vi + ni * ti_v;

                    Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
                    Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * Mat3::Identity();
                    Auk[i] /= (double)NN;

                    Eigen::Matrix<double, 6, 1> jjt = Auk[i].transpose() * uk;
                    int bi = state_stride * i;
                    b.segment(bi, 6) -= coe * jjt;

                    Mat3 HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
                    Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
                    Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat -
                                            2.0 / (NN * (double)NN) * viRiTuk[i] * viRiTuk[i].transpose() -
                                            0.5 * Hat(jjt.segment<3>(0));
                    Hb.block<3, 3>(0, 3) += HRt;
                    Hb.block<3, 3>(3, 0) += HRt.transpose();
                    Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

                    H.block<6, 6>(bi, bi) += coe * Hb;
                }

            for (int i = 0; i < win_size - 1; i++)
                if (sig_orig[i].N != 0) {
                    double ni = sig_orig[i].N;
                    for (int j = i + 1; j < win_size; j++)
                        if (sig_orig[j].N != 0) {
                            double nj = sig_orig[j].N;
                            Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
                            Hb.block<3, 3>(0, 0) += -2.0 / (NN * (double)NN) * viRiTuk[i] * viRiTuk[j].transpose();
                            Hb.block<3, 3>(0, 3) += -2.0 * nj / (NN * (double)NN) * viRiTukukT[i];
                            Hb.block<3, 3>(3, 0) += -2.0 * ni / (NN * (double)NN) * viRiTukukT[j].transpose();
                            Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / (NN * (double)NN) * ukukT;

                            int bi = state_stride * i, bj = state_stride * j;
                            H.block<6, 6>(bi, bj) += coe * Hb;
                            H.block<6, 6>(bj, bi) += coe * Hb.transpose();
                        }
                }
        }
    }
};

class VoxelInitializer {
   public:
    std::int64_t start_time_ns;
    std::int64_t end_time_ns;
    int N = 0;
    std::int64_t scan_interval = S_TO_NS(0.1);
    double scan_ds_size;
    double plane_thresh;
    std::shared_ptr<LaserMappingParam> param;
    obs_t obss;
    nav_t nav;
    std::vector<State> states;
    Vec3 gravity;
    std::vector<std::vector<std::pair<std::int64_t, Vec3>>> raw_scans_points;
    std::vector<std::int64_t> scan_end_times;
    std::vector<Imu> imus_;
    std::vector<ImuPreintegration> preints;
    LidarBAFactor lidar_factor;
    double min_normal_eig = 0.0;

    std::unordered_map<int, SatTrack> sat_tracks_;
    std::unordered_map<std::int64_t, int> epoch_idx_;  // GNSS epoch local time -> state index
    std::vector<TdcpRow> tdcp_rows_;
    std::vector<TdcpFactor> tdcp_factors_;
    VoxelInitializer(const std::int64_t start_time_ns, const std::int64_t& end_time_ns, double scan_vs, double plane_th)
        : start_time_ns(start_time_ns), end_time_ns(end_time_ns), scan_ds_size(scan_vs), plane_thresh(plane_th) {
        N = (end_time_ns - start_time_ns) / scan_interval;
    }

    LaserMapping::DynamicInitResult Run(const std::vector<Point>& points, const std::vector<Imu>& imus,
                                        const std::vector<int>& obss_ids,
                                        const std::vector<std::int64_t>& obss_time_ns) {
        // GNSS
        for (int i = 0; i < obss_ids.size(); i++) {
            std::int64_t local_time = obss_time_ns[i];
            int ei = obss_ids[i];
            int n = 0;
            while (ei + n < obss.n && std::abs(timediff(obss.data[ei].time, obss.data[ei + n].time)) < DTTOL) ++n;
            // SPP position
            sol_t sol;
            char err_msg[128];
            if (!pntpos(obss.data + ei, n, &nav, &param->prcopt, &sol, nullptr, nullptr, err_msg)) {
                LOG(INFO) << "SPP failed: " << err_msg;
                continue;  // sol.rr is the linearisation point of every line of sight below
            }
            epoch_idx_[local_time] = i;

            // satellite pos
            std::vector<double> rs(6 * n, 0.0), dts(2 * n, 0.0), svar(n, 0.0);
            std::vector<int> svh(n, 0);
            satposs(sol.time, obss.data + ei, n, &nav, param->prcopt.sateph, rs.data(), dts.data(), svar.data(),
                    svh.data());
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
                const int track_size = static_cast<int>(track.obs_ids.size());
                if (param->tdcp_enable && arc_continu && track_size >= 2) {
                    const int j = track_size - 2;
                    const obsd_t& anchor_obs = obss.data[track.obs_ids[j]];
                    const auto anchor_it = epoch_idx_.find(track.local_times[j]);
                    // both ranges use the same nominal receiver position, so the displacement between
                    // the two epochs is what is left over after the geometric difference is removed
                    double e_anchor[3];
                    const double range_cur = geodist(&rs[si * 6], sol.rr, e);
                    Vec3 sv_anchor = track.sv_poss[j];
                    const double range_anchor = geodist(sv_anchor.data(), sol.rr, e_anchor);
                    if (anchor_it != epoch_idx_.end() && range_cur > 0.0 && range_anchor > 0.0) {
                        const double cp_diff = lambda * (obs.L[0] - anchor_obs.L[0]);
                        const double sv_dt_diff = dts[si * 2 + 0] - track.sv_dts[j];
                        // u points from the satellite to the receiver, i.e. against geodist's e
                        const double u_ecef[3] = {-e[0], -e[1], -e[2]};
                        double u_enu[3];
                        ecef2enu(rcv_llh, u_ecef, u_enu);
                        const double sin_el = std::max(std::sin(azel[1]), std::sin(5.0 * D2R));
                        TdcpRow row;
                        row.anchor_idx = anchor_it->second;
                        row.cur_idx = i;
                        row.sys_idx = sys_idx;
                        row.u_enu = Vec3(u_enu[0], u_enu[1], u_enu[2]);
                        row.z = cp_diff + CLIGHT * sv_dt_diff - (range_cur - range_anchor);
                        // the two epochs contribute one phase noise each; the zenith std also has to
                        // cover multipath and the antenna-to-IMU mismatch, hence param->tdcp_std
                        row.var = 2.0 * (param->tdcp_std / sin_el) * (param->tdcp_std / sin_el);
                        tdcp_rows_.push_back(row);
                    }
                }
            }  // for satellite
        } // for epoch

        // imu
        imus_ = imus;

        // determine the end of the scan time
        if (obss_ids.empty()) {
            scan_end_times.resize(N);
            for (int i = 0; i < N; i++) scan_end_times[i] = start_time_ns + (i + 1) * scan_interval;
        } else {
            N = obss_ids.size();
            scan_end_times.resize(N);
            for (int i = 0; i < N; i++) scan_end_times[i] = obss_time_ns[i];
        }

        // split the points
        SplitScans(points);

        // check the points in the scan
        for (int i = 0; i < N; i++) {
            if (raw_scans_points[i].size() < 50) {
                LOG(WARNING) << "dynamic init: scan " << i << " has only " << raw_scans_points[i].size() << " points";
                return LaserMapping::DynamicInitResult::FAIL;
            }
        }

        // init the states
        states.resize(N, State());
        for (int i = 0; i < N; i++) states[i].time_ns = scan_end_times[i];

        // GNSS displacement factors; the measurement does not depend on the states, so build once
        BuildTdcpFactors();

        // lidra gyro odometry
        if (!InitStateByLGO()) return LaserMapping::DynamicInitResult::FAIL;

        double max_rot = 0.0;
        for (int i = 1; i < N; i++)
            max_rot = std::max(max_rot, LogMat(Mat3(states[0].R.transpose() * states[i].R)).norm());
        double travel = (states.back().p - states.front().p).norm();
        if (travel < 0.05 && max_rot < 1.0 * M_PI / 180.0) {
            LOG(INFO) << "dynamic init: stationary window (travel " << travel << " m)";
            return LaserMapping::DynamicInitResult::STATIONARY;
        }

        // solve the gravity and vel
        BuildPreintegrations();
        if (!InitVelocityGravity()) return LaserMapping::DynamicInitResult::FAIL;
        LogTdcpRms("before BA");

        const double plane_ratio = 0.04, plane_abs = 0.01, voxel_size = 0.35;
        const int min_pts = 6;
        bool converged = false;
        for (int iter = 0; iter < 40; iter++) {
            lidar_factor.build(states, DeskewAll(), voxel_size, plane_ratio, plane_abs, min_pts);
            if (lidar_factor.pcrs.size() < 10) {
                LOG(WARNING) << "dynamic init: only " << lidar_factor.pcrs.size() << " planes";
                return LaserMapping::DynamicInitResult::FAIL;
            }
            auto [cost_before, cost_after] = Optimize(6);
            BuildPreintegrations();

            converged = iter >= 2 && std::fabs(cost_before - cost_after) / std::max(cost_before, 1e-9) < 0.01;
            LOG(INFO) << "dynamic init iter " << iter << " cost before=" << cost_before << " cost after=" << cost_after
                      << " planes=" << lidar_factor.pcrs.size() << " |g|=" << gravity.norm();
            if (converged) break;
        }
        AlignGravity();

        // Voxel-SLAM motion_init degeneracy test: plane normals must span all three directions
        Mat3 nnt = Mat3::Zero();
        for (const Mat3& U : lidar_factor.eig_vectors) nnt += U.col(0) * U.col(0).transpose();
        min_normal_eig = Eigen::SelfAdjointEigenSolver<Mat3>(nnt).eigenvalues()(0);

        LogTdcpRms("after BA");

        double gnorm = gravity.norm();
        LOG(INFO) << "dynamic init: converged=" << converged << " |g|=" << gnorm
                  << " planes=" << lidar_factor.pcrs.size() << " min_normal_eig=" << min_normal_eig
                  << " |v_end|=" << states.back().v.norm();
        if (!converged || min_normal_eig < 15.0) return LaserMapping::DynamicInitResult::FAIL;
        return LaserMapping::DynamicInitResult::SUCCESS;
    }

    // scan i in its scan-end body frame, motion-compensated with the IMU (Voxel-SLAM motion_blur)
    std::vector<Vec3> DeskewScan(int i) const {
        const State& state = states[i];
        const auto& pts = raw_scans_points[i];
        std::int64_t scan_b = i == 0 ? pts.front().first : scan_end_times[i - 1];

        struct ImuPredict {
            std::int64_t time_ns;
            Mat3 R;
            Vec3 p, v, w, a_w;
        };

        // find all the imu in the interval
        std::vector<std::tuple<std::int64_t, Vec3, Vec3, double>> interval_imus;
        ForEachImuInterval(imus_, scan_b, state.time_ns, [&](const Imu& imu, std::int64_t head, double dt) {
            interval_imus.emplace_back(head, imu.angular_velocity, imu.linear_acceleration, dt);
        });
        // integrate backwards from the scan-end state, keeping the pose at each segment start
        std::vector<ImuPredict> predicts(interval_imus.size());
        Mat3 R = state.R;
        Vec3 p = state.p, v = state.v;
        for (int k = (int)interval_imus.size() - 1; k >= 0; k--) {
            const auto& [head, gyro, acc, dt] = interval_imus[k];
            Vec3 w = gyro - state.bg;
            Mat3 R_b = R * ExpMat(Vec3(-w * dt));
            Vec3 a_w = R_b * (acc - state.ba) + gravity;
            Vec3 v_b = v - a_w * dt;
            Vec3 p_b = p - v_b * dt - 0.5 * a_w * dt * dt;
            predicts[k] = ImuPredict{head, R_b, p_b, v_b, w, a_w};
            R = R_b;
            v = v_b;
            p = p_b;
        }

        std::vector<Vec3> deskewed;
        deskewed.reserve(pts.size());
        size_t k = 0;
        for (const auto& [t, q] : pts) {
            if (predicts.empty()) {
                deskewed.push_back(q);
                continue;
            }
            while (k + 1 < predicts.size() && predicts[k + 1].time_ns <= t) ++k;
            const ImuPredict& n = predicts[k];
            double tau = NS_TO_S(t - n.time_ns);
            Mat3 R_pt = n.R * ExpMat(Vec3(n.w * tau));
            Vec3 p_pt = n.p + n.v * tau + 0.5 * n.a_w * tau * tau;
            deskewed.push_back(state.R.transpose() * (R_pt * q + p_pt - state.p));
        }
        return deskewed;
    }

    // IMU-propagated pose at t from the latest state at or before t
    void PoseAt(std::int64_t t_ns, Mat3& R, Vec3& p) const {
        int k = 0;
        while (k + 1 < N && states[k + 1].time_ns <= t_ns) ++k;
        const State& s = states[k];
        R = s.R;
        p = s.p;
        Vec3 v = s.v;
        ForEachImuInterval(imus_, s.time_ns, t_ns, [&](const Imu& imu, std::int64_t, double dt) {
            Vec3 a_w = R * (imu.linear_acceleration - s.ba) + gravity;
            p += v * dt + 0.5 * a_w * dt * dt;
            v += a_w * dt;
            R = R * ExpMat(Vec3((imu.angular_velocity - s.bg) * dt));
        });
    }

   private:
    // Step 1 of the yaw-free TDCP scheme: per epoch pair, solve the ENU displacement and the receiver
    // clock steps from the single-satellite rows by weighted least squares.  Step 2 turns the solution
    // into the two rotation-about-the-vertical invariants, which is what the factor constrains.
    void BuildTdcpFactors() {
        tdcp_factors_.clear();
        if (tdcp_rows_.empty()) return;

        std::map<std::pair<int, int>, std::vector<const TdcpRow*>> groups;
        for (const TdcpRow& row : tdcp_rows_) {
            if (row.anchor_idx < 0 || row.cur_idx >= N || row.cur_idx <= row.anchor_idx) continue;
            groups[{row.anchor_idx, row.cur_idx}].push_back(&row);
        }

        int num_pairs = 0;
        double sum_std = 0.0;
        for (const auto& group : groups) {
            const std::vector<const TdcpRow*>& all = group.second;
            // a satellite alone in its constellation is absorbed exactly by that system's clock step
            int per_sys[4] = {0, 0, 0, 0};
            for (const TdcpRow* row : all) per_sys[row->sys_idx]++;
            int clk_col[4] = {-1, -1, -1, -1};
            int cols = 3;
            std::vector<const TdcpRow*> rows;
            for (const TdcpRow* row : all)
                if (per_sys[row->sys_idx] >= 2) {
                    if (clk_col[row->sys_idx] < 0) clk_col[row->sys_idx] = cols++;
                    rows.push_back(row);
                }
            // cols + 1 rows is the minimum that leaves a covariance; tdcp_min_sat is the same
            // sanity floor the RESPLE_LIO update uses
            if ((int)rows.size() < std::max(cols + 1, param->tdcp_min_sat)) continue;

            // weighted least squares with one pass of residual rejection
            std::vector<bool> inlier(rows.size(), true);
            VecX theta;
            MatX cov;
            bool ok = false;
            for (int pass = 0; pass < 2; pass++) {
                MatX HtWH = MatX::Zero(cols, cols);
                VecX HtWz = VecX::Zero(cols);
                int used = 0;
                for (size_t k = 0; k < rows.size(); k++) {
                    if (!inlier[k]) continue;
                    VecX h = VecX::Zero(cols);
                    h.head<3>() = rows[k]->u_enu;
                    h(clk_col[rows[k]->sys_idx]) = 1.0;
                    const double w = 1.0 / rows[k]->var;
                    HtWH += w * h * h.transpose();
                    HtWz += w * h * rows[k]->z;
                    used++;
                }
                if (used < cols + 1) break;
                Eigen::LDLT<MatX> ldlt(HtWH);
                if (ldlt.info() != Eigen::Success) break;
                cov = ldlt.solve(MatX::Identity(cols, cols));
                theta = cov * HtWz;
                if (!theta.allFinite() || !cov.allFinite()) break;
                ok = true;
                if (pass == 0) {
                    for (size_t k = 0; k < rows.size(); k++) {
                        VecX h = VecX::Zero(cols);
                        h.head<3>() = rows[k]->u_enu;
                        h(clk_col[rows[k]->sys_idx]) = 1.0;
                        const double r = rows[k]->z - h.dot(theta);
                        if (std::abs(r) > param->tdcp_outlier || r * r / rows[k]->var > param->tdcp_chi2)
                            inlier[k] = false;
                    }
                }
            }
            if (!ok) continue;

            const Mat3 cov_dp = cov.topLeftCorner<3, 3>();
            const double max_std = std::sqrt(cov_dp.diagonal().maxCoeff());
            if (!(max_std < param->tdcp_init_max_std)) continue;  // degenerate satellite geometry

            // the invariants and their first-order covariance
            const Vec3 dp = theta.head<3>();
            const double up = dp.z();
            const double horiz = dp.head<2>().norm();
            Eigen::Matrix<double, 2, 3> Jg = Eigen::Matrix<double, 2, 3>::Zero();
            Jg(0, 2) = 1.0;
            const bool use_horiz = horiz > param->tdcp_init_min_horiz;
            if (use_horiz) {
                Jg(1, 0) = dp.x() / horiz;
                Jg(1, 1) = dp.y() / horiz;
            }
            const Eigen::Matrix2d cov_y = Jg * cov_dp * Jg.transpose();

            TdcpFactor factor;
            factor.i = group.first.first;
            factor.j = group.first.second;
            factor.y << up, horiz;
            if (use_horiz) {
                factor.info = cov_y.inverse();
            } else {
                // with no horizontal direction the second invariant is dropped, which leaves the
                // marginal of the first
                factor.info.setZero();
                factor.info(0, 0) = 1.0 / std::max(cov_y(0, 0), 1e-12);
            }
            if (!factor.info.allFinite()) continue;
            factor.num_sat = (int)rows.size();
            tdcp_factors_.push_back(factor);
            num_pairs++;
            sum_std += max_std;
        }
        LOG(INFO) << "dynamic init: " << tdcp_factors_.size() << "/" << groups.size()
                  << " TDCP epoch pairs from " << tdcp_rows_.size() << " rows, mean WLS std "
                  << (num_pairs ? sum_std / num_pairs : 0.0) << " m";
    }

    // r = [up . dp - y_up ; |dp_horiz| - y_horiz] with dp the antenna displacement between the two
    // states and "up" the current gravity direction.  J columns: [R_i p_i | R_j p_j | gravity(2)].
    void EvalTdcpFactor(const std::vector<State>& sts, const Vec3& grav, const TdcpFactor& f,
                        Eigen::Vector2d& r, const Eigen::Matrix<double, 3, 2>* B,
                        Eigen::Matrix<double, 2, 14>* J) const {
        const Vec3& lever = param->level_arm;
        const Mat3& Ri = sts[f.i].R;
        const Mat3& Rj = sts[f.j].R;
        const Vec3 dp = (sts[f.j].p + Rj * lever) - (sts[f.i].p + Ri * lever);
        const Vec3 up = -grav.normalized();
        const double up_comp = up.dot(dp);
        const Vec3 horiz = dp - up_comp * up;
        const double horiz_norm = horiz.norm();
        r << up_comp - f.y(0), horiz_norm - f.y(1);
        if (!J) return;

        Eigen::Matrix<double, 2, 3> Jd = Eigen::Matrix<double, 2, 3>::Zero();
        Jd.row(0) = up.transpose();
        Vec3 horiz_hat = Vec3::Zero();
        if (horiz_norm > 1e-4) {
            horiz_hat = horiz / horiz_norm;
            Jd.row(1) = horiz_hat.transpose();  // horiz_hat is already orthogonal to up
        }
        J->setZero();
        J->block<2, 3>(0, 0) = Jd * (Ri * Hat(lever));
        J->block<2, 3>(0, 3) = -Jd;
        J->block<2, 3>(0, 6) = Jd * (-Rj * Hat(lever));
        J->block<2, 3>(0, 9) = Jd;
        if (B) {
            // g <- Exp(B dw) g leaves the norm alone, so d(up)/d(dw) = -[up]x B
            const Eigen::Matrix<double, 3, 2> dup_dw = -Hat(up) * (*B);
            J->block<1, 2>(0, 12) = dp.transpose() * dup_dw;
            J->block<1, 2>(1, 12) = -up_comp * horiz_hat.transpose() * dup_dw;
        }
    }

    // rms of the two invariant residuals, for judging how much the GNSS and the LiDAR/IMU window
    // actually disagree before and after the joint solve
    void LogTdcpRms(const char* tag) const {
        if (tdcp_factors_.empty()) return;
        double sq_up = 0.0, sq_horiz = 0.0;
        Eigen::Vector2d r;
        for (const TdcpFactor& f : tdcp_factors_) {
            EvalTdcpFactor(states, gravity, f, r, nullptr, nullptr);
            sq_up += r(0) * r(0);
            sq_horiz += r(1) * r(1);
        }
        const double num = (double)tdcp_factors_.size();
        LOG(INFO) << "dynamic init: TDCP " << tdcp_factors_.size() << " pairs " << tag << ", rms up "
                  << std::sqrt(sq_up / num) << " m, rms horizontal " << std::sqrt(sq_horiz / num) << " m";
    }

    double EvalTdcpCost(const std::vector<State>& sts, const Vec3& grav) const {
        double cost = 0.0;
        Eigen::Vector2d r;
        for (const TdcpFactor& f : tdcp_factors_) {
            EvalTdcpFactor(sts, grav, f, r, nullptr, nullptr);
            cost += r.dot(f.info * r);
        }
        return cost;
    }

    void SplitScans(const std::vector<Point>& points) {
        raw_scans_points.assign(N, {});
        size_t pi = 0;
        for (int i = 0; i < N; i++) {
            std::unordered_map<VOXEL_LOCATION, bool> seen_voxs;
            while (pi < points.size() && points[pi].GetTimeNs() <= scan_end_times[i]) {
                Vec3 p(points[pi].x, points[pi].y, points[pi].z);
                if (seen_voxs.emplace(VOXEL_LOCATION(p, scan_ds_size), true).second)
                    raw_scans_points[i].emplace_back(points[pi].GetTimeNs(), p);
                pi++;
            }
        }  // for frame
    }

    std::vector<std::vector<Vec3>> DeskewAll() const {
        std::vector<std::vector<Vec3>> deskew_scans(N);
        for (int i = 0; i < N; i++) deskew_scans[i] = DeskewScan(i);
        return deskew_scans;
    }

    // LiDAR-gyroscope odometry initialize value
    bool InitStateByLGO() {
        const double kCovGyr = 1e-4, kCovVel = 1.0, kCovBiasGyr = 1e-4, kCovLidar = 1e-3;
        const int kMaxIter = 20, kMaxUndistort = 7;
        gravity.setZero();
        Eigen::Matrix<double, 12, 12> P = Eigen::Matrix<double, 12, 12>::Identity();
        P.block<3, 3>(9, 9) *= 1e-5;
        Vec3 v_body = Vec3::Zero(), bg = Vec3::Zero();
        pcl::PointCloud<pcl::PointXYZ>::Ptr map(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> nn_idx(5);
        std::vector<float> nn_d2(5);

        for (int i = 0; i < N; i++) {
            const auto& scan_points = raw_scans_points[i];
            const std::int64_t t_b = i == 0 ? scan_points.front().first : scan_end_times[i - 1];
            const std::int64_t t_e = scan_end_times[i];
            State& state = states[i];

            // forward propagation with the gyroscope
            std::vector<std::tuple<std::int64_t, Mat3, Vec3>> predicts;
            Mat3 R_b_e = Mat3::Identity();
            ForEachImuInterval(imus_, t_b, t_e, [&](const Imu& imu, std::int64_t head, double dt) {
                predicts.emplace_back(head, R_b_e, imu.angular_velocity);
                if (i > 0) {
                    Eigen::Matrix<double, 12, 12> F = Eigen::Matrix<double, 12, 12>::Identity();
                    F.block<3, 3>(0, 0) = ExpMat(Vec3(-imu.angular_velocity * dt));
                    F.block<3, 3>(0, 9) = -Mat3::Identity() * dt;
                    F.block<3, 3>(3, 6) = states[i - 1].R * dt;
                    P = F * P * F.transpose();
                    P.block<3, 3>(0, 0).diagonal().array() += kCovGyr * dt * dt;
                    P.block<3, 3>(6, 6).diagonal().array() += kCovVel * dt * dt;
                    P.block<3, 3>(9, 9).diagonal().array() += kCovBiasGyr * dt * dt;
                }
                R_b_e = R_b_e * ExpMat(Vec3(imu.angular_velocity * dt));
            });

            // rotational distortion compensation into the scan-end frame
            std::vector<Vec3> desk_points;
            desk_points.reserve(scan_points.size());
            size_t k = 0;
            for (const auto& [point_time_ns, point] : scan_points) {
                if (predicts.empty()) {
                    desk_points.push_back(point);
                    continue;
                }
                while (k + 1 < predicts.size() && std::get<0>(predicts[k + 1]) <= point_time_ns) ++k;
                const auto& [scan_time_ns, R_b_k, w] = predicts[k];
                Vec3 p_deskew =
                    R_b_e.transpose() * R_b_k * ExpMat(Vec3(w * NS_TO_S(point_time_ns - scan_time_ns))) * point;
                desk_points.push_back(p_deskew);
            }
            std::vector<Vec3> q_end = desk_points;

            // iterative update
            if (i > 0) {
                const State& s_prev = states[i - 1];
                state.R = s_prev.R * R_b_e;
                state.p = s_prev.p + state.R * v_body * NS_TO_S(t_e - s_prev.time_ns);
                const Mat3 R_prop = state.R;
                const Vec3 t_prop = state.p, v_prop = v_body, bg_prop = bg;
                const size_t stride = std::max<size_t>(1, scan_points.size() / 2000);
                kdtree.setInputCloud(map);

                Eigen::Matrix<double, 12, 12> KH = Eigen::Matrix<double, 12, 12>::Zero();
                State last_converge = state;
                bool undistort = true;
                int n_undistort = 0;
                for (int iter = 0; iter < kMaxIter; iter++) {
                    // translational distortion compensation by linear interpolation (Eq. 8-11)
                    if (undistort) {
                        for (size_t j = 0; j < scan_points.size(); j++) {
                            double scale = std::clamp(double(scan_points[j].first - t_b) / double(t_e - t_b), 0.0, 1.0);
                            q_end[j] = desk_points[j] + (scale - 1.0) * state.R.transpose() * (state.p - s_prev.p);
                        }
                        undistort = false;
                        ++n_undistort;
                    }

                    // point-to-plane residuals (Eq. 3)
                    Eigen::Matrix<double, 12, 12> HtH = Eigen::Matrix<double, 12, 12>::Zero();
                    Eigen::Matrix<double, 12, 1> Htz = Eigen::Matrix<double, 12, 1>::Zero();
                    int n_eff = 0;
                    for (size_t j = 0; j < scan_points.size(); j += stride) {
                        const Vec3& q = q_end[j];
                        Vec3 pw = state.R * q + state.p;
                        if (kdtree.nearestKSearch(pcl::PointXYZ(pw.x(), pw.y(), pw.z()), 5, nn_idx, nn_d2) < 5 ||
                            nn_d2[4] > 5.0)
                            continue;
                        Eigen::Matrix<double, 5, 3> A;
                        for (int m = 0; m < 5; m++)
                            A.row(m) = (*map)[nn_idx[m]].getVector3fMap().cast<double>().transpose();
                        Vec3 n = A.colPivHouseholderQr().solve(-Eigen::Matrix<double, 5, 1>::Ones());
                        double d = 1.0 / n.norm();
                        n *= d;
                        if (((A * n).array() + d).abs().maxCoeff() > plane_thresh) continue;
                        double pd2 = n.dot(pw) + d;
                        if (1.0 - 0.9 * std::abs(pd2) / std::sqrt(q.norm()) <= 0.9) continue;
                        Eigen::Matrix<double, 1, 12> h = Eigen::Matrix<double, 1, 12>::Zero();
                        h.head<3>() = -n.transpose() * state.R * Hat(q);
                        h.segment<3>(3) = n.transpose();
                        HtH += h.transpose() * h;
                        Htz -= h.transpose() * pd2;
                        ++n_eff;
                    }
                    if (n_eff < 100) {
                        LOG(WARNING) << "dynamic init: LGO scan " << i << " has only " << n_eff << " planes";
                        return false;
                    }

                    // iterated Kalman update: x += K z + (x_prop - x) - K H (x_prop - x)
                    Eigen::Matrix<double, 12, 1> dx_prop;
                    dx_prop << LogMat(Mat3(state.R.transpose() * R_prop)), t_prop - state.p, v_prop - v_body,
                        bg_prop - bg;
                    Eigen::Matrix<double, 12, 12> M = (HtH / kCovLidar + P.inverse()).inverse();
                    KH = M * HtH / kCovLidar;
                    Eigen::Matrix<double, 12, 1> sol = M * Htz / kCovLidar + dx_prop - KH * dx_prop;
                    state.R = state.R * ExpMat(Vec3(sol.head<3>()));
                    state.p += sol.segment<3>(3);
                    v_body += sol.segment<3>(6);
                    bg += sol.segment<3>(9);

                    // on convergence, re-deskew once the converged pose has settled
                    if (sol.head<3>().norm() * 57.3 < 0.01 && sol.segment<3>(3).norm() * 100 < 0.015) {
                        bool settled = LogMat(Mat3(last_converge.R.transpose() * state.R)).norm() * 57.3 < 1.0 &&
                                       (state.p - last_converge.p).norm() * 100 < 1.0 && iter >= 4;
                        last_converge = state;
                        if (settled) {
                            if (n_undistort >= kMaxUndistort) break;
                            undistort = true;
                        }
                    }
                }  // iterative update
                P = (Eigen::Matrix<double, 12, 12>::Identity() - KH) * P;
                state.v = state.R * v_body;
            }

            for (const Vec3& q : q_end) {
                Vec3 pw = state.R * q + state.p;
                map->push_back(pcl::PointXYZ(pw.x(), pw.y(), pw.z()));
            }
        }  // for frame

        // gyroscope bias from the LGO relative rotations (Eq. 15-16)
        BuildPreintegrations();
        Mat3 A = Mat3::Zero();
        Vec3 b = Vec3::Zero();
        for (int i = 0; i < N - 1; i++) {
            const Mat3& J = preints[i].dR_dbg;
            Vec3 r = LogMat(Mat3(preints[i].dR.transpose() * states[i].R.transpose() * states[i + 1].R));
            A += J.transpose() * J;
            b += J.transpose() * r;
        }
        Vec3 bg_init = A.ldlt().solve(b);
        for (State& st : states) st.bg = bg_init;
        LOG(INFO) << "dynamic init: LGO gyro bias " << bg_init.transpose();
        return true;
    }

    // Linear velocity/gravity from the registered poses with zero bias (LI-Init style):
    //   v_i * dt + 0.5 * g * dt^2 = p_j - p_i - R_i dp,
    //   v_j - v_i - g * dt = R_i dv
    bool InitVelocityGravity() {
        const int rows = 6 * (N - 1);
        const int cols = 3 * N + 3;
        MatX A = MatX::Zero(rows, cols);
        VecX b = VecX::Zero(rows);
        for (int i = 0; i < N - 1; i++) {
            const ImuPreintegration& pre = preints[i];
            const double T = pre.dt_sum;
            if (T <= 0.0) return false;
            const State& si = states[i];
            const State& sj = states[i + 1];
            A.block(6 * i, 3 * i, 3, 3) = Mat3::Identity();
            A.block(6 * i, 3 * N, 3, 3) = 0.5 * T * Mat3::Identity();
            b.segment(6 * i, 3) = (sj.p - si.p - si.R * pre.dp) / T;
            A.block(6 * i + 3, 3 * i, 3, 3) = -Mat3::Identity();
            A.block(6 * i + 3, 3 * (i + 1), 3, 3) = Mat3::Identity();
            A.block(6 * i + 3, 3 * N, 3, 3) = -T * Mat3::Identity();
            b.segment(6 * i + 3, 3) = si.R * pre.dv;
        }
        VecX x = A.colPivHouseholderQr().solve(b);
        Vec3 g = x.tail<3>();

        LOG(INFO) << "dynamic init gravity: " << g.norm() << " vel: " << x.head(3).transpose();
        if (g.norm() < 7.0 || g.norm() > 12.5) return false;

        const double g0 = RESPLE_LIO::GRAVITY_NORM;
        const MatX Av = A.leftCols(3 * N);  // velocity columns, unchanged by the re-parameterisation
        const MatX C = A.rightCols(3);      // the columns that multiplied g
        Vec3 g_bar = g.normalized();
        VecX v = x.head(3 * N);
        for (int iter = 0; iter < 4; iter++) {
            const Eigen::Matrix<double, 3, 2> B = GravityTangentBasis(g_bar);

            MatX Aw(rows, 3 * N + 2);
            Aw << Av, C * B;
            const VecX xw = Aw.colPivHouseholderQr().solve(VecX(b - C * (g0 * g_bar)));
            const Eigen::Vector2d w = xw.tail<2>();
            v = xw.head(3 * N);
            g_bar = (g0 * g_bar + B * w).normalized();
            if (w.norm() < 1e-3) break;
        }
        gravity = g0 * g_bar;
        LOG(INFO) << "dynamic init gravity refined: " << gravity.transpose() << std::endl;
        for (int i = 0; i < N; i++) states[i].v = v.segment<3>(3 * i);
        return true;
    }

    void BuildPreintegrations() {
        preints.assign(N - 1, ImuPreintegration());
        for (int i = 0; i < N - 1; i++) {
            preints[i].reset(states[i].bg, states[i].ba);
            ForEachImuInterval(imus_, scan_end_times[i], scan_end_times[i + 1],
                               [&](const Imu& imu, std::int64_t, double dt) {
                                   preints[i].integrate(imu.angular_velocity, imu.linear_acceleration, dt);
                               });
        }
    }

    // LI_BA_OptimizerGravity::only_residual
    double EvalCost(const std::vector<State>& sts, const Vec3& grav) {
        double cost = 0.0;
        for (int i = 0; i < N - 1; i++)
            if (preints[i].dt_sum > 0) cost += preints[i].Evaluate(sts[i], sts[i + 1], grav, nullptr, nullptr, nullptr);
        return 0.5 * kInitImuCoef * cost + lidar_factor.eval_error(sts) +
               0.5 * param->tdcp_init_weight * EvalTdcpCost(sts, grav);
    }

    // LI_BA_OptimizerGravity::damping_iter, with gravity on S2 (fixed norm, 2-DoF tangent update
    // g <- Exp(B w) g, so dg/dw = -[g]x B)
    std::pair<double, double> Optimize(int max_inner_iters) {
        const int DIM = 15 * N + 2;
        double u = 1e-2;
        double cost0 = EvalCost(states, gravity);
        double cost = cost0;

        for (int it = 0; it < max_inner_iters; it++) {
            MatX H = MatX::Zero(DIM, DIM);
            VecX b = VecX::Zero(DIM);

            const Eigen::Matrix<double, 3, 2> B = GravityTangentBasis(gravity);
            const Eigen::Matrix<double, 3, 2> dg_dw = -Hat(gravity) * B;
            Eigen::Matrix<double, 15, 33> J;
            Eigen::Matrix<double, 15, 32> Jw;
            Eigen::Matrix<double, 15, 1> r;
            Eigen::Matrix<double, 15, 15> cov_inv;
            for (int i = 0; i < N - 1; i++) {
                const ImuPreintegration& pre = preints[i];
                if (pre.dt_sum <= 0) continue;
                pre.Evaluate(states[i], states[i + 1], gravity, &J, &r, &cov_inv);
                Jw.leftCols(30) = J.leftCols(30);
                Jw.rightCols(2) = J.rightCols(3) * dg_dw;
                Eigen::Matrix<double, 32, 32> JtJ = kInitImuCoef * Jw.transpose() * cov_inv * Jw;
                Eigen::Matrix<double, 32, 1> Jtr = kInitImuCoef * Jw.transpose() * cov_inv * r;
                int idx[32];
                for (int c = 0; c < 15; c++) idx[c] = 15 * i + c;
                for (int c = 0; c < 15; c++) idx[15 + c] = 15 * (i + 1) + c;
                for (int c = 0; c < 2; c++) idx[30 + c] = 15 * N + c;
                for (int a = 0; a < 32; a++) {
                    b(idx[a]) -= Jtr(a);
                    for (int c2 = 0; c2 < 32; c2++) H(idx[a], idx[c2]) += JtJ(a, c2);
                }
            }

            // TDCP factors; same normal-equation convention as the IMU block above
            {
                Eigen::Matrix<double, 2, 14> Jt;
                Eigen::Vector2d r;
                for (const TdcpFactor& f : tdcp_factors_) {
                    EvalTdcpFactor(states, gravity, f, r, &B, &Jt);
                    const Eigen::Matrix<double, 14, 14> JtJ =
                        param->tdcp_init_weight * Jt.transpose() * f.info * Jt;
                    const Eigen::Matrix<double, 14, 1> Jtr =
                        param->tdcp_init_weight * Jt.transpose() * f.info * r;
                    int idx[14];
                    for (int c = 0; c < 6; c++) idx[c] = 15 * f.i + c;
                    for (int c = 0; c < 6; c++) idx[6 + c] = 15 * f.j + c;
                    for (int c = 0; c < 2; c++) idx[12 + c] = 15 * N + c;
                    for (int a = 0; a < 14; a++) {
                        b(idx[a]) -= Jtr(a);
                        for (int c2 = 0; c2 < 14; c2++) H(idx[a], idx[c2]) += JtJ(a, c2);
                    }
                }
            }

            lidar_factor.eval_error(states);
            lidar_factor.eval_H(states, H, b, 15);

            H.block(0, 0, 6, DIM).setZero();
            H.block(0, 0, DIM, 6).setZero();
            H.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity();
            b.segment(0, 6).setZero();

            MatX D = MatX::Identity(DIM, DIM);
            for (int k = 0; k < DIM; k++) D(k, k) = std::max(H(k, k), 1e-6);

            bool accepted = false;
            for (int trial = 0; trial < 10 && !accepted; trial++) {
                VecX dx = (H + u * D).ldlt().solve(b);

                std::vector<State> trial_states = states;
                Vec3 trial_gravity = gravity;
                for (int i = 0; i < N; i++) {
                    trial_states[i].R = trial_states[i].R * ExpMat(Vec3(dx.segment<3>(15 * i + 0)));
                    trial_states[i].p += dx.segment<3>(15 * i + 3);
                    trial_states[i].v += dx.segment<3>(15 * i + 6);
                    trial_states[i].bg += dx.segment<3>(15 * i + 9);
                    trial_states[i].ba += dx.segment<3>(15 * i + 12);
                }
                trial_gravity = ExpMat(Vec3(B * dx.segment<2>(15 * N))) * gravity;

                double new_cost = EvalCost(trial_states, trial_gravity);
                if (new_cost < cost) {
                    states = trial_states;
                    gravity = trial_gravity;
                    cost = new_cost;
                    u = std::max(u / 3.0, 1e-7);
                    accepted = true;
                } else {
                    u *= 3.0;
                }
            }
            if (!accepted) break;
        }
        return {cost0, cost};
    }

    // orthonormal basis of the tangent plane of g (Gram-Schmidt against a non-parallel axis);
    // unlike S2::B() it has no singularity at g = -z, which is exactly where gravity sits here
    static Eigen::Matrix<double, 3, 2> GravityTangentBasis(const Vec3& g) {
        const Vec3 g_bar = g.normalized();
        const Vec3 axis = std::fabs(g_bar.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY();
        const Vec3 b1 = (axis - g_bar * g_bar.dot(axis)).normalized();
        Eigen::Matrix<double, 3, 2> B;
        B << b1, Vec3(g_bar.cross(b1));
        return B;
    }

    // Initialization::align_gravity, but always onto -z so the world is z-up like the stationary init
    void AlignGravity() {
        Mat3 Rg = Eigen::Quaterniond::FromTwoVectors(gravity, Vec3(0, 0, -1)).toRotationMatrix();
        Vec3 p0 = states[0].p;
        for (auto& st : states) {
            st.p = Rg * (st.p - p0) + p0;
            st.R = Rg * st.R;
            st.v = Rg * st.v;
        }
        gravity = Rg * gravity;
    }
};
}  // namespace

LaserMapping::LaserMapping() = default;
void LaserMapping::AddScanToBuffer(const PointCloud::Ptr& scan) {
    if (scan->empty()) {
        LOG(INFO) << "empty in scan, no points pushed into buffer";
        return;
    }
    std::sort(scan->points.begin(), scan->points.end(),
              [](const Point& a, const Point& b) { return a.GetTimeNs() < b.GetTimeNs(); });
    int skip_scan_points = 0;
    for (int i = 0; i < scan->size(); ++i) {
        if (points_buffer_.empty()) {
            points_buffer_.push_back(scan->at(i));
            continue;
        }
        if (points_buffer_.back().GetTimeNs() <= scan->at(i).GetTimeNs()) {
            points_buffer_.push_back(scan->at(i));
        } else {
            skip_scan_points++;
        }
    }

    if (skip_scan_points != 0)
        LOG(INFO) << "skip " << skip_scan_points << " scan points for timestamp of point error, " << scan->size()
                  << " points total";
}
void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan(new PointCloud());
    switch (LidarTypeFromString(param->lidar_type)) {
        case LidarType::OUSTER:
            scan = preprocess_->OusterHandler(msg, S_TO_NS(timestamp));
            break;
        case LidarType::HESAI:
            scan = preprocess_->HesaiHandler(msg, S_TO_NS(timestamp));
            break;
        case LidarType::VELODYNE_POINTCLOUD2:
            scan = preprocess_->VelodynePointsHandler(msg, S_TO_NS(timestamp));
            break;
        default:
            throw std::logic_error("unknown lidar type");
    }
    AddScanToBuffer(scan);
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) {
        return;
    }
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan;
    scan = preprocess_->LivoxHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
}
void LaserMapping::ImageCallBack(const sensor_msgs::Image::ConstPtr& msg) {
    const double timestamp = msg->header.stamp.toSec();
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    RgbImage image;
    if (!image.FromRaw(msg->data.data(), msg->width, msg->height, msg->step, msg->encoding)) {
        LOG(WARNING) << "image skipped, encoding " << msg->encoding;
        return;
    }
    image.time_ns = S_TO_NS(timestamp - global_offset_time);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    image_buff_.push_back(std::move(image));
}

void LaserMapping::CompressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    const double timestamp = msg->header.stamp.toSec();
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    RgbImage image;
    if (!image.Decode(msg->data.data(), msg->data.size())) {
        LOG(WARNING) << "image skipped, format " << msg->format;
        return;
    }
    image.time_ns = S_TO_NS(timestamp - global_offset_time);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    image_buff_.push_back(std::move(image));
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr& msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    double timestamp = msg_in->header.stamp.toSec();
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    Imu imu;
    imu.time_ns = S_TO_NS(timestamp);
    imu.angular_velocity.x() = msg_in->angular_velocity.x;
    imu.angular_velocity.y() = msg_in->angular_velocity.y;
    imu.angular_velocity.z() = msg_in->angular_velocity.z;
    imu.linear_acceleration.x() = msg_in->linear_acceleration.x;
    imu.linear_acceleration.y() = msg_in->linear_acceleration.y;
    imu.linear_acceleration.z() = msg_in->linear_acceleration.z;
    if (param->acc_ratio_) imu.linear_acceleration *= RESPLE_LIO::GRAVITY_NORM;
    imu_buffer_.emplace_back(imu);
}

void LaserMapping::GnssSolutionCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    gtime_t gpst{};
    gpst.time = msg->header.stamp.sec;
    gpst.sec = NS_TO_S(msg->header.stamp.nsec);

    sol_t sol{};
    sol.time = gpst;
    const double pos[3] = {msg->latitude * D2R, msg->longitude * D2R, msg->altitude};
    pos2ecef(pos, sol.rr);
    const uint8_t fix_type = static_cast<uint8_t>(msg->status.status);
    const uint8_t carr_soln = static_cast<uint8_t>(msg->status.service);
    if (fix_type == 0 || fix_type == 5) {
        sol.stat = SOLQ_NONE;
    } else if (carr_soln == 2) {
        sol.stat = SOLQ_FIX;
    } else if (carr_soln == 1) {
        sol.stat = SOLQ_FLOAT;
    } else {
        sol.stat = SOLQ_SINGLE;
    }
    sol.type = 0;
}
bool LaserMapping::ReadObsNav(const std::string& obs_fname, const std::string& nav_fname, const std::string& sol_fname) {
    gtime_t ts{}, te{};  // zero epochs disable the time window filter

    // rcv index 1 for the observation file; the navigation file carries no receiver data.
    if (readrnxt(obs_fname.c_str(), 1, ts, te, 0.0, "", &obss, &nav, nullptr) <= 0) {
        LOG(ERROR) << "failed to read observation file: " << obs_fname;
        return false;
    }
    if (readrnxt(nav_fname.c_str(), 0, ts, te, 0.0, "", nullptr, &nav, nullptr) <= 0) {
        LOG(ERROR) << "failed to read navigation file: " << nav_fname;
        return false;
    }

    // RTKLIB expects observations sorted by time/sat and ephemerides uniquified before use.
    sortobs(&obss);
    uniqnav(&nav);
    if (resple_lio_) {
        resple_lio_->obss = obss;
        resple_lio_->nav = nav;
    }

    LOG(INFO) << "read " << obss.n << " observations, " << nav.n << " eph, " << nav.ng << " geph, " << nav.ns
              << " seph";
    if (obss.n <= 0 || nav.n + nav.ng <= 0) {
        LOG(ERROR) << "no usable GNSS data; check that RTKLIB is built with -DENAGLO/-DENAGAL/-DENACMP";
        return false;
    }
    CHECK(!std::isnan(global_offset_time));
    for (int i = 0; i < obss.n;) {
        obss_id_buffer_.push_back(i);
        gtime_t utc = gpst2utc(obss.data[i].time);
        double utc_d = double(utc.time) + utc.sec;
        obss_time_buffer_.push_back(S_TO_NS(utc_d - global_offset_time));
        int j = i;
        while (std::abs(timediff(obss.data[j].time, obss.data[i].time)) < DTTOL && j < obss.n) ++j;
        i = j;
    }
    std::vector<std::string> files{obs_fname, nav_fname};
    std::vector<const char*> infile;
    for (const auto& f : files) infile.push_back(f.c_str());

    // position
    filopt_t filopt{};
    const int ret = postpos(ts, te, 0.0, 0.0, &param->prcopt, &param->solopt, &filopt, infile.data(),
                            static_cast<int>(infile.size()), sol_fname.c_str(), "", "");
    if (ret != 0) {
        LOG(ERROR) << "postpos failed with status " << ret;
        return false;
    }
    return true;
}

bool LaserMapping::Init(const std::string& config_fname, const std::string& camera_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    // Eigen would otherwise split every large product over all cores (the build has -fopenmp): a thread-team
    // fork/join per filter iteration, which costs far more than the product itself at a few hundred rows
    Eigen::setNbThreads(1);
    param = std::make_shared<LaserMappingParam>();
    param->LoadFromYaml(config_fname);
    if (!camera_fname.empty()) param->LoadCameraFromYaml(camera_fname);
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->param = param;

    // the estimation algorithm
    if (param->front_end == "fastlio") {
        fast_lio_ = std::make_shared<FastLio>();
        fast_lio_->param = param;
        fast_lio_->Init();
    } else {
        resple_lio_ = std::make_shared<RESPLE_LIO>();
        resple_lio_->param = param;
        resple_lio_->Init();
    }

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}

void LaserMapping::InitialSubscribers(ros::NodeHandle& nh) {
    LidarType lidar_type = LidarTypeFromString(param->lidar_type);
    if (lidar_type == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000,
            [this](const livox_ros_driver::CustomMsg::ConstPtr& msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000,
            [this](const sensor_msgs::PointCloud2::ConstPtr& msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr& msg) { IMUCallBack(msg); });
    const std::string& img = param->img_topic_;
    if (img.size() >= 11 && img.compare(img.size() - 11, 11, "/compressed") == 0) {
        sub_img_ = nh.subscribe<sensor_msgs::CompressedImage>(
            img, 100, [this](const sensor_msgs::CompressedImage::ConstPtr& msg) { CompressedImageCallback(msg); });
    } else if (!img.empty()) {
        sub_img_ = nh.subscribe<sensor_msgs::Image>(
            img, 100, [this](const sensor_msgs::Image::ConstPtr& msg) { ImageCallBack(msg); });
    }
}
void LaserMapping::InitialPublishers(ros::NodeHandle& nh) {
    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
}

LaserMapping::DynamicInitResult LaserMapping::TryDynamicInit(std::int64_t window_begin_ns, std::int64_t window_end_ns) {
    int N = (window_end_ns - window_begin_ns) / S_TO_NS(0.1);

    // raw lidar points
    std::vector<faster_lio::Point> window_points;
    window_points.reserve(points_buffer_.size());
    for (const auto& pt : points_buffer_) {
        if (pt.GetTimeNs() < window_begin_ns) continue;
        if (pt.GetTimeNs() >= window_end_ns) break;
        Vec3 p_body = param->extrin_il_ * Vec3(pt.x, pt.y, pt.z);
        faster_lio::Point p = pt;
        p.x = static_cast<float>(p_body.x());
        p.y = static_cast<float>(p_body.y());
        p.z = static_cast<float>(p_body.z());
        window_points.push_back(p);
    }

    // imu
    std::vector<Imu> window_imus;
    window_imus.reserve(imu_buffer_.size());
    for (const auto& imu : imu_buffer_) {
        if (imu.time_ns < window_begin_ns) continue;
        if (imu.time_ns >= window_end_ns) break;
        window_imus.push_back(imu);
    }

    // GNSS
    std::vector<int> obss_ids;
    std::vector<std::int64_t> obss_time_ns;
    for (int i = 0; i < obss_id_buffer_.size(); ++i) {
        int obs_id = obss_id_buffer_[i];
        std::int64_t obs_time_ns = obss_time_buffer_[i];
        if (obs_time_ns < window_begin_ns) continue;
        if (obs_time_ns >= window_end_ns) break;
        obss_ids.push_back(obs_id);
        obss_time_ns.push_back(obs_time_ns);
    }

    VoxelInitializer dynamic_initializer(window_begin_ns, window_end_ns, param->resple_.ds_scan_voxel, param->resple_.plane_thresh);
    dynamic_initializer.param = param;
    dynamic_initializer.obss = obss;
    dynamic_initializer.nav = nav;

    const LaserMapping::DynamicInitResult result =
        dynamic_initializer.Run(window_points, window_imus, obss_ids, obss_time_ns);
    if (result != DynamicInitResult::SUCCESS) return result;

    const State& s0 = dynamic_initializer.states.front();
    const State& s_last = dynamic_initializer.states.back();
    resple_lio_->bg = s_last.bg;
    resple_lio_->ba = s_last.ba;
    // VoxelInitializer's `a_imu = R*(acc-ba) + gravity` uses the opposite sign convention from
    // resple's `a_w = a_w_no_g + gravity` (see RESPLE_LIO::prepIMU), hence the negation.
    resple_lio_->gravity = -dynamic_initializer.gravity;

    // On a uniform cubic B-spline the value at t_{j+1} = start + (j+1)*dt is (k_j + 4 k_{j+1} + k_{j+2})/6,
    // i.e. knot j shapes the trajectory around t_{j+1}. Seed each knot with the IMU-propagated pose at
    // that time; starting the spline one dt before scan 0 makes knot 0 scan 0's gauge-fixed pose.
    const std::int64_t dt_ns = param->resple_.dt_ns;
    resple_lio_->spl.init(dt_ns, s0.time_ns - dt_ns, Eigen::Quaterniond(s0.R));
    Mat3 R_prev = s0.R;
    for (std::int64_t t_k = s0.time_ns; t_k <= s_last.time_ns; t_k += dt_ns) {
        Mat3 R_k;
        Vec3 p_k;
        dynamic_initializer.PoseAt(t_k, R_k, p_k);
        resple_lio_->spl.addOneStateKnot(p_k, 0.5 * LogMat(Mat3(R_prev.transpose() * R_k)));
        R_prev = R_k;
    }

    // local map from the motion-compensated window scans
    std::vector<Vec3> init_points_world;
    for (int i = 0; i < N; ++i) {
        for (const Vec3& q : dynamic_initializer.DeskewScan(i)) {
            init_points_world.push_back(dynamic_initializer.states[i].R * q + dynamic_initializer.states[i].p);
        }
    }
    resple_lio_->lba->InitMap(init_points_world, param->lba_.range_err * param->lba_.range_err + param->lba_.pose_var);

    std::int64_t consumed_until = resple_lio_->spl.maxTimeNs();
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < consumed_until) {
        points_buffer_.pop_front();
    }
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < consumed_until) {
        imu_buffer_.pop_front();
    }
    while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < consumed_until) {
        obss_time_buffer_.pop_front();
        obss_id_buffer_.pop_front();
    }

    if_inertial_initialized = true;
    if_localmap_initialized = true;
    LOG(INFO) << "dynamic init succeeded at t=" << std::fixed << std::setprecision(3)
              << global_offset_time + NS_TO_S(s0.time_ns) << std::defaultfloat
              << ": gravity=" << resple_lio_->gravity.transpose() << " bg=" << resple_lio_->bg.transpose()
              << " ba=" << resple_lio_->ba.transpose() << " v_end=" << s_last.v.transpose()
              << " map points=" << init_points_world.size();
    return DynamicInitResult::SUCCESS;
}

void LaserMapping::RunFastLio() {
    static constexpr int INIT_IMU_COUNT = 20;
    static constexpr int64_t INIT_POINTS_SPAN = S_TO_NS(0.3);
    if (!if_localmap_initialized) {
        if (points_buffer_.empty() || imu_buffer_.empty()) return;
        if (static_cast<int>(imu_buffer_.size()) < INIT_IMU_COUNT) return;

        const std::int64_t end_time_ns = points_buffer_.front().GetTimeNs() + INIT_POINTS_SPAN;
        if (points_buffer_.back().GetTimeNs() < end_time_ns || imu_buffer_.back().time_ns < end_time_ns) return;

        // standing still: gravity and the gyroscope bias come from the first samples
        Vec3 acc_sum = Vec3::Zero();
        Vec3 gyr_sum = Vec3::Zero();
        for (int i = 0; i < INIT_IMU_COUNT; ++i) {
            acc_sum += imu_buffer_[i].linear_acceleration;
            gyr_sum += imu_buffer_[i].angular_velocity;
        }
        const Vec3 gravity_body = (acc_sum / INIT_IMU_COUNT).normalized() * FastLio::GRAVITY_NORM;
        Mat3 R0 = Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
        const double yaw = std::atan2(R0(1, 0), R0(0, 0));
        R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;

        FastLioState state;
        state.R = R0;
        state.bg = gyr_sum / INIT_IMU_COUNT;

        PointCloud::Ptr init_points(new PointCloud);
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < end_time_ns) {
            init_points->emplace_back(points_buffer_.front());
            points_buffer_.pop_front();
        }
        Imu last_imu = imu_buffer_.front();
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < end_time_ns) {
            last_imu = imu_buffer_.front();
            imu_buffer_.pop_front();
        }
        while (!image_buff_.empty() && image_buff_.front().time_ns < end_time_ns) {
            image_buff_.pop_front();
        }
        if (init_points->size() < 100) {
            LOG(WARNING) << "only " << init_points->size() << " points for map init, waiting";
            return;
        }
        PointCloud::Ptr init_points_world(new PointCloud);
        pcl::transformPointCloud(*init_points, *init_points_world,
                                 (Pose3(Eigen::Quaterniond(R0), Vec3::Zero()) * param->extrin_il_).Mat4d(), true);
        std::vector<Vec3> init_ps;
        for (const Point& pt : init_points_world->points) init_ps.push_back(pt.getVector3fMap().cast<double>());
        fast_lio_->lba->InitMap(init_ps, param->lba_.range_err * param->lba_.range_err + param->lba_.pose_var);
        fast_lio_->Start(state, R0 * gravity_body, end_time_ns, last_imu);
        if_localmap_initialized = true;
        LOG(INFO) << "localmap initialized with " << init_points_world->size() << " points";
        return;
    }

    while (CollectMeasuresFastLio()) {
        fast_lio_->ProcessMeasurement(measures_);
        PublishROSMsgFastLio();
        PostUpdateFastLio();
    }
}

void LaserMapping::RunRESPLE() {
    static constexpr int INIT_IMU_COUNT = 20;
    static constexpr int64_t INIT_POINTS_SPAN = S_TO_NS(0.3);
    static constexpr int64_t DYNAMIC_INIT_SPAN = S_TO_NS(1.0);
    // a failed window is dropped by this much before retrying (Voxel-SLAM system_reset)
    static constexpr int64_t DYNAMIC_INIT_RETRY_STEP = S_TO_NS(0.5);
    if (!if_localmap_initialized && dynamic_init_) {
        if (points_buffer_.empty() || imu_buffer_.empty()) return;
        std::int64_t first_point_time_ns = points_buffer_.front().GetTimeNs();
        std::int64_t first_imu_time_ns = imu_buffer_.front().time_ns;
        std::int64_t first_time_ns = std::max(first_point_time_ns, first_imu_time_ns);
        if (!obss_time_buffer_.empty()) {
            first_time_ns = std::max(first_time_ns, obss_time_buffer_.front());
        }
        // clear the data before the first time
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < first_time_ns) {
            points_buffer_.pop_front();
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < first_time_ns) {
            imu_buffer_.pop_front();
        }
        while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < first_time_ns) {
            obss_time_buffer_.pop_front();
            obss_id_buffer_.pop_front();
        }
        if (points_buffer_.empty() || imu_buffer_.empty()) {
            LOG(INFO) << "dynamic init buffer is empty";
            return;
        }
        std::int64_t end_time_ns = first_time_ns + DYNAMIC_INIT_SPAN;
        if (points_buffer_.back().GetTimeNs() < end_time_ns || imu_buffer_.back().time_ns < end_time_ns) {
            LOG(INFO) << "dynamic init buffer is not enough, wait";
            return;
        }
        switch (TryDynamicInit(first_time_ns, end_time_ns)) {
            case DynamicInitResult::SUCCESS:
                return;
            case DynamicInitResult::STATIONARY:
                // gravity is directly observable at rest, and the dynamic problem is degenerate there
                LOG(INFO) << "dynamic init window is stationary, using stationary init";
                break;
            case DynamicInitResult::FAIL: {
                LOG(WARNING) << "dynamic init failed, retrying " << NS_TO_S(DYNAMIC_INIT_RETRY_STEP) << " s later";
                const std::int64_t drop_until = first_time_ns + DYNAMIC_INIT_RETRY_STEP;
                while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < drop_until) {
                    points_buffer_.pop_front();
                }
                while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < drop_until) {
                    imu_buffer_.pop_front();
                }
                while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < drop_until) {
                    obss_time_buffer_.pop_front();
                    obss_id_buffer_.pop_front();
                }
                while (!image_buff_.empty() && image_buff_.front().time_ns < drop_until) {
                    image_buff_.pop_front();
                }
                return;
            }
        }
    }  // if not init
    if (!if_localmap_initialized) {
        if (points_buffer_.empty() || imu_buffer_.empty()) return;

        if (!if_inertial_initialized) {
            if (static_cast<int>(imu_buffer_.size()) < INIT_IMU_COUNT) {
                LOG(INFO) << "Imu buffer is not enough, wait";
                return;
            }
            Vec3 acc_sum = Vec3::Zero();
            Vec3 acc_gyro = Vec3::Zero();
            for (int i = 0; i < INIT_IMU_COUNT; ++i) {
                acc_sum += imu_buffer_[i].linear_acceleration;
                acc_gyro += imu_buffer_[i].angular_velocity;
            }

            Vec3 mean_acc = acc_sum / INIT_IMU_COUNT;
            Vec3 gravity_body = mean_acc.normalized() * RESPLE_LIO::GRAVITY_NORM;
            Mat3 R0 = Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
            double yaw = std::atan2(R0(1, 0), R0(0, 0));
            R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;
            Eigen::Quaterniond q0(R0);
            resple_lio_->bg = acc_gyro / INIT_IMU_COUNT;
            resple_lio_->gravity = q0 * gravity_body;
            resple_lio_->spl.init(param->resple_.dt_ns, 0, q0);
            if_inertial_initialized = true;
            LOG(INFO) << "Inertial initialized, gravity: " << resple_lio_->gravity.transpose();
        }

        std::int64_t first_point_time_ns = points_buffer_.front().GetTimeNs();
        std::int64_t end_time_ns = first_point_time_ns + INIT_POINTS_SPAN;
        if (points_buffer_.back().GetTimeNs() < end_time_ns || imu_buffer_.back().time_ns < end_time_ns) {
            LOG(INFO) << "buffer is not enough, wait";
            return;
        }
        PointCloud::Ptr init_points(new PointCloud);
        while (resple_lio_->spl.maxTimeNs() < end_time_ns) {
            resple_lio_->spl.addOneStateKnot(Vec3::Zero(), Vec3::Zero());
        }
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < resple_lio_->spl.maxTimeNs()) {
            init_points->emplace_back(points_buffer_.front());
            points_buffer_.pop_front();
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < resple_lio_->spl.maxTimeNs()) {
            imu_buffer_.pop_front();
        }
        while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < resple_lio_->spl.maxTimeNs()) {
            obss_time_buffer_.pop_front();
            obss_id_buffer_.pop_front();
        }
        while (!image_buff_.empty() && image_buff_.front().time_ns < resple_lio_->spl.maxTimeNs()) {
            image_buff_.pop_front();
        }

        if (init_points->size() < 100) {
            LOG(WARNING) << "only " << init_points->size() << " points for map init, waiting";
            return;
        }
        PointCloud::Ptr init_points_world(new PointCloud);
        Pose3 world_from_lidar(resple_lio_->spl.q_idle, Vec3::Zero());
        world_from_lidar = world_from_lidar * param->extrin_il_;
        pcl::transformPointCloud(*init_points, *init_points_world, world_from_lidar.Mat4d(), true);
        std::vector<Vec3> init_ps;
        for (const Point& pt : init_points_world->points) init_ps.push_back(pt.getVector3fMap().cast<double>());
        resple_lio_->lba->InitMap(init_ps, param->lba_.range_err * param->lba_.range_err + param->lba_.pose_var);
        if_localmap_initialized = true;
        LOG(INFO) << "localmap initialized with " << init_points_world->size() << " points";
        return;
    }

    while (CollectMeasures(resple_lio_->spl.maxTimeNs() + param->resple_.dt_ns)) {
        resple_lio_->propRCP();
        resple_lio_->ProcessMeasurement(measures_);
        PublishROSMsg();
        PostUpdate();
    }
}

void LaserMapping::PostUpdate() {
    int64_t interp_time_ns = resple_lio_->spl.maxTimeNs() - param->resple_.dt_ns;
    Eigen::Isometry3d body_pose = resple_lio_->spl.Interpolate(interp_time_ns);
    trajectory_.emplace_back(global_offset_time + interp_time_ns * 1e-9, body_pose);
}
void LaserMapping::PostUpdateFastLio() {
    const std::int64_t time_ns = measures_.mea_end_;
    trajectory_.emplace_back(global_offset_time + time_ns * 1e-9, fast_lio_->x.Isometry());

    if (param->pcd_save_en_) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "scans");
            return 0;
        }();
        static Vec3 last_save_pos = Vec3::Zero();
        if (last_save_pos == Vec3::Zero() || (last_save_pos - fast_lio_->x.p).norm() > param->pcd_save_interval_) {
            last_save_pos = fast_lio_->x.p;
            PointCloud::Ptr save_points{new PointCloud()};
            save_points->reserve(fast_lio_->points.size());
            for (const FastLioPoint& pt : fast_lio_->points) {
                Point point{};
                point.getVector3fMap() = pt.body.cast<float>();  // with traj_log.txt, what run_pgo and run_clean read
                point.SetTimeNs(time_ns);
                save_points->push_back(point);
            }

            double time_s = global_offset_time + time_ns * 1e-9;
            std::stringstream stamp_string;
            stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s << ".ply";
            pcl::io::savePLYFileBinary((fs::path(output_dir) / "scans" / stamp_string.str()).string(),
                                       *save_points);
        }
    }

    if (param->image_save_en_ && !measures_.rgb_image_.Empty()) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "images");
            return 0;
        }();
        static Vec3 last_save_pos = Vec3::Zero();
        if (last_save_pos == Vec3::Zero() || (last_save_pos - fast_lio_->x.p).norm() > param->image_save_interval_) {
            last_save_pos = fast_lio_->x.p;
            double time_s = global_offset_time + time_ns * 1e-9;
            std::stringstream stamp_string;
            stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s << ".png";
            if (!measures_.rgb_image_.Save((fs::path(output_dir) / "images" / stamp_string.str()).string()))
                LOG(INFO) << "failed to save image at " << time_s;
        }
    }
}

void LaserMapping::PublishROSMsg() {
    // publish
    PublishOdometry(resple_lio_->spl.Interpolate(resple_lio_->spl.maxTimeNs() - 1), resple_lio_->spl.maxTimeNs());
    PublishFrameWorld();
    PublishFrameEffectWorld();
    PublishFrustrum(resple_lio_->spl.maxTimeNs());
}

void LaserMapping::PublishROSMsgFastLio() {
    const std::int64_t time_ns = measures_.mea_end_;
    PublishOdometry(fast_lio_->x.Isometry(), time_ns);
    if (!fast_lio_->points.empty()) {
        const ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(time_ns));
        PointCloud::Ptr points_world(new PointCloud);
        PointCloud::Ptr eff_points(new PointCloud);
        for (const FastLioPoint& pt : fast_lio_->points) {
            Point point{};
            point.getVector3fMap() = pt.world.cast<float>();
            points_world->push_back(point);
            if (pt.valid) eff_points->push_back(point);
        }
        sensor_msgs::PointCloud2 scan_msg;
        pcl::toROSMsg(*points_world, scan_msg);
        scan_msg.header.stamp = ros_stamp;
        scan_msg.header.frame_id = "world";
        if (pub_laser_cloud_world_) pub_laser_cloud_world_.publish(scan_msg);
        if (bag_.isOpen()) bag_.write("/cloud_registered", ros_stamp, scan_msg);

        sensor_msgs::PointCloud2 eff_scan_msg;
        pcl::toROSMsg(*eff_points, eff_scan_msg);
        eff_scan_msg.header.stamp = ros_stamp;
        eff_scan_msg.header.frame_id = "world";
        if (pub_laser_cloud_effect_world_) pub_laser_cloud_effect_world_.publish(eff_scan_msg);
        if (bag_.isOpen()) bag_.write("/cloud_registered_effect_world", ros_stamp, eff_scan_msg);
    }
    PublishFrustrum(time_ns);
}
bool LaserMapping::CollectMeasuresFastLio() {
    std::int64_t meas_end;
    if (param->fast_lio_.mode == FastLioMode::LIVO) {
        while (!image_buff_.empty() && image_buff_.front().time_ns <= fast_lio_->last_end_ns) {
            image_buff_.pop_front();
        }
        if (image_buff_.empty()) return false;
        meas_end = image_buff_.front().time_ns;
    } else {
        meas_end = fast_lio_->last_end_ns + S_TO_NS(param->lba_.frame_dt);
    }

    if (points_buffer_.empty() || imu_buffer_.empty()) return false;
    if (imu_buffer_.back().time_ns < meas_end) return false;
    if (points_buffer_.back().GetTimeNs() < meas_end) return false;

    measures_.mea_end_ = meas_end;
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns <= meas_end) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    measures_.lidar_->clear();
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() <= meas_end) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }
    // no image processing yet, the frame only carries it
    measures_.rgb_image_ = RgbImage();
    while (!image_buff_.empty() && image_buff_.front().time_ns <= meas_end) {
        measures_.rgb_image_ = std::move(image_buff_.front());
        image_buff_.pop_front();
    }
    return true;
}

bool LaserMapping::CollectMeasures(const std::int64_t& meas_end) {
    if (points_buffer_.empty()) return false;
    if (imu_buffer_.empty()) return false;

    // buffer is not beyond the measures end
    if (imu_buffer_.back().time_ns < meas_end) return false;
    if (points_buffer_.back().GetTimeNs() < meas_end) return false;

    // push the imu data
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < meas_end) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    // push the lidar points data
    measures_.lidar_->clear();
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < meas_end) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }
    // process the raw measurement
    measures_.obs_id = -1;
    while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < meas_end) {
        measures_.obs_id = obss_id_buffer_.front();
        measures_.obs_time_ns = obss_time_buffer_.front();
        obss_id_buffer_.pop_front();
        obss_time_buffer_.pop_front();
    }
    // the latest image up to the window's end; an image can arrive after its window, so its time may be older
    measures_.rgb_image_ = RgbImage();
    while (!image_buff_.empty() && image_buff_.front().time_ns < meas_end) {
        measures_.rgb_image_ = std::move(image_buff_.front());
        image_buff_.pop_front();
    }

    if (measures_.lidar_->empty()) {
        LOG(INFO) << "Empty lidar points in measures";
    }
    return true;
}
void LaserMapping::PublishOdometry(const Eigen::Isometry3d& pub_pose, std::int64_t time_ns) {
    nav_msgs::Odometry body_odometry;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = pub_pose.translation().x();
    pose_stamped.pose.position.y = pub_pose.translation().y();
    pose_stamped.pose.position.z = pub_pose.translation().z();
    Eigen::Quaterniond q(pub_pose.linear());
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(time_ns));
    pose_stamped.header.stamp = ros_stamp;
    pose_stamped.header.frame_id = "world";

    body_odometry.header.frame_id = "world";
    body_odometry.child_frame_id = "body";
    body_odometry.header.stamp = ros_stamp;
    body_odometry.pose.pose = pose_stamped.pose;

    // publish
    static nav_msgs::Path path;
    static int call_once = [&]() {
        path.header.frame_id = "world";
        path.header.stamp = ros_stamp;
        return 0;
    }();
    if (pub_odom_aft_mapped_) pub_odom_aft_mapped_.publish(body_odometry);
    path.poses.push_back(pose_stamped);
    if (pub_path_) pub_path_.publish(path);
    if (bag_.isOpen()) bag_.write("/Odometry", ros_stamp, pose_stamped);
    if (bag_.isOpen()) bag_.write("/path", ros_stamp, path);

    // transform broadcast
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion tf_q;
    tf::Vector3 t;
    t.setX(pub_pose.translation().x());
    t.setY(pub_pose.translation().y());
    t.setZ(pub_pose.translation().z());
    tf_q.setX(q.x());
    tf_q.setY(q.y());
    tf_q.setZ(q.z());
    tf_q.setW(q.w());
    transform.setOrigin(t);
    transform.setRotation(tf_q);
    br.sendTransform(tf::StampedTransform(transform, body_odometry.header.stamp, "world", "body"));
}

void LaserMapping::PublishFrameWorld() {
    if (resple_lio_->pt_meas.empty()) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr points_world(new PointCloud);
    points_world->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        faster_lio::Point point{};
        point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
        points_world->push_back(point);
    }
    sensor_msgs::PointCloud2 scan_msg;
    pcl::toROSMsg(*points_world, scan_msg);
    scan_msg.header.stamp = ros_stamp;
    scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_world_) pub_laser_cloud_world_.publish(scan_msg);
    if (bag_.isOpen()) bag_.write("/cloud_registered", ros_stamp, scan_msg);
}

void LaserMapping::PublishFrameEffectWorld() {
    if (resple_lio_->pt_meas.empty()) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr eff_points(new PointCloud);
    eff_points->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        if (!resple_lio_->pt_meas[i].if_valid) continue;
        faster_lio::Point point{};
        point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
        eff_points->push_back(point);
    }
    sensor_msgs::PointCloud2 eff_scan_msg;
    pcl::toROSMsg(*eff_points, eff_scan_msg);
    eff_scan_msg.header.stamp = ros_stamp;
    eff_scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_effect_world_) pub_laser_cloud_effect_world_.publish(eff_scan_msg);
    if (bag_.isOpen()) bag_.write("/cloud_registered_effect_world", ros_stamp, eff_scan_msg);
}

void LaserMapping::PublishFrustrum(std::int64_t time_ns) {
    if (!pub_frustrum_) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(time_ns));
    // virtual pinhole camera, drawn at the body-to-camera extrinsic (identity if uncalibrated)
    constexpr double kDepth = 1.0;        // frustum depth, meters
    constexpr double kHalfFovH = 0.5236;  // 60 deg horizontal half-FOV, rad
    constexpr double kHalfFovV = 0.3927;  // 45 deg vertical half-FOV, rad
    const double half_w = kDepth * std::tan(kHalfFovH);
    const double half_h = kDepth * std::tan(kHalfFovV);

    const Pose3& body_from_cam = param->extrin_ic_;
    const Vec3 apex_cam(0, 0, 0);
    const std::array<Vec3, 4> corners_cam = {
        Vec3(-half_w, -half_h, kDepth),
        Vec3(half_w, -half_h, kDepth),
        Vec3(half_w, half_h, kDepth),
        Vec3(-half_w, half_h, kDepth),
    };

    const Vec3 apex = body_from_cam * apex_cam;
    std::array<Vec3, 4> corners;
    for (int i = 0; i < 4; ++i) corners[i] = body_from_cam * corners_cam[i];

    visualization_msgs::Marker marker;
    marker.header.frame_id = "body";

    marker.header.stamp = ros_stamp;
    marker.ns = "frustrum";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.02;  // line width
    marker.color.r = 1.0;
    marker.color.g = 0.65;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    auto add_point = [&marker](const Vec3& p) {
        geometry_msgs::Point pt;
        pt.x = p.x();
        pt.y = p.y();
        pt.z = p.z();
        marker.points.push_back(pt);
    };

    // apex to each corner
    for (int i = 0; i < 4; ++i) {
        add_point(apex);
        add_point(corners[i]);
    }
    // rectangle at the far plane
    for (int i = 0; i < 4; ++i) {
        add_point(corners[i]);
        add_point(corners[(i + 1) % 4]);
    }

    pub_frustrum_.publish(marker);
}

void LaserMapping::Savetrajectory() {
    LOG(INFO) << "Save states in imu frame";
    TrajectoryGenerator::save_to_tumtxt(trajectory_,
                                        (fs::path(output_dir) / "traj_log.txt").string());
}

void LaserMapping::Finish() {
    bag_.close();
}
}  // namespace faster_lio

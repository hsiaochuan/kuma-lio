//
// Created by hsiaochuan on 2026/09/08.
//

#include "rtk_processor.h"

#include "factor/GnssClockFactor.h"
#include "factor/GnssPsrDoppFactor.h"
#include "factor/GnssTdCpFactor.h"
#include "rtklib_func.h"
RTKProcessor::RTKProcessor() {
    prcopt = prcopt_default;
    prcopt.mode = PMODE_SINGLE;
    prcopt.soltype = 0;
    prcopt.nf = 1;
    prcopt.navsys = SYS_GPS | SYS_GLO | SYS_GAL | SYS_CMP;
    prcopt.glomodear = GLO_ARMODE_ON;
    prcopt.refpos = POSOPT_SINGLE;
    prcopt.thresdop = 1.0;
    for (int f = 0; f < NFREQ; ++f) {
        for (int i = 0; i < 9; ++i) prcopt.snrmask.mask[f][i] = 0;
    }
    prcopt.snrmask.ena[0] = prcopt.snrmask.ena[1] = 0;

    solopt = solopt_default;
    solopt.posf = SOLF_XYZ;     // ECEF, so readsolt() hands back sol.rr directly
    solopt.times = TIMES_GPST;  // same time base as the observations
    solopt.timef = 0;           // week + tow
    solopt.timeu = 6;
    solopt.outhead = 1;    // readsolt() needs the header to pick the decoder
    solopt.outvel = 1;     // velocity round-trips through the file as well
    solopt.solstatic = 0;  // one line per epoch, even in a static mode
    solopt.sstat = 0;
    solopt.maxsolstd = 0.0;
}
static int SysIdx(int sys) {
    if (sys == SYS_GPS) return 0;
    if (sys == SYS_GLO) return 1;
    if (sys == SYS_GAL) return 2;
    if (sys == SYS_CMP) return 3;
    if (sys == SYS_QZS) return 4;
    throw std::runtime_error("unknown system");
}
bool RTKProcessor::Alignment(Mat3& R_ecef_enu, Vec3& t_ecef_enu) {
    CHECK(locals.size() == ecefs.size());
    if (travel_len < param->ali_min_travel) return false;
    int N = locals.size();
    if (N < 3) return false;
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
void RTKProcessor::ProcessMeasurements(const int& obs_i, const RtkState& lio_state) {
    // SPP
    int n = 0;
    while (std::abs(timediff(obss.data[obs_i].time, obss.data[obs_i + n].time)) < DTTOL) ++n;
    sol_t sol;
    char err_msg[128];
    if (!pntpos(obss.data + obs_i, n, &nav, &prcopt, &sol, nullptr, nullptr, err_msg)) {
        printf("%s\n", err_msg);
        return;
    } else {
        outsolex(stdout,&sol, nullptr, &solopt);
    }

    // alignment
    if (!if_alignment) {
        if (locals.empty()) {
            locals.push_back(lio_state.pos);
            ecefs.push_back(Vec3(sol.rr[0], sol.rr[1], sol.rr[2]));
        } else {
            Vec3 last_local = locals.back();
            double incre = (lio_state.pos - last_local).norm();
            if (incre > 0.3) {
                locals.push_back(lio_state.pos);
                ecefs.push_back(Vec3(sol.rr[0], sol.rr[1], sol.rr[2]));
                travel_len += incre;
            }
        }
        if (travel_len > param->ali_min_travel && Alignment(R_ecef_enu, t_ecef_enu)) {
            if_alignment = true;
            LOG(INFO) << "GNSS alignment success";
            return;
        }
        LOG(INFO) << "GNSS alignment fail";
        return;
    }

    gtsam::NonlinearFactorGraph factors;
    gtsam::Values values;
    std::vector<int64_t> factor_times;
    // add values
    static int call_once = [&]() {
        values.insert(E(0), gtsam::Rot3(R_ecef_enu));
        values.insert(T(0), t_ecef_enu);
        return 0;
    }();

    int64_t local_time = lio_state.local_time;
    values.insert(R(local_time), gtsam::Rot3(lio_state.quat));
    values.insert(P(local_time), lio_state.pos);
    values.insert(D(local_time), gtsam::Vector4(sol.dtr[0], sol.dtr[1], sol.dtr[2], sol.dtr[3]));
    values.insert(O(local_time), gtsam::Vector1(0.0));
    gtsam::Vector6 bgba;
    bgba(0) = lio_state.bg.x();
    bgba(1) = lio_state.bg.y();
    bgba(2) = lio_state.bg.z();
    bgba(3) = lio_state.ba.x();
    bgba(4) = lio_state.ba.y();
    bgba(5) = lio_state.ba.z();
    values.insert(B(local_time), bgba);

    // satellite pos
    std::vector<double> rs(6 * n, 0.0), dts(2 * n, 0.0), svar(n, 0.0);
    std::vector<int> svh(n, 0);
    satposs(sol.time, obss.data + obs_i, n, &nav, prcopt.sateph, rs.data(), dts.data(), svar.data(), svh.data());
    double rcv_llh[3];
    ecef2pos(sol.rr, rcv_llh);

    for (int si = 0; si < n; ++si) {
        int oi = obs_i + si;
        const obsd_t& obs = obss.data[oi];
        int sys_idx = SysIdx(satsys(obs.sat, nullptr));
        // filter
        if (satexclude(obs.sat, svar[si], svh[si], &prcopt)) continue;
        double azel[2] = {0.0, PI / 2.0};
        double e[3];
        if (geodist(&rs[si * 6], sol.rr, e) < 0.0) continue;
        if (satazel(rcv_llh, e, azel) < prcopt.elmin) continue;
        if (testsnr(0, 0, azel[1], obs.SNR[0], &prcopt.snrmask)) continue;

        // iono and tro
        double iono_delay = 0.0, vion = 0.0, tro_delay = 0.0, vtrp = 0.0;
        ionocorr(sol.time, &nav, obs.sat, rcv_llh, azel, prcopt.ionoopt, &iono_delay, &vion);
        const double freq = sat2freq(obs.sat, obs.code[0], &nav);
        const double lambda = CLIGHT / freq;
        iono_delay *= (FREQL1 / freq) * (FREQL1 / freq);
        vion *= (FREQL1 / freq) * (FREQL1 / freq) * (FREQL1 / freq) * (FREQL1 / freq);
        tropcorr(sol.time, &nav, rcv_llh, azel, prcopt.tropopt, &tro_delay, &vtrp);

        // measure var
        double varerr = rtklib::varerr(&prcopt, &obs, azel[1], satsys(obs.sat, nullptr));

        // corrected range
        double vmeas = 0.0;
        double psr = rtklib::prange(&obs, &nav, &prcopt, &vmeas);

        // manage the sat track
        bool psr_valid = obs.P[0] > 0.0;
        bool cp_valid = obs.L[0] > 0.0;
        bool dopp_valid = obs.D[0] > 0.0;
        bool obs_valid = psr_valid && cp_valid && dopp_valid;

        bool arc_continu = true;
        do {
            if (obs_valid) {
                arc_continu = false;
                break;
            }
            if (sat_tracks_[obs.sat].obs_ids.empty()) {
                arc_continu = false;
                break;
            }
            if ((obs.LLI[0] & 0x3) != 0) {
                arc_continu = false;
                break;
            }

            int last_oi = sat_tracks_[obs.sat].obs_ids.back();
            obsd_t& last_obs = obss.data[last_oi];
            double dt = timediff(obs.time, last_obs.time);
            const double dpt = (-0.5) * lambda * (obs.D[0] + last_obs.D[0]) * dt;
            const double dcp = obs.L[0] * lambda - last_obs.L[0] * lambda;
            if (std::abs(dcp - dpt) > 3 * prcopt.thresdop) {
                arc_continu = false;
                break;
            }
        } while (false);

        Eigen::Vector3d sv(rs[si * 6], rs[si * 6 + 1], rs[si * 6 + 2]);
        if (arc_continu) {
            int last_oi = sat_tracks_[obs.sat].obs_ids.back();
            obsd_t& last_obs = obss.data[last_oi];
            sat_tracks_[obs.sat].obs_ids.push_back(oi);
            sat_tracks_[obs.sat].sv_poss.push_back(sv);
            int N = sat_tracks_[obs.sat].obs_ids.size();
            sat_tracks_[obs.sat].hatch =
                psr / N + (N - 1.0) / N * (sat_tracks_[obs.sat].hatch + obs.L[0] * lambda - last_obs.L[0] * lambda);
            sat_tracks_[obs.sat].local_times.push_back(local_time);
        } else {
            // clear the track
            sat_tracks_[obs.sat].obs_ids.clear();
            sat_tracks_[obs.sat].sv_poss.clear();
            sat_tracks_[obs.sat].hatch = 0.0;
            sat_tracks_[obs.sat].local_times.clear();
            // insert the one observation
            sat_tracks_[obs.sat].obs_ids.push_back(oi);
            sat_tracks_[obs.sat].sv_poss.push_back(sv);
            sat_tracks_[obs.sat].hatch = psr;
            sat_tracks_[obs.sat].local_times.push_back(local_time);
        }

        // psr factor
        gtsam::SharedNoiseModel psr_weight = gtsam::noiseModel::Isotropic::Sigma(1, param->psr_weight);
        GnssPsrFactor psr_factor(P(local_time), D(local_time), T(0), E(0), psr_weight);
        psr_factor.sv_pos = Eigen::Vector3d(rs[si * 6 + 0], rs[si * 6 + 1], rs[si * 6 + 2]);
        psr_factor.svdt = dts[si * 2 + 0];
        psr_factor.ion_delay = iono_delay;
        psr_factor.tro_delay = tro_delay;
        psr_factor.noise_var = svar[si] + vmeas + vion + vtrp + varerr;
        psr_factor.psr_measure = sat_tracks_[obs.sat].hatch;
        psr_factor.sys_idx = sys_idx;
        factors.add(psr_factor);
        factor_times.push_back(local_time);
        // TDCP factor
        if (arc_continu && sat_tracks_[obs.sat].obs_ids.size() >= 2) {
            int N = sat_tracks_[obs.sat].obs_ids.size();
            for (int i = 0; i < N - 1; ++i) {
                SatTrack& sat_track = sat_tracks_[obs.sat];
                if (sat_track.local_times[i] < slw_times.front()) continue;
                gtsam::SharedNoiseModel tdcp_weight = gtsam::noiseModel::Isotropic::Sigma(1, param->psr_weight);
                GnssTdCpFactor tdcp_factor(P(sat_track.local_times[i]), P(local_time), D(sat_track.local_times[i]),
                                           D(local_time), T(0), E(0), tdcp_weight);
                obsd_t& prev_obs = obss.data[sat_track.obs_ids[i]];
                tdcp_factor.sv_pos_i = sat_track.sv_poss[i];
                tdcp_factor.sv_pos_j = Vec3(rs[si * 6], rs[si * 6 + 1], rs[si * 6 + 2]);
                tdcp_factor.le_arm_local = param->level_arm;
                tdcp_factor.sys_idx = sys_idx;
                tdcp_factor.cp_measured = obs.L[0] - prev_obs.L[0];
                factors.add(tdcp_factor);
                factor_times.push_back(sat_track.local_times[i]);
            }
        }
    }  // for satellite

    if (slw_times.size() > 0) {
        // dt factor
        std::int64_t prev_time = slw_times.back();
        gtsam::SharedNoiseModel dt_weight = gtsam::noiseModel::Isotropic::Sigma(1, param->psr_weight);
        double dt = NS_TO_S(local_time - prev_time);
        GnssDtDdtFactor dt_factor(D(local_time), D(prev_time), O(local_time), O(prev_time), dt, dt_weight);
        factors.add(dt_factor);
        factor_times.push_back(prev_time);

        // ddt factor
        gtsam::SharedNoiseModel ddt_weight = gtsam::noiseModel::Isotropic::Sigma(1, param->psr_weight);
        GnssDdtSmoothFactor ddt_factor(O(local_time), O(prev_time), ddt_weight);
        factors.add(ddt_factor);
        factor_times.push_back(prev_time);
    }

    // update values
    slw_times.push_back(local_time);
    gtsam::ISAM2Result isam_result = isam.update(factors, values);
    for (int i = 0; i < factors.size(); ++i) {
        std::int64_t factor_time = factor_times[i];
        auto factor_index = isam_result.newFactorsIndices.at(i);
        slw_factors[factor_time].push_back(factor_index);
    }

    if (slw_times.size() > param->slw_size) {
        // find the factor that connect the oldest time
        std::int64_t oldest_time = slw_times[0];
        gtsam::FactorIndices delete_factors = slw_factors[oldest_time];
        isam.update(gtsam::NonlinearFactorGraph{}, gtsam::Values{}, delete_factors);
    }
}

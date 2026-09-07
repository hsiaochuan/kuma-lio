//
// Created by hsiaochuan on 2026/09/01.
//

#include "gnss_processor.h"

#include <glog/logging.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

#include "factor/GnssClockFactor.h"
#include "factor/GnssPsrDoppFactor.h"
#include "factor/GnssTdCpFactor.h"
#include "factor/LioOdomFactor.h"

// RTKLIB declares these three but only defines them for shared-library builds, so every
// application that links postpos() has to supply them.  postpos() calls showmsg() for its
// progress line and aborts as soon as it returns a non-zero value.
extern "C" {
int showmsg(const char* format, ...) {
    if (VLOG_IS_ON(2)) {
        char buf[256];
        va_list ap;
        va_start(ap, format);
        vsnprintf(buf, sizeof(buf), format, ap);
        va_end(ap);
        VLOG(2) << buf;
    }
    return 0;
}
void settspan(gtime_t, gtime_t) {}
void settime(gtime_t) {}
}

using gtsam::symbol_shorthand::A;  // Vector3, local position at one GNSS epoch
using gtsam::symbol_shorthand::B;  // Vector4, receiver clock bias per constellation (m)
using gtsam::symbol_shorthand::C;  // Vector1, receiver clock drift (m/s)
using gtsam::symbol_shorthand::E;  // Vector3, ECEF position of the LIO frame origin
using gtsam::symbol_shorthand::P;  // Rot3,    R_ecef_local

namespace {
Eigen::Matrix3d EnuFromEcefRotation(const double* pos) {
    double e[9];
    xyz2enu(pos, e);
    return Eigen::Map<const Eigen::Matrix3d>(e);
}

/// Index into the four clock biases we carry.  QZSS shares the GPS clock, exactly as
/// RTKLIB's pntpos() does; anything else has no slot and is dropped.
int SysToIdx(int sys) {
    switch (sys) {
        case SYS_GPS:
        case SYS_QZS:
            return GnssProcessor::SYSGPS;
        case SYS_GLO:
            return GnssProcessor::SYSGLO;
        case SYS_GAL:
            return GnssProcessor::SYSGAL;
        case SYS_CMP:
            return GnssProcessor::SYSBDS;
        default:
            return -1;
    }
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

}  // namespace

GnssProcessor::GnssProcessor() { SetupOptions(); }

GnssProcessor::~GnssProcessor() {
    free(obs.data);
    obs.data = nullptr;
    obs.n = obs.nmax = 0;
    freenav(&nav, 0xFF);
    freesolbuf(&solbuf);
}

/// Called once, from the constructor: it only moves prcopt_default to what this pipeline
/// needs.  Every option stays a prcopt/solopt field the caller overrides afterwards.
void GnssProcessor::SetupOptions() {
    prcopt = prcopt_default;
    prcopt.mode = PMODE_SINGLE;
    prcopt.soltype = 0;
    prcopt.nf = 1;
    prcopt.navsys = SYS_GPS | SYS_GLO | SYS_GAL | SYS_CMP | SYS_QZS;
    prcopt.glomodear = GLO_ARMODE_ON;
    prcopt.refpos = POSOPT_SINGLE;
    prcopt.thresdop = 1.0;
    SetSnrMask(25.0);
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

void GnssProcessor::SetSnrMask(double snr_dbhz) {
    // snrmask_t is an elevation-banded table; a flat threshold is the same value in all bands.
    for (int f = 0; f < NFREQ; ++f) {
        for (int i = 0; i < 9; ++i) prcopt.snrmask.mask[f][i] = snr_dbhz;
    }
}

bool GnssProcessor::LoadLioStates() {
    if (!lio_states.empty()) {
        LOG(INFO) << "using " << lio_states.size() << " lio states supplied by the caller";
        return true;
    }
    if (config_.lio_fname.empty()) {
        LOG(ERROR) << "no lio trajectory given and lio_states is empty";
        return false;
    }

    std::ifstream fin(config_.lio_fname);
    if (!fin) {
        LOG(ERROR) << "failed to open lio trajectory: " << config_.lio_fname;
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        gtime_t gpst;
        Eigen::Vector3d pos;
        Eigen::Quaterniond quat;
        iss >> gpst.time >> gpst.sec >> pos.x() >> pos.y() >> pos.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();
        LioState ls;
        ls.time = gpst;
        ls.pos = pos;
        ls.quat = quat;
        lio_states.push_back(ls);
    }
    fin.close();

    if (lio_states.size() < 2) {
        LOG(ERROR) << "lio trajectory has fewer than two poses: " << config_.lio_fname;
        return false;
    }

    LOG(INFO) << "loaded " << lio_states.size() << " lio states";
    return true;
}

bool GnssProcessor::ReadRinex() {
    gtime_t ts{}, te{};  // zero epochs disable the time window filter

    // rcv index 1 for the observation file; the navigation file carries no receiver data.
    if (readrnxt(config_.obs_fname.c_str(), 1, ts, te, 0.0, "", &obs, &nav, nullptr) <= 0) {
        LOG(ERROR) << "failed to read observation file: " << config_.obs_fname;
        return false;
    }
    if (readrnxt(config_.nav_fname.c_str(), 0, ts, te, 0.0, "", nullptr, &nav, nullptr) <= 0) {
        LOG(ERROR) << "failed to read navigation file: " << config_.nav_fname;
        return false;
    }

    // RTKLIB expects observations sorted by time/sat and ephemerides uniquified before use.
    sortobs(&obs);
    uniqnav(&nav);

    LOG(INFO) << "read " << obs.n << " observations, " << nav.n << " eph, " << nav.ng << " geph, " << nav.ns << " seph";
    if (obs.n <= 0 || nav.n + nav.ng <= 0) {
        LOG(ERROR) << "no usable GNSS data; check that RTKLIB is built with -DENAGLO/-DENAGAL/-DENACMP";
        return false;
    }
    return true;
}

bool GnssProcessor::RunPostpos() {
    if (config_.sol_fname.empty()) {
        if (output_dir.empty()) {
            LOG(ERROR) << "no --sol_fname and no --out_dir to derive one from";
            return false;
        }
        config_.sol_fname = output_dir + "/gnss_postpos.pos";
    }

    std::vector<std::string> files{config_.obs_fname, config_.nav_fname};
    files.insert(files.end(), config_.extra_files.begin(), config_.extra_files.end());
    std::vector<const char*> infile;
    for (const auto& f : files) infile.push_back(f.c_str());

    // position
    gtime_t ts{}, te{};
    filopt_t filopt{};
    LOG(INFO) << "postpos: mode " << prcopt.mode << ", " << infile.size() << " input files -> " << config_.sol_fname;
    const int ret = postpos(ts, te, 0.0, 0.0, &prcopt, &solopt, &filopt, infile.data(), static_cast<int>(infile.size()),
                            config_.sol_fname.c_str(), "", "");
    if (ret != 0) {
        LOG(ERROR) << "postpos failed with status " << ret;
        return false;
    }

    // read position result
    const char* solfiles[1] = {config_.sol_fname.c_str()};
    readsolt(solfiles, 1, ts, te, 0.0, 0, 0, &solbuf);
    if (solbuf.n <= 0) {
        LOG(ERROR) << "postpos produced no solution in " << config_.sol_fname;
        return false;
    }
    LOG(INFO) << "postpos produced " << solbuf.n << " epochs";
    return true;
}

int GnssProcessor::BuildNodes() {
    nodes_.clear();

    size_t si = 0, li = 0;
    int no_sol = 0, no_lio = 0, few_sat = 0;
    for (int i = 0; i < obs.n;) {
        int n = 0;
        while (i + n < obs.n && n < MAXOBS && fabs(timediff(obs.data[i + n].time, obs.data[i].time)) <= DTTOL) {
            ++n;
        }
        const int head = i;
        const gtime_t t = obs.data[head].time;
        i += n;

        // solbuf and lio_states are both time ordered, so a single merge sweep suffices.
        while (si + 1 < static_cast<size_t>(solbuf.n) &&
               std::fabs(timediff(solbuf.data[si + 1].time, t)) < std::fabs(timediff(solbuf.data[si].time, t))) {
            ++si;
        }
        while (li + 1 < lio_states.size() &&
               std::fabs(timediff(lio_states[li + 1].time, t)) < std::fabs(timediff(lio_states[li].time, t))) {
            ++li;
        }

        const sol_t& sol = solbuf.data[si];
        if (std::fabs(timediff(sol.time, t)) > DTTOL || sol.stat == SOLQ_NONE) {
            ++no_sol;
            continue;
        }
        if (sol.ns < config_.min_sat) {
            ++few_sat;
            continue;
        }
        if (std::fabs(timediff(lio_states[li].time, t)) > config_.match_tol) {
            ++no_lio;
            continue;
        }

        Node nd;
        nd.obs_i = head;
        nd.obs_n = n;
        nd.sol_i = static_cast<int>(si);
        nd.lio_i = static_cast<int>(li);
        nodes_.push_back(nd);
    }  // for obs

    LOG(INFO) << "paired " << nodes_.size() << " epochs (dropped " << no_sol << " without a postpos fix, " << few_sat
              << " below min_sat, " << no_lio << " without a lio state)";
    if (nodes_.size() < 2) {
        LOG(ERROR) << "fewer than two usable epochs; check lio_time_offset/match_tol and the postpos mode";
    }
    return static_cast<int>(nodes_.size());
}

void GnssProcessor::EstimateClockBias(size_t k, double* dt) const {
    // Inverting the pseudorange residual of GnssPsrDoppFactor at the postpos position gives
    // the receiver clock bias directly, which keeps the initial value consistent with the
    // model the graph actually optimises (the .pos file does not carry sol.dtr).
    std::vector<double> rs, dts, var;
    std::vector<int> svh;
    const int n = nodes_[k].obs_n;
    rs.assign(6 * n, 0.0);
    dts.assign(2 * n, 0.0);
    var.assign(n, 0.0);
    svh.assign(n, 0);
    satposs(EpochSol(k).time, EpochObs(k), n, &nav, prcopt.sateph, rs.data(), dts.data(), var.data(), svh.data());

    const sol_t& sol = EpochSol(k);
    double rcv_llh[3];
    ecef2pos(sol.rr, rcv_llh);

    std::vector<double> per_sys[SYSTOTAL];
    for (int j = 0; j < n; ++j) {
        const obsd_t& o = EpochObs(k)[j];

        const int sys_idx = SysToIdx(satsys(o.sat, nullptr));
        if (sys_idx < 0) continue;  // no clock slot for this constellation

        if (satexclude(o.sat, var[j], svh[j], &prcopt)) continue;

        double e[3];
        const double range = geodist(&rs[j * 6], sol.rr, e);
        if (range <= 0.0) continue;
        double azel[2] = {0.0, PI / 2.0};
        if (satazel(rcv_llh, e, azel) < prcopt.elmin) continue;

        // prcopt.snrmask.ena is left off so that postpos() does not drop every observation of a
        // file without S# records; the mask still has to apply here, so enable it on a local copy.
        snrmask_t mask = prcopt.snrmask;
        mask.ena[0] = 1;
        if (testsnr(0, 0, azel[1], o.SNR[0], &mask)) continue;

        // sat2freq() returns 0 for a GLONASS satellite with no frequency channel number and for
        // an unknown code; letting that through makes lambda infinite and NaNs the whole solve.
        const double freq = sat2freq(o.sat, o.code[0], &nav);
        if (freq <= 0.0) continue;

        double vmeas = 0.0;
        const double psr = rtklib::prange(&o, &nav, &prcopt, &vmeas);
        if (psr == 0.0) continue;

        const double svdt = dts[j * 2 + 0];

        // rescode()'s residual with the clock left in: v = P - (r - c*dts + dion + dtrp).
        double iono = 0.0, vion = 0.0, trop = 0.0, vtrp = 0.0;
        ionocorr(sol.time, &nav, o.sat, rcv_llh, azel, prcopt.ionoopt, &iono, &vion);
        iono *= (FREQL1 / freq) * (FREQL1 / freq);
        tropcorr(sol.time, &nav, rcv_llh, azel, prcopt.tropopt, &trop, &vtrp);
        per_sys[sys_idx].push_back(psr - (range - svdt * CLIGHT + iono + trop));
    }

    const double gps = Median(per_sys[SYSGPS]);
    for (int s = 0; s < SYSTOTAL; ++s) {
        // A constellation with no satellite this epoch inherits the GPS clock; the graph's
        // inter-system offset then grows out of its own residuals.
        dt[s] = per_sys[s].empty() ? gps : Median(per_sys[s]);
    }
}

bool GnssProcessor::FitTransform(const std::vector<Eigen::Vector3d>& local, const std::vector<Eigen::Vector3d>& ecef,
                                 const std::vector<size_t>& idx, Eigen::Matrix3d& R, Eigen::Vector3d& t) const {
    if (idx.size() < 3) return false;

    Eigen::Vector3d mean_l = Eigen::Vector3d::Zero(), mean_e = Eigen::Vector3d::Zero();
    for (size_t i : idx) {
        mean_l += local[i];
        mean_e += ecef[i];
    }
    mean_l /= static_cast<double>(idx.size());
    mean_e /= static_cast<double>(idx.size());

    if (config_.align_yaw_only) {
        // The LIO frame is gravity aligned, so ECEF<-local is the local ENU rotation at the
        // anchor times an unknown heading.  Solving for that single angle in the ENU plane is
        // far better conditioned than a full 3-DoF rotation fit on noisy fixes.
        double anc_llh[3];
        ecef2pos(mean_e.data(), anc_llh);
        const Eigen::Matrix3d R_enu_ecef = EnuFromEcefRotation(anc_llh);

        double num = 0.0, den = 0.0;
        for (size_t i : idx) {
            const Eigen::Vector3d s = local[i] - mean_l;
            const Eigen::Vector3d d = R_enu_ecef * (ecef[i] - mean_e);
            num += s.x() * d.y() - s.y() * d.x();  // sin part of sum d' * Rz(yaw) * s
            den += s.x() * d.x() + s.y() * d.y();  // cos part
        }
        const double yaw = (std::fabs(num) + std::fabs(den) < 1e-9) ? 0.0 : std::atan2(num, den);
        const Eigen::Matrix3d Rz(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        R = R_enu_ecef.transpose() * Rz;
    } else {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i : idx) {
            H += (local[i] - mean_l) * (ecef[i] - mean_e).transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
        R = svd.matrixV() * D * svd.matrixU().transpose();
    }

    t = mean_e - R * mean_l;
    return true;
}

bool GnssProcessor::Alignment() {
    if (nodes_.size() < 3) {
        LOG(ERROR) << "too few paired epochs to align";
        return false;
    }

    // Both trajectories describe the same physical point: the antenna phase centre.
    std::vector<Eigen::Vector3d> local, ecef;
    for (size_t k = 0; k < nodes_.size(); ++k) {
        const LioState& ls = EpochLio(k);
        const sol_t& sol = EpochSol(k);
        local.push_back(ls.pos + ls.quat.toRotationMatrix() * config_.lever_arm);
        ecef.emplace_back(sol.rr[0], sol.rr[1], sol.rr[2]);
    }

    Eigen::Vector3d span_min = local.front(), span_max = local.front();
    for (const auto& p : local) {
        span_min = span_min.cwiseMin(p);
        span_max = span_max.cwiseMax(p);
    }
    const double span = (span_max - span_min).norm();
    bool heading_observable_ = span > config_.align_min_span;
    if (!heading_observable_) {
        LOG(WARNING) << "local trajectory spans only " << span << " m; heading is not observable, assuming yaw = 0";
    }

    std::vector<size_t> inlier(local.size());
    std::iota(inlier.begin(), inlier.end(), 0);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double rms = 0.0;
    for (int it = 0; it < std::max(1, config_.align_iterations); ++it) {
        if (heading_observable_) {
            if (!FitTransform(local, ecef, inlier, R, t)) return false;
        } else {
            // Degenerate case: keep the ENU frame at the mean fix with zero heading.
            Eigen::Vector3d mean_e = Eigen::Vector3d::Zero(), mean_l = Eigen::Vector3d::Zero();
            for (size_t i : inlier) {
                mean_e += ecef[i];
                mean_l += local[i];
            }
            mean_e /= static_cast<double>(inlier.size());
            mean_l /= static_cast<double>(inlier.size());
            double llh[3];
            ecef2pos(mean_e.data(), llh);
            R = EnuFromEcefRotation(llh).transpose();
            t = mean_e - R * mean_l;
        }

        std::vector<double> res(local.size());
        for (size_t i = 0; i < local.size(); ++i) {
            res[i] = (R * local[i] + t - ecef[i]).norm();
        }
        std::vector<double> res_in;
        res_in.reserve(inlier.size());
        for (size_t i : inlier) res_in.push_back(res[i]);
        const double thres = std::max(config_.align_outlier, 3.0 * Median(res_in));

        std::vector<size_t> next;
        double sq = 0.0;
        for (size_t i = 0; i < local.size(); ++i) {
            if (res[i] <= thres) {
                next.push_back(i);
                sq += res[i] * res[i];
            }
        }
        if (next.size() < 3) break;
        rms = std::sqrt(sq / static_cast<double>(next.size()));
        const bool converged = next.size() == inlier.size();
        inlier.swap(next);
        if (converged) break;
    }

    R_ecef_local_ = R;
    anc_ecef_ = t;

    double anc_llh[3];
    ecef2pos(anc_ecef_.data(), anc_llh);
    LOG(INFO) << "alignment used " << inlier.size() << "/" << local.size() << " epochs, rms " << rms << " m";
    LOG(INFO) << "anchor ecef " << anc_ecef_.transpose() << "  lla " << anc_llh[0] * R2D << " " << anc_llh[1] * R2D
              << " " << anc_llh[2];

    // Only the epochs that survived the fit stay as graph nodes; a wild fix would otherwise
    // seed the clock states with hundreds of metres of error.
    if (inlier.size() < nodes_.size()) {
        std::vector<Node> kept;
        kept.reserve(inlier.size());
        for (size_t i : inlier) kept.push_back(nodes_[i]);
        nodes_.swap(kept);
    }
    if (nodes_.size() < 2) {
        LOG(ERROR) << "fewer than two epochs survived the alignment";
        return false;
    }
    return true;
}

void GnssProcessor::AddFactor() {
    graph_ = gtsam::NonlinearFactorGraph();
    values_ = gtsam::Values();

    const size_t nn = nodes_.size();

    gtsam::SharedNoiseModel psr_base = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);
    gtsam::SharedNoiseModel cp_base = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);
    gtsam::SharedNoiseModel psr_noise = psr_base, cp_noise = cp_base;
    if (config_.robust) {
        psr_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(config_.robust_psr_k), psr_base);
        cp_noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(config_.robust_cp_k),
                                                     cp_base);
    }

    // --- initial values -----------------------------------------------------------------
    std::vector<std::array<double, SYSTOTAL>> dt_init(nn);
    for (size_t k = 0; k < nn; ++k) EstimateClockBias(k, dt_init[k].data());

    // a single median over the whole run to init the ddt
    std::vector<double> ddts;
    ddts.reserve(nn);
    for (size_t k = 1; k < nn; ++k) {
        const double dt = timediff(EpochSol(k).time, EpochSol(k - 1).time);
        ddts.push_back((dt_init[k][SYSGPS] - dt_init[k - 1][SYSGPS]) / dt);
    }
    const double ddt_init = Median(ddts);

    // init the gtsam values
    for (size_t k = 0; k < nn; ++k) {
        values_.insert(A(k), gtsam::Vector3(EpochLio(k).pos));
        values_.insert(B(k), gtsam::Vector4(dt_init[k][0], dt_init[k][1], dt_init[k][2], dt_init[k][3]));
        values_.insert(C(k), gtsam::Vector1(ddt_init));
    }
    values_.insert(E(0), gtsam::Vector3(anc_ecef_));
    values_.insert(P(0), gtsam::Rot3(R_ecef_local_));

    struct SatTrack {
        size_t node = 0;
        gtime_t last_time{-1, -1};
        double cp = 0.0;               ///< carrier phase, satellite clock corrected (m)
        double dopp_range_rate = 0.0;  ///< -lambda * D at that epoch (m/s), for the slip test
        bool dopp_valid = false;
        Eigen::Vector3d sv_pos = Eigen::Vector3d::Zero();
        int hatch_n = 0;
        double hatch = 0.0;
        double last_cp = 0.0;
    };
    std::map<int, SatTrack> sat_tracks;

    std::vector<double> rs, dts, var;
    std::vector<int> svh;
    size_t n_psr = 0, n_cp = 0, n_dtddt = 0, n_slip = 0;
    for (size_t k = 0; k < nn; ++k) {
        const obsd_t* od = EpochObs(k);
        const sol_t& sol = EpochSol(k);
        const LioState& lio_state = EpochLio(k);
        const int n = nodes_[k].obs_n;
        rs.assign(6 * n, 0.0);
        dts.assign(2 * n, 0.0);
        var.assign(n, 0.0);
        svh.assign(n, 0);
        satposs(EpochSol(k).time, EpochObs(k), n, &nav, prcopt.sateph, rs.data(), dts.data(), var.data(), svh.data());
        double rcv_llh[3];
        ecef2pos(sol.rr, rcv_llh);

        for (int j = 0; j < n; ++j) {
            const obsd_t& o = od[j];

            const int sys_idx = SysToIdx(satsys(o.sat, nullptr));
            if (sys_idx < 0) continue;  // no clock slot for this constellation

            if (satexclude(o.sat, var[j], svh[j], &prcopt)) continue;

            double e[3];
            const double range = geodist(&rs[j * 6], sol.rr, e);
            if (range <= 0.0) continue;
            double azel[2] = {0.0, PI / 2.0};
            if (satazel(rcv_llh, e, azel) < prcopt.elmin) continue;

            // prcopt.snrmask.ena is left off so that postpos() does not drop every observation of
            // a file without S# records; the mask still has to apply here, so enable it on a copy.
            snrmask_t mask = prcopt.snrmask;
            mask.ena[0] = 1;
            if (testsnr(0, 0, azel[1], o.SNR[0], &mask)) continue;

            // sat2freq() returns 0 for a GLONASS satellite with no frequency channel number and
            // for an unknown code; letting that through makes lambda infinite and NaNs the solve.
            const double freq = sat2freq(o.sat, o.code[0], &nav);
            if (freq <= 0.0) continue;
            const double lambda = CLIGHT / freq;

            double vmeas = 0.0;
            const double psr = rtklib::prange(&o, &nav, &prcopt, &vmeas);
            if (psr == 0.0) continue;

            const Eigen::Vector3d sv_pos(rs[j * 6 + 0], rs[j * 6 + 1], rs[j * 6 + 2]);
            const double svdt = dts[j * 2 + 0];
            const double sv_var = std::max(var[j], 0.0);

            SatTrack& sat_tk = sat_tracks[od[j].sat];
            const bool dopp_valid = od[j].D[0] != 0.0f;
            const bool phase_valid = od[j].L[0] != 0.0;
            const double cp_lambda = od[j].L[0] * lambda;

            bool arc_continues = true;
            if (sat_tk.last_time.time < 0 || !dopp_valid || !phase_valid || (od[j].LLI[0] & 0x3) != 0 ||
                timediff(sol.time, sat_tk.last_time) > config_.cp_max_gap)
                arc_continues = false;

            // doppler integrated to detect the slip
            if (arc_continues && sat_tk.dopp_valid) {
                const double cp_now = cp_lambda + svdt * CLIGHT;
                const double gap = timediff(sol.time, sat_tk.last_time);
                const double predicted = 0.5 * (-lambda * od[j].D[0] + sat_tk.dopp_range_rate) * gap;
                if (std::fabs((cp_now - sat_tk.cp) - predicted) > prcopt.thresdop) arc_continues = false;
            }

            // hatch filter
            double psr_used = psr;
            int hatch_n = 1;
            if (config_.hatch_window > 0 && phase_valid) {
                if (arc_continues && sat_tk.hatch_n > 0) {
                    hatch_n = std::min(sat_tk.hatch_n + 1, config_.hatch_window);
                    psr_used = psr / hatch_n + (hatch_n - 1.0) / hatch_n * (sat_tk.hatch + cp_lambda - sat_tk.last_cp);
                }
                sat_tk.hatch_n = hatch_n;
                sat_tk.hatch = psr_used;
                sat_tk.last_cp = cp_lambda;
            }

            double ion_delay = 0.0, vion = 0.0, tro_delay = 0.0, vtrp = 0.0;
            ionocorr(sol.time, &nav, o.sat, rcv_llh, azel, prcopt.ionoopt, &ion_delay, &vion);
            ion_delay *= (FREQL1 / freq) * (FREQL1 / freq);  // ionocorr() reports the L1 delay
            tropcorr(sol.time, &nav, rcv_llh, azel, prcopt.tropopt, &tro_delay, &vtrp);
            // vion is reported at L1; it scales with delay^2, so rescaling to the tracked
            // frequency takes the fourth power.
            vion *= (FREQL1 / freq) * (FREQL1 / freq) * (FREQL1 / freq) * (FREQL1 / freq);
            const double varerr_term = rtklib::varerr(&prcopt, &o, azel[1], satsys(o.sat, nullptr));

            faster_lio::GnssPsrFactor::Params p;
            const Eigen::Vector3d le_arm_local = lio_state.quat.toRotationMatrix() * config_.lever_arm;
            p.le_arm_local = le_arm_local;
            p.sv_pos = sv_pos;
            p.svdt = svdt;
            p.ion_delay = ion_delay;
            p.tro_delay = tro_delay;
            const double code_var = od[j].Pstd[0] * od[j].Pstd[0];
            p.noise_var = sv_var + vmeas + code_var + vion + vtrp + varerr_term;
            p.weight = config_.psr_weight;
            p.psr_measure = psr_used;
            p.sys_idx = sys_idx;
            graph_.add(faster_lio::GnssPsrFactor(A(k), B(k), E(0), P(0), p, psr_noise));
            ++n_psr;

            const double cp = cp_lambda + svdt * CLIGHT;
            if (arc_continues) {
                faster_lio::GnssTdCpFactor::Params c;
                c.le_arm_local = le_arm_local;
                c.sv_pos_i = sat_tk.sv_pos;
                c.sv_pos_j = sv_pos;
                c.cp_measured = cp - sat_tk.cp;
                c.cp_weight = std::sin(azel[1]) / std::max(1e-3, config_.cp_std);
                c.sys_idx = sys_idx;
                graph_.add(
                    faster_lio::GnssTdCpFactor(A(sat_tk.node), A(k), B(sat_tk.node), B(k), E(0), P(0), c, cp_noise));
                ++n_cp;
            } else {
                ++n_slip;
            }

            sat_tk.node = k;
            sat_tk.last_time = sol.time;
            sat_tk.cp = cp;
            sat_tk.dopp_range_rate = -lambda * od[j].D[0];
            sat_tk.dopp_valid = dopp_valid;
            sat_tk.sv_pos = sv_pos;
        } // for sta

        if (k == 0) continue;

        const LioState& pls = EpochLio(k - 1);
        const double delta_t = timediff(sol.time, EpochSol(k - 1).time);

        graph_.add(
            faster_lio::LioOdomFactor(A(k - 1), A(k), lio_state.pos - pls.pos, gtsam::noiseModel::Isotropic::Sigma(3, config_.lio_pos_std)));
        graph_.add(faster_lio::GnssDdtSmoothFactor(C(k - 1), C(k), gtsam::noiseModel::Isotropic::Sigma(1, config_.ddt_std)));
        graph_.add(faster_lio::GnssDtDdtFactor(B(k - 1), B(k), C(k - 1), C(k), delta_t,
                                                   gtsam::noiseModel::Isotropic::Sigma(4, config_.dt_std)));

    } // for epoch

    LOG(INFO) << "graph: " << nn << " nodes, " << n_psr << " psr, " << n_cp << " tdcp, " << n_dtddt
              << " dt-ddt factors, " << graph_.size() << " factors total (" << n_slip
              << " arc restarts, carrier smoothing " << (config_.hatch_window > 0 ? "on" : "off") << ")";
}

void GnssProcessor::LogResiduals(const gtsam::Values& v, const char* tag) const {
    struct Acc {
        const char* name;
        const char* unit;
        size_t n = 0, over3 = 0;
        double sq = 0.0, max = 0.0;
        void add(double r) {
            ++n;
            sq += r * r;
            max = std::max(max, r);
            if (r > 3.0) ++over3;
        }
        double rms() const { return n ? std::sqrt(sq / n) : 0.0; }
    };

    Acc psr{"psr", "sigma"}, cp{"tdcp", "sigma"}, lio{"lio", "m"}, dtddt{"dt-ddt", "m"}, ddt{"ddt", "m/s"},
        prior{"priors", "mixed"};

    for (const auto& f : graph_) {
        // unwhitenedError() lives on NoiseModelFactor, not on the NonlinearFactor base.
        const auto* nf = dynamic_cast<const gtsam::NoiseModelFactor*>(f.get());
        if (!nf) continue;
        const double r = nf->unwhitenedError(v).norm();
        if (dynamic_cast<const faster_lio::GnssPsrFactor*>(f.get())) {
            psr.add(r);
        } else if (dynamic_cast<const faster_lio::GnssTdCpFactor*>(f.get())) {
            cp.add(r);
        } else if (dynamic_cast<const faster_lio::LioOdomFactor*>(f.get())) {
            lio.add(r);
        } else if (dynamic_cast<const faster_lio::GnssDtDdtFactor*>(f.get())) {
            dtddt.add(r);
        } else if (dynamic_cast<const faster_lio::GnssDdtSmoothFactor*>(f.get())) {
            ddt.add(r);
        } else {
            prior.add(r);
        }
    }

    for (const Acc* a : {&psr, &cp, &lio, &dtddt, &ddt, &prior}) {
        if (a->n == 0) continue;
        LOG(INFO) << "residuals " << tag << "  " << std::setw(7) << a->name << "  n " << std::setw(7) << a->n
                  << "  rms " << std::setw(9) << a->rms() << " " << a->unit << "  max " << std::setw(9) << a->max
                  << "  >3: " << a->over3 << " (" << (100.0 * a->over3 / a->n) << "%)";
    }
}

bool GnssProcessor::Optimize() {
    if (graph_.empty()) {
        LOG(ERROR) << "nothing to optimise";
        return false;
    }

    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = config_.max_iterations;
    params.relativeErrorTol = 1e-6;
    if (config_.verbose) params.setVerbosityLM("SUMMARY");

    const double err0 = graph_.error(values_);
    LogResiduals(values_, "before");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, values_, params);
    result_ = optimizer.optimize();
    const double err1 = graph_.error(result_);
    LogResiduals(result_, " after");
    LOG(INFO) << "LM: error " << err0 << " -> " << err1 << " in " << optimizer.iterations() << " iterations";

    return true;
}

bool GnssProcessor::SaveResults() const {
    if (output_dir.empty()) {
        LOG(WARNING) << "no --out_dir given, skipping result files";
        return true;
    }
    const size_t nn = nodes_.size();
    if (nn == 0) {
        LOG(ERROR) << "no optimised trajectory to save";
        return false;
    }


    std::ofstream f_ecef(output_dir + "/gnss_fgo.txt");
    std::ofstream f_local(output_dir + "/gnss_local.txt");
    if (!f_ecef || !f_local) {
        LOG(ERROR) << "failed to open output files under " << output_dir;
        return false;
    }

    f_ecef << "# unix_time(UTC) ecef_x ecef_y ecef_z lat_deg lon_deg height   (IMU)\n";
    f_local << "# unix_time(UTC) x y z qx qy qz qw   (TUM, LIO frame, IMU, position refined by GNSS)\n";


    for (auto* f : {&f_ecef, &f_local}) {
        *f << std::fixed << std::setprecision(6);
    }

    const Eigen::Vector3d anc_ecef_opt = result_.at<gtsam::Vector3>(E(0));
    const Eigen::Matrix3d R_ecef_local_opt = result_.at<gtsam::Rot3>(P(0)).matrix();

    for (size_t k = 0; k < nn; ++k) {
        const sol_t& sol = EpochSol(k);
        const LioState& ls = EpochLio(k);
        gtime_t utc_time = gpst2utc(sol.time);
        double utc_d = double(utc_time.time) + utc_time.sec;

        // optimized result transformed to  ECEF frame
        const Eigen::Vector3d pos_local_opt = result_.at<gtsam::Vector3>(A(k));
        const Eigen::Vector3d imu_ecef_fgo = R_ecef_local_opt * pos_local_opt + anc_ecef_opt;
        double llh_fgo[3];
        ecef2pos(imu_ecef_fgo.data(), llh_fgo);
        f_ecef << utc_d << " " << imu_ecef_fgo.transpose() << " " << llh_fgo[0] * R2D << " " << llh_fgo[1] * R2D << " "
              << llh_fgo[2] << "\n";

        // optimized result
        f_local << utc_d << " " << pos_local_opt.transpose() << " " << ls.quat.x() << " " << ls.quat.y() << " "
                << ls.quat.z() << " " << ls.quat.w() << "\n";
    }

    return true;
}

bool GnssProcessor::ProcessGnss() {
    if (!LoadLioStates()) return false;
    if (!RunPostpos()) return false;
    if (!ReadRinex()) return false;
    BuildNodes();
    if (!Alignment()) {
        LOG(ERROR) << "GNSS alignment failed";
        return false;
    }

    AddFactor();
    if (!Optimize()) return false;
    return SaveResults();
}

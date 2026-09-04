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

/// UNIX epoch, so that a ROS stamp can be turned into an RTKLIB gtime_t.
const double kEp1970[6] = {1970, 1, 1, 0, 0, 0};

gtime_t UnixToGtime(double unix_sec, bool utc) {
    gtime_t t = timeadd(epoch2time(kEp1970), unix_sec);
    return utc ? utc2gpst(t) : t;
}

double GtimeToUnix(gtime_t t) { return timediff(t, epoch2time(kEp1970)); }

/// R_enu_ecef at the geodetic position `pos` = {lat, lon, h}.  RTKLIB stores matrices
/// column-major, which is exactly Eigen's default, so the map needs no transpose.
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
    prcopt.nf = 1;  // the graph re-linearises L1/E1/B1 only
    prcopt.navsys = SYS_GPS | SYS_GLO | SYS_GAL | SYS_CMP | SYS_QZS;
    prcopt.glomodear = GLO_ARMODE_ON;  // prcopt_default asks for an external calibration we have none of
    // prcopt_default leaves the base coordinate at {0,0,0}, which silently yields an empty
    // solution in the relative modes; rnx2rtkp's post-processing default is used instead.
    prcopt.refpos = POSOPT_SINGLE;
    // Doppler slip threshold, shared with the TDCP track in AddFactor().
    prcopt.thresdop = 1.0;
    // ena stays off so postpos() does not mask: RTKLIB rejects a satellite whenever the SNR
    // is absent, which throws away every observation of a RINEX file that carries no S#
    // records.  The screening in EstimateClockBias()/AddFactor() reads the table regardless.
    SetSnrMask(25.0);
    prcopt.snrmask.ena[0] = prcopt.snrmask.ena[1] = 0;

    solopt = solopt_default;
    solopt.posf = SOLF_XYZ;    // ECEF, so readsolt() hands back sol.rr directly
    solopt.times = TIMES_GPST; // same time base as the observations
    solopt.timef = 0;          // week + tow
    solopt.timeu = 3;
    solopt.outhead = 1;        // readsolt() needs the header to pick the decoder
    solopt.outvel = 1;         // velocity round-trips through the file as well
    solopt.solstatic = 0;      // one line per epoch, even in a static mode
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
    std::vector<double> stamps;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double t, x, y, z, qx, qy, qz, qw;
        if (!(iss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;

        LioState s;
        s.time = UnixToGtime(t + config_.lio_time_offset, config_.lio_time_is_utc);
        s.pos << x, y, z;
        s.quat = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        lio_states.push_back(s);
        stamps.push_back(t + config_.lio_time_offset);
    }
    if (lio_states.size() < 2) {
        LOG(ERROR) << "lio trajectory has fewer than two poses: " << config_.lio_fname;
        return false;
    }

    LOG(INFO) << "loaded " << lio_states.size() << " lio states spanning " << (stamps.back() - stamps.front())
              << " s from " << config_.lio_fname;
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

    LOG(INFO) << "read " << obs.n << " observations, " << nav.n << " eph, " << nav.ng << " geph, " << nav.ns
              << " seph";
    if (obs.n <= 0 || nav.n + nav.ng <= 0) {
        LOG(ERROR) << "no usable GNSS data; check that RTKLIB is built with -DENAGLO/-DENAGAL/-DENACMP";
        return false;
    }
    return true;
}

bool GnssProcessor::RunPostpos() {
    if (config_.sol_fname.empty()) {
        config_.sol_fname = (config_.out_prefix.empty() ? std::string("gnss") : config_.out_prefix) + "_postpos.pos";
    }

    std::vector<std::string> files{config_.obs_fname, config_.nav_fname};
    files.insert(files.end(), config_.extra_files.begin(), config_.extra_files.end());
    std::vector<const char*> infile;
    for (const auto& f : files) infile.push_back(f.c_str());

    gtime_t ts{}, te{};  // zero epochs disable the time window filter
    filopt_t filopt{};
    LOG(INFO) << "postpos: mode " << prcopt.mode << ", " << infile.size() << " input files -> " << config_.sol_fname;

    const int ret = postpos(ts, te, 0.0, 0.0, &prcopt, &solopt, &filopt, infile.data(),
                            static_cast<int>(infile.size()), config_.sol_fname.c_str(), "", "");
    if (ret != 0) {
        LOG(ERROR) << "postpos failed with status " << ret;
        return false;
    }

    const char* solfiles[1] = {config_.sol_fname.c_str()};
    readsolt(solfiles, 1, ts, te, 0.0, 0, 0, &solbuf);
    if (solbuf.n <= 0) {
        LOG(ERROR) << "postpos produced no solution in " << config_.sol_fname;
        return false;
    }
    LOG(INFO) << "postpos produced " << solbuf.n << " epochs";
    if (!ReadRinex()) return false;
    return BuildNodes() >= 2;
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
    } // for obs

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

    const obsd_t* od = EpochObs(k);
    const sol_t& sol = EpochSol(k);
    double rcv_llh[3];
    ecef2pos(sol.rr, rcv_llh);

    std::vector<double> per_sys[SYSTOTAL];
    for (int j = 0; j < n; ++j) {
        const int sys_idx = SysToIdx(satsys(od[j].sat, nullptr));
        if (sys_idx < 0) continue;

        // satellite
        if (satexclude(od[j].sat, var[j], svh[j], &prcopt)) continue;
        double e[3], azel[2] = {0.0, PI / 2.0};
        const double r = geodist(&rs[j * 6], sol.rr, e);
        if (r <= 0.0) continue;
        // el
        if (satazel(rcv_llh, e, azel) < prcopt.elmin) continue;

        // snr
        snrmask_t mask = prcopt.snrmask;
        mask.ena[0] = 1;
        if (testsnr(0, 0, azel[1], od[j].SNR[0], &mask)) continue;

        const double freq = sat2freq(od[j].sat, od[j].code[0], &nav);
        if (freq <= 0.0) continue;
        double vmeas = 0.0;
        const double P = rtklib::prange(&od[j], &nav, &prcopt, &vmeas);
        if (P == 0.0) continue;

        // rescode()'s residual with the clock left in: v = P - (r - c*dts + dion + dtrp).
        double iono = 0.0, vion = 0.0, trop = 0.0, vtrp = 0.0;
        ionocorr(sol.time, &nav, od[j].sat, rcv_llh, azel, prcopt.ionoopt, &iono, &vion);
        iono *= (FREQL1 / freq) * (FREQL1 / freq);
        tropcorr(sol.time, &nav, rcv_llh, azel, prcopt.tropopt, &trop, &vtrp);
        per_sys[sys_idx].push_back(P - (r - dts[j * 2] * CLIGHT + iono + trop));
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
        local.push_back(ls.pos + ls.quat.toRotationMatrix() * config_.Tex_imu_r);
        ecef.emplace_back(sol.rr[0], sol.rr[1], sol.rr[2]);
    }

    Eigen::Vector3d span_min = local.front(), span_max = local.front();
    for (const auto& p : local) {
        span_min = span_min.cwiseMin(p);
        span_max = span_max.cwiseMax(p);
    }
    const double span = (span_max - span_min).norm();
    heading_observable_ = span > config_.align_min_span;
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
        cp_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(config_.robust_cp_k), cp_base);
    }

    auto lio_noise = gtsam::noiseModel::Isotropic::Sigma(3, config_.lio_pos_std);
    auto dt_noise = gtsam::noiseModel::Isotropic::Sigma(4, config_.dt_std);
    auto ddt_noise = gtsam::noiseModel::Isotropic::Sigma(1, config_.ddt_std);

    // --- initial values -----------------------------------------------------------------
    std::vector<std::array<double, SYSTOTAL>> dt_init(nn);
    for (size_t k = 0; k < nn; ++k)
        EstimateClockBias(k, dt_init[k].data());

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


    graph_.add(gtsam::PriorFactor<gtsam::Vector3>(E(0), gtsam::Vector3(anc_ecef_),
                                                 gtsam::noiseModel::Isotropic::Sigma(3, config_.prior_anchor_std)));
    // Heading is only pinned by the ranges when the platform actually travels.  On a short or
    // stationary run the ranges say nothing about yaw, so the prior has to hold it instead of
    // letting the optimiser wander freely inside a loose one.
    const double ext_rot_std =
        heading_observable_ ? config_.prior_ext_rot_std : std::min(config_.prior_ext_rot_std, 1e-4);
    graph_.add(gtsam::PriorFactor<gtsam::Rot3>(P(0), gtsam::Rot3(R_ecef_local_),
                                               gtsam::noiseModel::Isotropic::Sigma(3, ext_rot_std)));
    graph_.add(gtsam::PriorFactor<gtsam::Vector4>(
        B(0), values_.at<gtsam::Vector4>(B(0)), gtsam::noiseModel::Isotropic::Sigma(4, config_.prior_dt_std)));
    graph_.add(gtsam::PriorFactor<gtsam::Vector1>(
        C(0), gtsam::Vector1(ddt_init), gtsam::noiseModel::Isotropic::Sigma(1, config_.prior_ddt_std)));
    // A weak prior on the first node keeps the Hessian well conditioned; the global gauge is
    // already fixed by the anchor and rotation priors above.
    graph_.add(gtsam::PriorFactor<gtsam::Vector3>(A(0), values_.at<gtsam::Vector3>(A(0)),
                                                 gtsam::noiseModel::Isotropic::Sigma(3, config_.align_outlier)));

    // --- per-epoch factors --------------------------------------------------------------
    // sat -> what the TDCP factor needs from the last epoch on which that satellite was
    // tracked without a slip.  This is the one piece of cross-epoch state RTKLIB does not
    // keep for us, because postpos() has already thrown its own away.
    struct CpTrack {
        size_t node;
        double time;
        double cp;               ///< carrier phase, satellite clock and TGD corrected (m)
        double dopp_range_rate;  ///< -lambda * D at that epoch (m/s), for the slip test
        bool dopp_valid;
        Eigen::Vector3d sv_pos;
        Eigen::Vector3d Tex;
    };
    std::map<int, CpTrack> cp_track;

    std::vector<double> rs, dts, var;
    std::vector<int> svh;
    size_t n_psr = 0, n_cp = 0, n_dtddt = 0;
    for (size_t k = 0; k < nn; ++k) {
        const obsd_t* od = EpochObs(k);
        const sol_t& sol = EpochSol(k);
        const LioState& ls = EpochLio(k);
        const int n = nodes_[k].obs_n;
        rs.assign(6 * n, 0.0);
        dts.assign(2 * n, 0.0);
        var.assign(n, 0.0);
        svh.assign(n, 0);
        satposs(EpochSol(k).time, EpochObs(k), n, &nav, prcopt.sateph, rs.data(), dts.data(), var.data(), svh.data());
        double rcv_llh[3];
        ecef2pos(sol.rr, rcv_llh);

        // The factors expect the lever arm already rotated into the local frame; the LIO
        // attitude is treated as known here, only the position is estimated.
        const Eigen::Vector3d Tex_local = ls.quat.toRotationMatrix() * config_.Tex_imu_r;
        const double epoch_time = GtimeToUnix(sol.time);

        for (int j = 0; j < n; ++j) {
            const int sys_idx = SysToIdx(satsys(od[j].sat, nullptr));
            if (sys_idx < 0) continue;

            // satellite
            if (satexclude(od[j].sat, var[j], svh[j], &prcopt)) continue;
            double e[3], azel[2] = {0.0, PI / 2.0};
            if (geodist(&rs[j * 6], sol.rr, e) <= 0.0) continue;
            // el
            if (satazel(rcv_llh, e, azel) < prcopt.elmin) continue;

            // snr
            snrmask_t mask = prcopt.snrmask;
            mask.ena[0] = 1;
            if (testsnr(0, 0, azel[1], od[j].SNR[0], &mask)) continue;

            const double freq = sat2freq(od[j].sat, od[j].code[0], &nav);
            double vmeas = 0.0;
            const double psr = rtklib::prange(&od[j], &nav, &prcopt, &vmeas);
            if (psr == 0.0) continue;

            const double lambda = CLIGHT / freq;
            const bool dopp_valid = od[j].D[0] != 0.0f;

            faster_lio::GnssPsrDoppFactor::Params p;
            p.Tex_imu_r = Tex_local;
            p.sv_pos << rs[j * 6 + 0], rs[j * 6 + 1], rs[j * 6 + 2];
            p.svdt = dts[j * 2 + 0];
            // Broadcast ephemeris variance from satposs() plus the receiver code noise; the
            // factor divides by this and multiplies by sin^2(el), so it is a sigma in metres.
            const double code_std = od[j].Pstd[0] > 0.0f ? od[j].Pstd[0] : config_.psr_std;
            p.pr_uura = std::max(0.1, std::sqrt(std::max(var[j], 0.0) + vmeas + code_std * code_std));
            p.relative_sqrt_info = config_.relative_sqrt_info;
            p.freq = freq;
            p.psr_measure = psr;
            p.nav = &nav;
            p.time = sol.time;
            p.sat = od[j].sat;
            p.ionoopt = prcopt.ionoopt;
            p.tropopt = prcopt.tropopt;
            p.sys_idx = sys_idx;
            graph_.add(faster_lio::GnssPsrDoppFactor(A(k), B(k), E(0), P(0), p, psr_noise));
            ++n_psr;


            if (od[j].L[0] == 0.0 || std::fabs(od[j].L[0] * lambda) <= 100.0) {
                cp_track.erase(od[j].sat);
                continue;
            }
            // Only the satellite clock has to go: any constant per-satellite bias, the group
            // delay included, cancels when two epochs of the same arc are differenced.
            const double cp = od[j].L[0] * lambda + p.svdt * CLIGHT;
            const double range_rate = -lambda * od[j].D[0];

            auto it = cp_track.find(od[j].sat);
            bool usable = (od[j].LLI[0] & 0x3) == 0 && it != cp_track.end();
            if (usable) {
                const double gap = epoch_time - it->second.time;
                if (gap <= 0.0 || gap > config_.cp_max_gap) {
                    usable = false;
                } else if (dopp_valid && it->second.dopp_valid) {
                    // A cycle slip shows up as a carrier-phase increment that disagrees with
                    // the Doppler-integrated range change; both carry the receiver clock
                    // drift, so it cancels out of the comparison.  Without Doppler only the
                    // LLI flag and the arc gap above are left to catch a slip.
                    const double predicted = 0.5 * (range_rate + it->second.dopp_range_rate) * gap;
                    if (std::fabs((cp - it->second.cp) - predicted) > prcopt.thresdop) usable = false;
                }
            }

            if (usable) {
                faster_lio::GnssTdCpFactor::Params c;
                c.Tex_imu_r_i = it->second.Tex;
                c.Tex_imu_r_j = Tex_local;
                c.sv_pos_i = it->second.sv_pos;
                c.sv_pos_j = p.sv_pos;
                c.cp_measured = cp - it->second.cp;
                c.cp_weight = std::sin(azel[1]) / std::max(1e-3, config_.cp_std);
                c.sys_idx = sys_idx;
                graph_.add(faster_lio::GnssTdCpFactor(A(it->second.node), A(k), B(it->second.node), B(k), E(0), P(0), c,
                                                      cp_noise));
                ++n_cp;
            }

            cp_track[od[j].sat] = CpTrack{k, epoch_time, cp, range_rate, dopp_valid, p.sv_pos, Tex_local};
        }

        if (k == 0) continue;

        const LioState& pls = EpochLio(k - 1);
        const double delta_t = timediff(sol.time, EpochSol(k - 1).time);

        graph_.add(faster_lio::LioOdomFactor(A(k - 1), A(k), ls.pos - pls.pos, lio_noise));
        graph_.add(faster_lio::GnssDdtSmoothFactor(C(k - 1), C(k), ddt_noise));

        // Receivers periodically steer their clock by a whole millisecond.  Propagating the
        // bias across such a jump would drag the position with it, so the link is dropped and
        // only the drift stays connected.
        const double jump = std::fabs(dt_init[k][SYSGPS] - dt_init[k - 1][SYSGPS] - ddt_init * delta_t);
        if (jump < 0.5 * 1e-3 * CLIGHT) {
            graph_.add(faster_lio::GnssDtDdtFactor(B(k - 1), B(k), C(k - 1), C(k), delta_t, dt_noise));
            ++n_dtddt;
        } else {
            graph_.add(gtsam::PriorFactor<gtsam::Vector4>(B(k), values_.at<gtsam::Vector4>(B(k)),
                                                         gtsam::noiseModel::Isotropic::Sigma(4, config_.prior_dt_std)));
            LOG(WARNING) << "receiver clock jump of " << jump << " m before node " << k << "; dt link dropped";
        }
    }

    LOG(INFO) << "graph: " << nn << " nodes, " << n_psr << " psr, " << n_cp << " tdcp, " << n_dtddt
              << " dt-ddt factors, " << graph_.size() << " factors total";
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
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, values_, params);
    result_ = optimizer.optimize();
    const double err1 = graph_.error(result_);
    LOG(INFO) << "LM: error " << err0 << " -> " << err1 << " in " << optimizer.iterations() << " iterations";

    const Eigen::Matrix3d R_before = R_ecef_local_;
    anc_ecef_ = result_.at<gtsam::Vector3>(E(0));
    R_ecef_local_ = result_.at<gtsam::Rot3>(P(0)).matrix();
    LOG(INFO) << "extrinsic rotation moved "
              << gtsam::Rot3(R_before.transpose() * R_ecef_local_).axisAngle().second * R2D
              << " deg from the alignment estimate";

    opt_pos_.clear();
    opt_ecef_.clear();
    for (size_t k = 0; k < nodes_.size(); ++k) {
        const Eigen::Vector3d pos = result_.at<gtsam::Vector3>(A(k));
        opt_pos_.push_back(pos);
        opt_ecef_.push_back(
            R_ecef_local_ * (pos + EpochLio(k).quat.toRotationMatrix() * config_.Tex_imu_r) + anc_ecef_);
    }

    double anc_llh[3];
    ecef2pos(anc_ecef_.data(), anc_llh);
    LOG(INFO) << "refined anchor ecef " << anc_ecef_.transpose() << "  lla " << anc_llh[0] * R2D << " "
              << anc_llh[1] * R2D << " " << anc_llh[2];

    double sq = 0.0;
    for (size_t k = 0; k < nodes_.size(); ++k) {
        const sol_t& sol = EpochSol(k);
        sq += (opt_ecef_[k] - Eigen::Vector3d(sol.rr[0], sol.rr[1], sol.rr[2])).squaredNorm();
    }
    LOG(INFO) << "rms distance to the postpos solution: " << std::sqrt(sq / static_cast<double>(nodes_.size()))
              << " m";
    return true;
}

bool GnssProcessor::SaveResults() const {
    if (config_.out_prefix.empty()) {
        LOG(WARNING) << "no out_prefix given, skipping result files";
        return true;
    }
    if (opt_ecef_.size() != nodes_.size()) {
        LOG(ERROR) << "no optimised trajectory to save";
        return false;
    }

    std::ofstream f_ref(config_.out_prefix + "_rtklib.txt");
    std::ofstream f_fgo(config_.out_prefix + "_fgo.txt");
    std::ofstream f_loc(config_.out_prefix + "_local.txt");
    if (!f_ref || !f_fgo || !f_loc) {
        LOG(ERROR) << "failed to open output files with prefix " << config_.out_prefix;
        return false;
    }
    f_ref << "# unix_time ecef_x ecef_y ecef_z lat_deg lon_deg height stat nsat   (postpos solution)\n";
    f_fgo << "# unix_time ecef_x ecef_y ecef_z lat_deg lon_deg height\n";
    f_loc << "# unix_time x y z qx qy qz qw   (TUM, LIO frame, position refined by GNSS)\n";

    f_ref << std::fixed << std::setprecision(6);
    f_fgo << std::fixed << std::setprecision(6);
    f_loc << std::fixed << std::setprecision(6);

    for (size_t k = 0; k < nodes_.size(); ++k) {
        const sol_t& sol = EpochSol(k);
        const LioState& ls = EpochLio(k);
        const double t = GtimeToUnix(sol.time);

        double llh[3];
        ecef2pos(sol.rr, llh);
        f_ref << t << " " << sol.rr[0] << " " << sol.rr[1] << " " << sol.rr[2] << " " << llh[0] * R2D << " "
              << llh[1] * R2D << " " << llh[2] << " " << static_cast<int>(sol.stat) << " "
              << static_cast<int>(sol.ns) << "\n";

        double ecef_o[3] = {opt_ecef_[k].x(), opt_ecef_[k].y(), opt_ecef_[k].z()};
        ecef2pos(ecef_o, llh);
        f_fgo << t << " " << opt_ecef_[k].transpose() << " " << llh[0] * R2D << " " << llh[1] * R2D << " " << llh[2]
              << "\n";

        f_loc << t << " " << opt_pos_[k].transpose() << " " << ls.quat.x() << " " << ls.quat.y() << " " << ls.quat.z()
              << " " << ls.quat.w() << "\n";
    }

    LOG(INFO) << "wrote " << nodes_.size() << " epochs to " << config_.out_prefix << "_{rtklib,fgo,local}.txt";
    return true;
}

bool GnssProcessor::ProcessGnss() {
    if (!LoadLioStates()) return false;

    // RTKLIB solves the trajectory in whatever mode prcopt asks for; the factor graph below
    // always re-linearises the raw single-frequency observations on top of it.
    if (!RunPostpos()) return false;

    // align the postpos and lio trajectory
    if (!Alignment()) {
        LOG(ERROR) << "GNSS alignment failed";
        return false;
    }

    AddFactor();
    if (!Optimize()) return false;
    return SaveResults();
}

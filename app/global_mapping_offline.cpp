#include "gnss_processor.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

namespace fs = boost::filesystem;

DEFINE_string(obs_fname, "", "rover RINEX observation file");
DEFINE_string(nav_fname, "", "RINEX navigation file");
DEFINE_string(extra_files, "", "comma separated extra inputs for postpos (base obs, SP3, CLK, IONEX, ANTEX)");
DEFINE_string(lio_fname, "", "path to the lidar-inertial trajectory (TUM: t x y z qx qy qz qw)");
DEFINE_string(out_prefix, "", "results are written to <out_prefix>_{rtklib,fgo,local}.txt");
DEFINE_string(sol_fname, "", "postpos output file (default <out_prefix>_postpos.pos)");

DEFINE_string(mode, "single", "postpos positioning mode: single|dgps|kinematic|static|ppp-kine|ppp-static");
DEFINE_int32(nf, 1, "number of frequencies postpos should use");
DEFINE_string(sateph, "brdc", "satellite ephemeris: brdc|precise");
DEFINE_string(ionoopt, "brdc", "ionosphere option for postpos: off|brdc|iflc|tec");
DEFINE_string(tropopt, "saas", "troposphere option for postpos: off|saas|est");
DEFINE_string(refpos, "single", "base station coordinate in the relative modes: single|rinex|xyz");

DEFINE_bool(lio_time_is_utc, true, "lio stamps are ROS/UNIX (UTC) and need converting to GPST");
DEFINE_double(lio_time_offset, 0.0, "constant offset added to every lio stamp (s)");
DEFINE_double(match_tol, 0.05, "max |t_lio - t_obs| accepted as the same instant (s)");

DEFINE_double(lever_x, 0.0, "IMU->antenna lever arm x in the body frame (m)");
DEFINE_double(lever_y, 0.0, "IMU->antenna lever arm y in the body frame (m)");
DEFINE_double(lever_z, 0.0, "IMU->antenna lever arm z in the body frame (m)");

DEFINE_double(elev_mask_deg, 15.0, "elevation mask (deg)");
DEFINE_double(snr_mask_dbhz, 25.0, "SNR mask (dBHz), 0 disables");
DEFINE_int32(min_sat, 5, "minimum satellites in the postpos fix for an epoch to be used");
DEFINE_double(relative_sqrt_info, 10.0, "global scale on the pseudorange residuals");
DEFINE_double(cp_std, 0.02, "time-differenced carrier-phase noise (m)");
DEFINE_double(cp_max_gap, 3.0, "a longer hole in a satellite arc restarts the TDCP track (s)");
DEFINE_double(cp_slip_thres, 1.0, "|TDCP - Doppler prediction| above this is treated as a cycle slip (m)");
DEFINE_double(lio_pos_std, 0.05, "per-increment lio position noise (m)");
DEFINE_bool(align_yaw_only, true, "solve only heading during alignment (lio frame is gravity aligned)");
DEFINE_double(align_min_span, 5.0, "below this local trajectory extent the heading is unobservable (m)");
DEFINE_double(align_outlier, 15.0, "epochs further than this from the alignment fit are dropped (m)");
DEFINE_bool(robust, true, "wrap the GNSS residuals in a Cauchy m-estimator");
DEFINE_int32(max_iterations, 50, "Levenberg-Marquardt iteration cap");
DEFINE_bool(verbose, false, "print per-iteration optimiser output");

namespace {

/// Looks `key` up in `table`; returns `fallback` and complains when it is not there.
int Lookup(const std::string& key, const std::map<std::string, int>& table, const char* what, int fallback) {
    auto it = table.find(key);
    if (it != table.end()) return it->second;
    LOG(WARNING) << "unknown " << what << " '" << key << "', falling back to the default";
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    // postpos() reads from stdin when a path is empty, which silently hangs forever.
    if (FLAGS_obs_fname.empty() || FLAGS_nav_fname.empty() || FLAGS_lio_fname.empty()) {
        LOG(ERROR) << "usage: global_mapping_offline --obs_fname=<xxx.obs> --nav_fname=<xxx.nav> "
                      "--lio_fname=<traj_log.txt> [--out_prefix=<dir/run>] [--mode=single]";
        return 1;
    }

    GnssProcessor gnss_processor;
    GnssProcessor::Config& cfg = gnss_processor.config_;
    // Everything RTKLIB has an option for is set on prcopt itself, so postpos() and the
    // factor graph screen with the same masks and models.
    prcopt_t& prcopt = gnss_processor.prcopt;
    cfg.obs_fname = FLAGS_obs_fname;
    cfg.nav_fname = FLAGS_nav_fname;
    if (!FLAGS_extra_files.empty()) {
        boost::split(cfg.extra_files, FLAGS_extra_files, boost::is_any_of(","));
    }
    cfg.lio_fname = FLAGS_lio_fname;
    cfg.out_prefix = FLAGS_out_prefix.empty() ? fs::path(FLAGS_lio_fname).replace_extension("").string() + "_gnss"
                                              : FLAGS_out_prefix;
    cfg.sol_fname = FLAGS_sol_fname;

    for (const auto& f : {cfg.obs_fname, cfg.nav_fname, cfg.lio_fname}) {
        if (!fs::exists(f)) {
            LOG(ERROR) << "file not found: " << f;
            return 1;
        }
    }
    for (const auto& f : cfg.extra_files) {
        if (!fs::exists(f)) {
            LOG(ERROR) << "file not found: " << f;
            return 1;
        }
    }

    prcopt.mode = Lookup(FLAGS_mode,
                      {{"single", PMODE_SINGLE},
                       {"dgps", PMODE_DGPS},
                       {"kinematic", PMODE_KINEMA},
                       {"static", PMODE_STATIC},
                       {"ppp-kine", PMODE_PPP_KINEMA},
                       {"ppp-static", PMODE_PPP_STATIC}},
                         "mode", PMODE_SINGLE);
    prcopt.nf = FLAGS_nf;
    prcopt.sateph = Lookup(FLAGS_sateph, {{"brdc", EPHOPT_BRDC}, {"precise", EPHOPT_PREC}}, "sateph", EPHOPT_BRDC);
    prcopt.ionoopt = Lookup(FLAGS_ionoopt,
                            {{"off", IONOOPT_OFF}, {"brdc", IONOOPT_BRDC}, {"iflc", IONOOPT_IFLC},
                             {"tec", IONOOPT_TEC}},
                            "ionoopt", IONOOPT_BRDC);
    prcopt.tropopt = Lookup(FLAGS_tropopt, {{"off", TROPOPT_OFF}, {"saas", TROPOPT_SAAS}, {"est", TROPOPT_EST}},
                            "tropopt", TROPOPT_SAAS);
    prcopt.refpos = Lookup(FLAGS_refpos, {{"single", POSOPT_SINGLE}, {"rinex", POSOPT_RINEX}, {"xyz", POSOPT_POS_XYZ}},
                           "refpos", POSOPT_SINGLE);

    cfg.lio_time_is_utc = FLAGS_lio_time_is_utc;
    cfg.lio_time_offset = FLAGS_lio_time_offset;
    cfg.match_tol = FLAGS_match_tol;
    cfg.Tex_imu_r << FLAGS_lever_x, FLAGS_lever_y, FLAGS_lever_z;

    prcopt.elmin = FLAGS_elev_mask_deg * D2R;
    gnss_processor.SetSnrMask(FLAGS_snr_mask_dbhz);
    prcopt.thresdop = FLAGS_cp_slip_thres;
    cfg.min_sat = FLAGS_min_sat;
    cfg.relative_sqrt_info = FLAGS_relative_sqrt_info;
    cfg.cp_std = FLAGS_cp_std;
    cfg.cp_max_gap = FLAGS_cp_max_gap;
    cfg.lio_pos_std = FLAGS_lio_pos_std;
    cfg.align_yaw_only = FLAGS_align_yaw_only;
    cfg.align_min_span = FLAGS_align_min_span;
    cfg.align_outlier = FLAGS_align_outlier;
    cfg.robust = FLAGS_robust;
    cfg.max_iterations = FLAGS_max_iterations;
    cfg.verbose = FLAGS_verbose;

    return gnss_processor.ProcessGnss() ? 0 : 1;
}

#include "gnss_processor.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

namespace fs = boost::filesystem;

DEFINE_string(obs_fname, "", "rover RINEX observation file");
DEFINE_string(nav_fname, "", "RINEX navigation file");
DEFINE_string(extra_files, "", "comma separated extra inputs for postpos (base obs, SP3, CLK, IONEX, ANTEX)");
DEFINE_string(lio_fname, "", "path to the lidar-inertial trajectory (GPST t x y z qx qy qz qw)");
DEFINE_string(out_dir, "", "results directory");
DEFINE_string(sol_fname, "", "");

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
    gnss_processor.output_dir = FLAGS_out_dir;
    GnssProcessor::Config& cfg = gnss_processor.config_;
    cfg.obs_fname = FLAGS_obs_fname;
    cfg.nav_fname = FLAGS_nav_fname;
    if (!FLAGS_extra_files.empty()) {
        boost::split(cfg.extra_files, FLAGS_extra_files, boost::is_any_of(","));
    }
    cfg.lio_fname = FLAGS_lio_fname;
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
    return gnss_processor.ProcessGnss() ? 0 : 1;
}

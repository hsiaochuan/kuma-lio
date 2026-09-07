//
// Created by hsiaochuan on 2026/09/01.
//

#ifndef FASTER_LIO_GNSSPROCESSOR_H
#define FASTER_LIO_GNSSPROCESSOR_H

#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "rtklib_.h"
#include "rtklib_func.h"

struct LioState {
    gtime_t time;
    Eigen::Vector3d pos;
    Eigen::Quaterniond quat;
};

class GnssProcessor {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum SysIdx { SYSGPS = 0, SYSGLO = 1, SYSGAL = 2, SYSBDS = 3, SYSTOTAL = 4 };
    struct Config {
        std::string obs_fname;
        std::string nav_fname;
        std::vector<std::string> extra_files;
        std::string lio_fname;
        std::string sol_fname;

        double match_tol = 0.05;
        Eigen::Vector3d lever_arm = Eigen::Vector3d(-0.30845, 0.18636, -0.0284);  ///< IMU->ublox antenna, body frame

        int min_sat = 5;
        double cp_std = 0.01;
        double cp_max_gap = 3.0;
        int hatch_window = 100;

        double psr_weight = 1.0;
        double lio_pos_std = 0.01;
        double dt_std = 1.6;
        double ddt_std = 1.6;
        bool robust = true;
        double robust_psr_k = 1.0;
        double robust_cp_k = 1.0;

        // --- alignment ---
        bool align_yaw_only = true;
        double align_outlier = 15.0;
        int align_iterations = 5;
        double align_min_span = 5.0;

        int max_iterations = 100;
        bool verbose = false;
    };

    struct Node {
        int obs_i = 0;
        int obs_n = 0;
        int sol_i = 0;
        int lio_i = 0;
    };

    Config config_;

    nav_t nav{};
    obs_t obs{};
    prcopt_t prcopt{};
    solopt_t solopt{};
    solbuf_t solbuf{};

    std::vector<LioState> lio_states;
    std::vector<Node> nodes_;
    Eigen::Vector3d anc_ecef_ = Eigen::Vector3d::Zero();          ///< ECEF origin of the LIO frame
    Eigen::Matrix3d R_ecef_local_ = Eigen::Matrix3d::Identity();  ///< R_ecef_local
    GnssProcessor();
    ~GnssProcessor();

    void SetSnrMask(double snr_dbhz);

    bool LoadLioStates();
    bool RunPostpos();
    bool Alignment();
    void AddFactor();
    bool Optimize();
    bool SaveResults() const;
    bool ProcessGnss();

    const obsd_t* EpochObs(size_t k) const { return &obs.data[nodes_[k].obs_i]; }
    const sol_t& EpochSol(size_t k) const { return solbuf.data[nodes_[k].sol_i]; }
    const LioState& EpochLio(size_t k) const { return lio_states[nodes_[k].lio_i]; }

    void LogResiduals(const gtsam::Values& v, const char* tag) const;

    void SetupOptions();
    bool ReadRinex();
    int BuildNodes();
    void EstimateClockBias(size_t k, double* dt) const;
    bool FitTransform(const std::vector<Eigen::Vector3d>& local, const std::vector<Eigen::Vector3d>& ecef,
                      const std::vector<size_t>& idx, Eigen::Matrix3d& R, Eigen::Vector3d& t) const;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Values result_;
    std::string output_dir;
};

#endif  // FASTER_LIO_GNSSPROCESSOR_H

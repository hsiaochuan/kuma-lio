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

/// One lidar-inertial state, expressed in the LIO local frame (gravity aligned, arbitrary yaw/origin).
struct LioState {
    gtime_t time;
    Eigen::Vector3d pos;
    Eigen::Quaterniond quat;
};

/// Offline GNSS/LIO fusion.  RTKLIB's postpos() produces the reference trajectory in whatever
/// mode the caller configured (SPP, DGPS, RTK, PPP, ...), that trajectory is aligned with the
/// lidar-inertial one, and the raw observations are then re-linearised in a GTSAM factor graph
/// on top of the LIO increments.
class GnssProcessor {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum SysIdx { SYSGPS = 0, SYSGLO = 1, SYSGAL = 2, SYSBDS = 3, SYSTOTAL = 4 };

    /// Anything RTKLIB already has an option for is *not* repeated here: the positioning
    /// mode, the frequencies, the constellations, the ephemeris/iono/trop options, the
    /// elevation and SNR masks and the Doppler slip threshold all live in `prcopt` and are
    /// set there directly.  This holds only what the factor graph adds on top.
    struct Config {
        // --- input ---
        std::string obs_fname;                 ///< rover RINEX observation file
        std::string nav_fname;                 ///< RINEX navigation file
        std::vector<std::string> extra_files;  ///< base obs, SP3, CLK, IONEX, ANTEX ... for postpos()
        std::string lio_fname;                 ///< TUM trajectory produced by the lidar-inertial run
        std::string out_prefix;                ///< results go to <out_prefix>_{rtklib,fgo,local}.txt
        std::string sol_fname;                 ///< postpos() output; defaults to <out_prefix>_postpos.pos

        // --- time base ---
        bool lio_time_is_utc = true;   ///< LIO stamps are ROS/UNIX (UTC); convert them to GPST
        double lio_time_offset = 0.0;  ///< added to every LIO stamp before matching (s)
        double match_tol = 0.05;       ///< max |t_lio - t_obs| accepted as the same instant (s)

        // --- geometry ---
        Eigen::Vector3d Tex_imu_r = Eigen::Vector3d::Zero();  ///< IMU->antenna lever arm, body frame

        // --- measurement screening; the elevation and SNR masks are prcopt.elmin/snrmask ---
        int min_sat = 5;          ///< epochs whose postpos solution used fewer satellites are dropped
        double psr_std = 3.0;     ///< fallback code noise when RINEX carries no Pstd (m)
        double cp_std = 0.02;     ///< time-differenced carrier-phase noise (m)
        double cp_max_gap = 3.0;  ///< a longer hole in a satellite arc restarts the TDCP track (s)

        // --- factor weights ---
        double relative_sqrt_info = 10.0;  ///< global scale on the pseudorange residuals
        double lio_pos_std = 0.05;         ///< per-increment LIO position noise (m)
        double dt_std = 0.5;               ///< clock propagation noise (m)
        double ddt_std = 0.5;              ///< clock drift random walk noise (m/s)
        double prior_anchor_std = 5.0;     ///< prior on the ECEF anchor (m)
        double prior_ext_rot_std = 0.05;   ///< prior on the ECEF<-local rotation (rad)
        double prior_dt_std = 100.0;       ///< prior on the initial clock bias (m)
        double prior_ddt_std = 20.0;       ///< prior on the initial clock drift (m/s)
        bool robust = true;                ///< wrap the GNSS residuals in a Cauchy m-estimator
        double robust_psr_k = 1.0;
        double robust_cp_k = 1.0;

        // --- alignment ---
        bool align_yaw_only = true;   ///< the LIO frame is gravity aligned, so only yaw is unknown
        double align_outlier = 15.0;  ///< epochs further than this from the fit are dropped (m)
        int align_iterations = 5;
        double align_min_span = 5.0;  ///< below this trajectory extent yaw is not observable (m)

        int max_iterations = 50;
        bool verbose = false;
    };

    /// One graph node.  It stores cross-references only: the measurements stay in `obs.data`,
    /// the reference solution in `solbuf.data` and the attitude in `lio_states`.
    struct Node {
        int obs_i = 0;  ///< first obsd_t of this epoch in obs.data
        int obs_n = 0;  ///< number of obsd_t in this epoch
        int sol_i = 0;  ///< matching sol_t in solbuf.data
        int lio_i = 0;  ///< matching entry in lio_states
    };

    Config config_;

    // --- RTKLIB state, used as-is.  prcopt/solopt get their defaults from the constructor
    //     and are the one place the processing options live; change them directly before
    //     calling ProcessGnss().  The screening and the factors read them back, so postpos()
    //     and the graph always see the same masks and models. ---
    nav_t nav{};        ///< ephemerides, iono parameters, code biases
    obs_t obs{};        ///< every observation of the rover file
    prcopt_t prcopt{};  ///< processing options handed to postpos()
    solopt_t solopt{};  ///< solution output options handed to postpos()
    solbuf_t solbuf{};  ///< postpos() result, read back with readsolt()

    std::vector<LioState> lio_states;
    std::vector<Node> nodes_;

    // --- results ---
    Eigen::Vector3d anc_ecef_ = Eigen::Vector3d::Zero();          ///< ECEF origin of the LIO frame
    Eigen::Matrix3d R_ecef_local_ = Eigen::Matrix3d::Identity();  ///< R_ecef_local
    bool heading_observable_ = true;  ///< false when the trajectory is too short to fix yaw
    std::vector<Eigen::Vector3d> opt_pos_;  ///< optimised local position, one per graph node
    std::vector<Eigen::Vector3d> opt_ecef_;

    GnssProcessor();
    ~GnssProcessor();


    /// Fills every elevation band of prcopt.snrmask with the same threshold (dBHz).
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

   private:
    void SetupOptions();
    bool ReadRinex();
    int BuildNodes();
    void EstimateClockBias(size_t k, double* dt) const;
    bool FitTransform(const std::vector<Eigen::Vector3d>& local, const std::vector<Eigen::Vector3d>& ecef,
                      const std::vector<size_t>& idx, Eigen::Matrix3d& R, Eigen::Vector3d& t) const;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Values result_;
};

#endif  // FASTER_LIO_GNSSPROCESSOR_H

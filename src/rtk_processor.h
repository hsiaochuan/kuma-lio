//
// Created by hsiaochuan on 2026/09/08.
//

#ifndef FASTER_LIO_GNSS_SLIDE_WINDOW_H
#define FASTER_LIO_GNSS_SLIDE_WINDOW_H
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include "laser_mapping_param.h"
#include "rtklib_.h"
using gtsam::symbol_shorthand::R; // rotation
using gtsam::symbol_shorthand::P; // position
using gtsam::symbol_shorthand::D; // dt
using gtsam::symbol_shorthand::O; // ddt
using gtsam::symbol_shorthand::B; // bias
using gtsam::symbol_shorthand::G; // gravity

using gtsam::symbol_shorthand::E; // R_ecef_enu
using gtsam::symbol_shorthand::T; // t_ecef_enu
namespace faster_lio {
struct RtkState {
    gtime_t time;
    int64_t local_time;
    Eigen::Vector3d pos;
    Eigen::Quaterniond quat;
    Vec3 ba;
    Vec3 bg;
};
struct SatTrack {
    std::vector<int> obs_ids;
    std::vector<Eigen::Vector3d> sv_poss;
    std::vector<int64_t> local_times;
    double hatch = 0.0;
};
class RTKProcessor {
   public:
    RTKProcessor();
    std::shared_ptr<LaserMappingParam> param;
    nav_t nav{};
    obs_t obss{};
    prcopt_t prcopt{};
    solopt_t solopt{};
    std::unordered_map<int, SatTrack> sat_tracks_;

    std::shared_ptr<gtsam::ISAM2> isam;
    std::vector<std::int64_t> slw_times;
    std::unordered_map<std::int64_t, gtsam::FactorIndices> slw_factors;

    Mat3 R_ecef_enu;
    Vec3 t_ecef_enu;

    // for alignment
    std::vector<Vec3> locals;
    std::vector<Vec3> ecefs;
    double travel_len = 0.0;
    bool if_alignment = false;
    bool Alignment(Mat3& R_ecef_enu, Vec3& t_ecef_enu);
    void ProcessMeasurements(const int& obs_i, const RtkState& lio_state);
};

}  // namespace faster_lio

#endif  // FASTER_LIO_GNSS_SLIDE_WINDOW_H

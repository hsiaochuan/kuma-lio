// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <unordered_map>
#include <vector>

#include "common_lib.h"
#include "laser_mapping_param.h"

namespace kuma_lio {

// Receiver clock index of a satellite system: GPS (and QZSS, which keeps GPS time), GLO, GAL, BDS; -1 otherwise.
int GnssSysIdx(int sys);

// Local-to-ECEF alignment from antenna positions in the local frame and their SPP fixes: R_ecef_enu is the ENU
// frame at the mean fix turned by the yaw that best matches the horizontal tracks.  False below 3 fixes or when the
// horizontal spread is under `min_travel` [m].
bool GnssAlignment(Mat3& R_ecef_enu, Vec3& t_ecef_enu, const std::vector<Vec3>& locals,
                   const std::vector<Vec3>& ecefs, double min_travel);

// Fixed-anchor TDCP (FAST-LIVGO eq. 11) of one satellite: epoch k is the anchor, epoch k+1 the current one.
struct TdcpObs {
    int64_t time_ns;         // epoch k+1, local time
    int64_t anchor_time_ns;  // epoch k
    Vec3 sv_pos;             // satellite ECEF at k+1
    Vec3 sv_pos_anchor;      // satellite ECEF at k
    double sv_dt;            // satellite clock bias at k+1 [s]
    double sv_dt_anchor;     // satellite clock bias at k [s]
    double cp_diff;          // lambda * (L_{k+1} - L_k) [m]
    int sys_idx;             // which receiver clock the satellite reads
    double var;              // measurement variance [m^2]
    double sin_el;           // sine of the elevation at k+1, floored at 5 deg
};

// One GNSS epoch starting at obss.data[ei], seen at `local_time`: SPP into `sol`, the satellite arcs in `tracks`
// extended or restarted, and, when `emit_tdcp`, one TdcpObs per satellite whose arc continues appended to `tdcp`.
// False when SPP fails (nothing else is touched then).
bool ProcessGnssEpoch(const obs_t& obss, const nav_t& nav, const LaserMappingParam& param,
                      std::unordered_map<int, SatTrack>& tracks, int ei, int64_t local_time, bool emit_tdcp,
                      sol_t& sol, std::vector<TdcpObs>* tdcp);

}  // namespace kuma_lio

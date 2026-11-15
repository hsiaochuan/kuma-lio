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

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "eigen_type.h"
#include "rtklib_.h"
#include "so3_math.h"

namespace kuma_lio {

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
    double sin_el;           // sine of the elevation at k+1, floored at 5 deg
};

// Everything one GNSS epoch leaves for the front end: the SPP fix and clocks the alignment uses and the TDCP rows.
struct GnssEpoch {
    int64_t time_ns = 0;             // local time of the epoch = the frame end it closes
    Vec3 ecef_spp = Vec3::Zero();    // pntpos receiver position
    Vec4 clk_spp = Vec4::Zero();     // [m] pntpos receiver clocks per system (GPS, GLO, GAL, BDS)
    std::vector<TdcpObs> tdcp_obs;
};

// Placing the local frame in ECEF, ecef = R_ecef_enu * Rz(yaw) * local + t_ecef_enu, takes two steps.
// Step 1, from the first SPP fix: R_ecef_enu, the ENU frame there (columns E, N, U in ECEF); only the yaw about
// its up axis is left unknown, gravity already fixes the local frame's tilt.
inline Mat3 EnuRotation(const Vec3& ecef) {
    double llh[3];
    ecef2pos(ecef.data(), llh);
    Mat3 R_enu_ecef;
    xyz2enu(llh, R_enu_ecef.data());
    return R_enu_ecef.transpose();
}

// Step 2, once the track has spread horizontally: the yaw and t_ecef_enu that best match the antenna positions
// in the local frame to their SPP fixes, in the ENU frame `R_ecef_enu` of step 1.  False below 3 fixes or when
// the horizontal spread is under `min_travel` [m].
inline bool GnssAlignment(const Mat3& R_ecef_enu, const std::vector<Vec3>& locals, const std::vector<Vec3>& ecefs,
                          double min_travel, double* yaw, Vec3* t_ecef_enu) {
    CHECK(locals.size() == ecefs.size());
    int N = locals.size();
    if (N < 3) return false;
    // the yaw needs real horizontal spread; summed path length is fooled by jitter on a vertical climb
    double spread = 0.0;
    for (int i = 1; i < N; ++i) {
        spread = std::max(spread, (locals[i] - locals[0]).head<2>().norm());
    }
    if (spread < min_travel) return false;
    Vec3 mean_l(Vec3::Zero());
    Vec3 mean_e(Vec3::Zero());
    for (int i = 0; i < N; ++i) {
        mean_l += locals[i];
        mean_e += ecefs[i];
    }
    mean_l /= N;
    mean_e /= N;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const Vec3 s = locals[i] - mean_l;
        const Vec3 d = R_ecef_enu.transpose() * (ecefs[i] - mean_e);
        num += s.x() * d.y() - s.y() * d.x();
        den += s.x() * d.x() + s.y() * d.y();
    }
    *yaw = std::atan2(num, den);
    *t_ecef_enu = mean_e - R_ecef_enu * Eigen::AngleAxisd(*yaw, Vec3::UnitZ()).toRotationMatrix() * mean_l;
    return true;
}

// Step 3, optional: the yaw re-estimated from the TDCP displacements of the alignment window (GVINS' yaw alignment
// with TDCP in place of Doppler).  `window` holds every epoch since the first SPP fix by its time, with the antenna
// in the local frame.  Each TDCP row, linearized at the SPP fix, measures u . dr + the clock step of its system;
// the odometry gives dr = R_ecef_enu Rz(yaw) d, d the antenna displacement in the local frame, so the rows fix the
// yaw independently of t_ecef_enu and of any pseudorange bias.  The clock step of each system and epoch is removed
// as the weighted mean over its rows (its Schur complement).  Gauss-Newton from `yaw`; false (yaw untouched) with
// fewer than 20 usable rows.
using GnssAlignWindow = std::map<int64_t, std::pair<Vec3, GnssEpoch>>;

inline bool RefineYawTdcp(const Mat3& R_ecef_enu, const GnssAlignWindow& window, double tdcp_std, double* yaw) {
    struct Row {
        double r, J, w;
        int sys;
    };
    double yaw_new = *yaw;
    for (int iter = 0; iter < 10; ++iter) {
        const Mat3 Rz = Eigen::AngleAxisd(yaw_new, Vec3::UnitZ()).toRotationMatrix();
        const Mat3 dRz = Hat(Vec3::UnitZ()) * Rz;
        double H = 0.0, g = 0.0;
        int used = 0;
        const double gate = iter == 0 ? 1e9 : 0.3;  // [m] on the clock-free residual, from the second iteration
        for (const auto& [time_ns, entry] : window) {
            const GnssEpoch& epoch = entry.second;
            std::vector<Row> rows;
            for (const TdcpObs& o : epoch.tdcp_obs) {
                const auto anchor = window.find(o.anchor_time_ns);
                if (anchor == window.end()) continue;
                const Vec3 d = entry.first - anchor->second.first;
                double e1[3], e0[3];
                const double rho1 = geodist(o.sv_pos.data(), epoch.ecef_spp.data(), e1);
                const double rho0 = geodist(o.sv_pos_anchor.data(), epoch.ecef_spp.data(), e0);
                if (rho1 <= 0.0 || rho0 <= 0.0) continue;
                const Vec3 u = -Vec3(e1[0], e1[1], e1[2]);
                const double z = o.cp_diff + CLIGHT * (o.sv_dt - o.sv_dt_anchor) - (rho1 - rho0);
                const double s = tdcp_std / o.sin_el;
                rows.push_back({z - u.dot(R_ecef_enu * Rz * d), u.dot(R_ecef_enu * dRz * d), 1.0 / (2.0 * s * s), o.sys_idx});
            }
            for (int sys = 0; sys < 4; ++sys) {
                double sw = 0.0, swr = 0.0, swJ = 0.0;
                int n = 0;
                for (const Row& row : rows) {
                    if (row.sys != sys) continue;
                    sw += row.w;
                    swr += row.w * row.r;
                    swJ += row.w * row.J;
                    ++n;
                }
                if (n < 2) continue;  // one row only fixes its own clock step
                for (const Row& row : rows) {
                    if (row.sys != sys) continue;
                    const double r = row.r - swr / sw;
                    if (std::abs(r) > gate) continue;
                    const double J = row.J - swJ / sw;
                    H += row.w * J * J;
                    g += row.w * J * r;
                    ++used;
                }
            }
        }
        if (used < 20 || !(H > 0.0)) return false;
        const double step = g / H;
        yaw_new += step;
        if (std::abs(step) < 1e-7) break;
    }
    if (!std::isfinite(yaw_new)) return false;
    *yaw = yaw_new;
    return true;
}

}  // namespace kuma_lio

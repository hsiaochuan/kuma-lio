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

#include "gnss_epoch.h"

#include <glog/logging.h>

#include <algorithm>

#include "rtklib_func.h"

namespace kuma_lio {

int GnssSysIdx(int sys) {
    if (sys == SYS_GPS || sys == SYS_QZS) return 0;
    if (sys == SYS_GLO) return 1;
    if (sys == SYS_GAL) return 2;
    if (sys == SYS_CMP) return 3;
    return -1;
}

bool ProcessGnssEpoch(const obs_t& obss, const nav_t& nav, const LaserMappingParam& param,
                      std::unordered_map<int, SatTrack>& tracks, int ei, int64_t local_time, bool emit_tdcp,
                      sol_t& sol, std::vector<TdcpObs>* tdcp) {
    int n = 0;
    while (ei + n < obss.n && std::abs(timediff(obss.data[ei].time, obss.data[ei + n].time)) < DTTOL) ++n;

    char err_msg[128];
    if (!pntpos(obss.data + ei, n, &nav, &param.prcopt, &sol, nullptr, nullptr, err_msg)) {
        LOG(INFO) << "SPP failed: " << err_msg;
        return false;
    }

    // satellite pos
    std::vector<double> rs(6 * n, 0.0), dts(2 * n, 0.0), svar(n, 0.0);
    std::vector<int> svh(n, 0);
    satposs(sol.time, obss.data + ei, n, &nav, param.prcopt.sateph, rs.data(), dts.data(), svar.data(), svh.data());
    double rcv_llh[3];
    ecef2pos(sol.rr, rcv_llh);

    for (int si = 0; si < n; ++si) {
        int oi = ei + si;
        const obsd_t& obs = obss.data[oi];
        // filter
        if (satexclude(obs.sat, svar[si], svh[si], &param.prcopt)) continue;
        const int sys = satsys(obs.sat, nullptr);
        const int sys_idx = GnssSysIdx(sys);
        if (sys_idx < 0) continue;
        double azel[2] = {0.0, PI / 2.0};
        double e[3];
        if (geodist(&rs[si * 6], sol.rr, e) < 0.0) continue;
        if (satazel(rcv_llh, e, azel) < param.prcopt.elmin) continue;
        if (testsnr(0, 0, azel[1], obs.SNR[0], &param.prcopt.snrmask)) continue;

        const double freq = sat2freq(obs.sat, obs.code[0], &nav);
        if (freq <= 0.0) continue;
        const double lambda = CLIGHT / freq;

        // manage the sat track; phase and Doppler are signed, so only zero means missing
        bool psr_valid = obs.P[0] > 0.0;
        bool cp_valid = obs.L[0] != 0.0;
        bool dopp_valid = obs.D[0] != 0.0;
        bool obs_valid = psr_valid && cp_valid && dopp_valid;

        bool arc_continu = true;
        do {
            if (!obs_valid) {
                arc_continu = false;
                break;
            }
            if (tracks[obs.sat].obs_ids.empty()) {
                arc_continu = false;
                break;
            }
            int last_oi = tracks[obs.sat].obs_ids.back();
            const obsd_t& last_obs = obss.data[last_oi];
            if ((obs.LLI[0] & LLI_SLIP) || ((obs.LLI[0] & LLI_HALFC) != (last_obs.LLI[0] & LLI_HALFC))) {
                arc_continu = false;
                break;
            }
            // code change
            if (last_obs.code[0] != obs.code[0]) {
                arc_continu = false;
                break;
            }

            // time diff
            double dt = timediff(obs.time, last_obs.time);
            if (dt <= 0.0 || dt > param.max_gap + DTTOL) {
                arc_continu = false;
                break;
            }

            // geometry-free
            if (param.prcopt.thresslip > 0.0 && last_obs.L[1] > 0.0 && obs.L[1] > 0.0) {
                const double freq2 = sat2freq(obs.sat, obs.code[1], &nav);
                if (freq2 > 0.0) {
                    const double lambda2 = CLIGHT / freq2;
                    double last_gf = last_obs.L[0] * lambda - last_obs.L[1] * lambda2;
                    double cur_gf = obs.L[0] * lambda - obs.L[1] * lambda2;
                    if (std::abs(last_gf - cur_gf) > param.prcopt.thresslip) {
                        arc_continu = false;
                        break;
                    }
                }
            }

            // doppler-phase: the phase change against the Doppler integrated over the gap, in metres
            const double dpt = (-0.5) * lambda * (obs.D[0] + last_obs.D[0]) * dt;
            const double dcp = obs.L[0] * lambda - last_obs.L[0] * lambda;
            if (std::abs(dcp - dpt) > param.tdcp_dop_slip) {
                arc_continu = false;
                break;
            }
        } while (false);

        Eigen::Vector3d sv(rs[si * 6], rs[si * 6 + 1], rs[si * 6 + 2]);
        SatTrack& track = tracks[obs.sat];
        if (!arc_continu) {
            // restart the track from this one observation
            track.obs_ids.clear();
            track.sv_poss.clear();
            track.local_times.clear();
            track.sv_dts.clear();
        }
        track.obs_ids.push_back(oi);
        track.sv_poss.push_back(sv);
        track.local_times.push_back(local_time);
        track.sv_dts.push_back(dts[si * 2 + 0]);
        // the TDCP anchor only looks one epoch back: the arc goes on, its history does not
        if (track.obs_ids.size() > 2) {
            track.obs_ids.erase(track.obs_ids.begin());
            track.sv_poss.erase(track.sv_poss.begin());
            track.local_times.erase(track.local_times.begin());
            track.sv_dts.erase(track.sv_dts.begin());
        }

        const int N = static_cast<int>(track.obs_ids.size());

        if (emit_tdcp && arc_continu && N >= 2) {
            const int i = N - 2;
            const obsd_t& anchor_obs = obss.data[track.obs_ids[i]];
            TdcpObs tdcp_data;
            tdcp_data.time_ns = local_time;
            tdcp_data.anchor_time_ns = track.local_times[i];
            tdcp_data.sv_pos = sv;
            tdcp_data.sv_pos_anchor = track.sv_poss[i];
            tdcp_data.sv_dt = dts[si * 2 + 0];
            tdcp_data.sv_dt_anchor = track.sv_dts[i];
            tdcp_data.cp_diff = lambda * (obs.L[0] - anchor_obs.L[0]);
            tdcp_data.sys_idx = sys_idx;
            tdcp_data.sin_el = std::max(std::sin(azel[1]), std::sin(5.0 * D2R));
            tdcp->push_back(tdcp_data);
        }
    }  // for obs in one epoch

    return true;
}

}  // namespace kuma_lio

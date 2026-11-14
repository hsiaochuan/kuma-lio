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

#include "pgo/loop_detector.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace kuma_lio {
namespace pgo {
namespace {

constexpr int kMaxRadiusCands = 3;   // nearest radius candidates handed to the ICP
constexpr int kMaxOverlapCands = 3;  // best overlapping candidates handed to the ICP

std::vector<VOXEL_LOCATION> WorldVoxels(const std::vector<Eigen::Vector3f>& occupancy, const Eigen::Isometry3d& pose,
                                        double leaf) {
    std::unordered_set<VOXEL_LOCATION> seen;
    std::vector<VOXEL_LOCATION> out;
    for (const Eigen::Vector3f& p : occupancy) {
        const VOXEL_LOCATION loc(Vec3(pose * p.cast<double>()), leaf);
        if (seen.insert(loc).second) out.push_back(loc);
    }
    return out;
}

// Absolute heading difference between two poses, in degrees.
double YawDiffDeg(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
    const Mat3 Ra = a.linear(), Rb = b.linear();
    const double ya = std::atan2(Ra(1, 0), Ra(0, 0)), yb = std::atan2(Rb(1, 0), Rb(0, 0));
    double d = std::fmod(yb - ya + M_PI, 2.0 * M_PI);
    if (d < 0) d += 2.0 * M_PI;
    return std::abs(d - M_PI) * 180.0 / M_PI;
}

}  // namespace

LoopDetector::LoopDetector(const PgoParam& prm) : pgo_param_(prm), btc_db_(prm) {}

void LoopDetector::Reject(int from, int to, const char* reason, double value, const IcpResult* icp) {
    // keep the dump bounded on a long sequence; the early rejections are the informative ones
    if (rejections_.size() >= 10000) return;
    LoopRejection r;
    r.from = from;
    r.to = to;
    r.reason = reason;
    r.value = value;
    r.source = source_;
    if (icp) {
        r.matches = icp->matches;
        r.iters = icp->iters;
        r.step = icp->step;
        r.nn_median = icp->nn_median;
        r.src_n = icp->src_n;
        r.tar_n = icp->tar_n;
        r.seed_t = icp->seed_t;
    }
    rejections_.push_back(r);
}

bool LoopDetector::Eligible(const KeyFrames& kfs, int from, int to) const {
    return from + pgo_param_.loop_min_kf_gap <= to &&
           kfs[to].travel_len - kfs[from].travel_len >= pgo_param_.loop_min_travel;
}

LoopDetector::Candidates LoopDetector::BtcCandidates(const KeyFrames& kfs, const std::vector<Vec3>& kf_points,
                                                     const std::vector<PointCluster>& cell_pcs) {
    const int cur_kf_id = static_cast<int>(kfs.size()) - 1;
    CHECK_EQ(btc_db_.size(), cur_kf_id) << "Detect must see every keyframe once, in order";
    BtcFrame btc_frame = BuildBtcFrame(kf_points, kfs[cur_kf_id].planes, cell_pcs, pgo_param_);
    const auto btc_cands = btc_db_.Query(btc_frame, [&](int id) { return Eligible(kfs, id, cur_kf_id); });
    btc_db_.Add(std::move(btc_frame));  // after the query, so a keyframe never finds itself

    // paper VI-B: of the candidates RANSAC accepted, the one whose planes overlap best.  The candidates' trees are
    // built here, one per keyframe for good; the overlaps then in parallel, the best picked in candidate order.
    const int num_cands = static_cast<int>(btc_cands.size());
    if (static_cast<int>(plane_trees_.size()) < cur_kf_id + 1) plane_trees_.resize(cur_kf_id + 1);
    for (const BtcCandidate& c : btc_cands) {
        if (!plane_trees_[c.frame_id]) plane_trees_[c.frame_id] = MakePlaneTree(kfs[c.frame_id].planes);
    }
    std::vector<double> overlaps(num_cands);
#pragma omp parallel for schedule(dynamic, 1) if (num_cands > 1)
    for (int i = 0; i < num_cands; ++i) {
        const int id = btc_cands[i].frame_id;
        overlaps[i] = PlaneOverlap(kfs[cur_kf_id].planes, kfs[id].planes, plane_trees_[id].get(), btc_cands[i].T,
                                   pgo_param_);
    }
    int best = -1;
    double best_overlap = 0.0;
    for (int i = 0; i < num_cands; ++i) {
        if (overlaps[i] > best_overlap) best_overlap = overlaps[i], best = i;
    }
    if (best < 0) return {};
    if (best_overlap <= pgo_param_.btc_min_overlap) {
        source_ = LoopSource::kBtc;
        Reject(btc_cands[best].frame_id, cur_kf_id, "btc_overlap", best_overlap);
        return {};
    }
    return {{btc_cands[best].frame_id, btc_cands[best].T}};
}

LoopDetector::Candidates LoopDetector::RadiusCandidates(const KeyFrames& kfs) const {
    const int cur_kf_id = static_cast<int>(kfs.size()) - 1;
    const KeyFrame& cur_kf = kfs[cur_kf_id];

    std::vector<std::pair<double, int>> nears;
    for (int kf_id = 0; kf_id < static_cast<int>(kfs.size()) - 1; ++kf_id) {
        if (!Eligible(kfs, kf_id, cur_kf_id)) continue;
        const Vec3 offset = cur_kf.pose.translation() - kfs[kf_id].pose.translation();
        if (std::abs(offset.z()) > pgo_param_.loop_max_dz) continue;
        const double r = offset.norm();
        if (r > pgo_param_.loop_radius) continue;
        if (YawDiffDeg(kfs[kf_id].pose, cur_kf.pose) > pgo_param_.loop_max_yaw_diff) continue;
        nears.emplace_back(r, kf_id);
    }
    std::sort(nears.begin(), nears.end());
    if (static_cast<int>(nears.size()) > kMaxRadiusCands) nears.resize(kMaxRadiusCands);

    Candidates cands;
    // seeded with what the odometry currently believes the relative pose is
    for (const auto& nb_kf : nears) cands.push_back({nb_kf.second, kfs[nb_kf.second].pose.inverse() * cur_kf.pose});
    return cands;
}

LoopDetector::Candidates LoopDetector::OverlapCandidates(const KeyFrames& kfs) {
    const int cur_kf_id = static_cast<int>(kfs.size()) - 1;
    const KeyFrame& cur_kf = kfs[cur_kf_id];
    const double leaf = pgo_param_.overlap_voxel;

    // the index holds the keyframes before the current one at their present poses: after a solve moved
    // any of them it is built again, then the keyframes sealed since are added
    for (size_t i = 0; i < indexed_poses_.size(); ++i) {
        if (kfs[i].pose.matrix() != indexed_poses_[i].matrix()) {
            overlap_index_.clear();
            indexed_poses_.clear();
            break;
        }
    }
    for (int id = static_cast<int>(indexed_poses_.size()); id < cur_kf_id; ++id) {
        for (const VOXEL_LOCATION& loc : WorldVoxels(kfs[id].occupancy, kfs[id].pose, leaf)) overlap_index_[loc].push_back(id);
        indexed_poses_.push_back(kfs[id].pose);
    }

    const std::vector<VOXEL_LOCATION> cur_voxels = WorldVoxels(cur_kf.occupancy, cur_kf.pose, leaf);
    if (cur_voxels.empty()) return {};
    std::vector<int> hits(cur_kf_id, 0);
    for (const VOXEL_LOCATION& loc : cur_voxels) {
        const auto it = overlap_index_.find(loc);
        if (it == overlap_index_.end()) continue;
        for (const int id : it->second) ++hits[id];
    }
    // (-overlap, id): the largest overlap first, the older keyframe on a tie
    std::vector<std::pair<double, int>> ranked;
    double best_low = 0.0;
    int best_low_id = -1;
    for (int id = 0; id < cur_kf_id; ++id) {
        if (hits[id] == 0 || !Eligible(kfs, id, cur_kf_id)) continue;
        const double overlap = static_cast<double>(hits[id]) / cur_voxels.size();
        if (overlap >= pgo_param_.overlap_min) {
            ranked.emplace_back(-overlap, id);
        } else if (overlap > best_low) {
            best_low = overlap, best_low_id = id;
        }
    }
    if (ranked.empty()) {
        // the near miss, for tuning overlap_min
        if (best_low_id >= 0) {
            source_ = LoopSource::kOverlap;
            Reject(best_low_id, cur_kf_id, "overlap_low", best_low);
        }
        return {};
    }
    std::sort(ranked.begin(), ranked.end());
    if (static_cast<int>(ranked.size()) > kMaxOverlapCands) ranked.resize(kMaxOverlapCands);

    Candidates cands;
    // seeded with what the odometry currently believes the relative pose is
    for (const auto& r : ranked) cands.push_back({r.second, kfs[r.second].pose.inverse() * cur_kf.pose});
    return cands;
}

bool LoopDetector::Detect(const KeyFrames& kfs, const std::vector<Vec3>& kf_points,
                          const std::vector<PointCluster>& cell_pcs, LoopEdge* loop) {
    // BTC runs on every keyframe while enabled: its database has to see each one, in order
    if (pgo_param_.btc_enable && Verify(kfs, BtcCandidates(kfs, kf_points, cell_pcs), LoopSource::kBtc, loop))
        return true;
    if (pgo_param_.radius_enable && Verify(kfs, RadiusCandidates(kfs), LoopSource::kRadius, loop)) return true;
    // the overlap index catches up with the keyframes it skipped on its next call
    return pgo_param_.overlap_enable && Verify(kfs, OverlapCandidates(kfs), LoopSource::kOverlap, loop);
}

bool LoopDetector::Verify(const KeyFrames& kfs, const Candidates& cands, LoopSource source, LoopEdge* loop) {
    const int cur_kf_id = static_cast<int>(kfs.size()) - 1;
    const KeyFrame& cur_kf = kfs[cur_kf_id];
    source_ = source;
    for (const Candidate& cand : cands) {
        const KeyFrame& ki = kfs[cand.kf_id];
        Eigen::Isometry3d T_from_to = cand.T_from_to;
        const IcpResult icp = IcpNormal(cur_kf.planes, ki.planes, pgo_param_, &T_from_to);
        if (!icp.ok) {
            // Three quite different failures, kept apart because they call for different fixes:
            // starved means the seed pose was too far off for the gates to find anything,
            // unsettled means the Gauss-Newton never took a small enough step, and the last two
            // mean it aligned but the result is not trustworthy.
            if (icp.starved) {
                Reject(cand.kf_id, cur_kf_id, "icp_starved", icp.matches, &icp);
            } else if (!icp.converged) {
                Reject(cand.kf_id, cur_kf_id, "icp_unsettled", icp.step, &icp);
            } else if (icp.matches < pgo_param_.icp_min_match) {
                Reject(cand.kf_id, cur_kf_id, "icp_matches", icp.matches, &icp);
            } else {
                Reject(cand.kf_id, cur_kf_id, "icp_eigval", icp.eigval, &icp);
            }
            continue;
        }

        // how far the loop would move the current pose, against how far we drove to get here
        const Vec3 p_loop = ki.pose * T_from_to.translation();
        const double drift = (p_loop - cur_kf.pose.translation()).norm();
        const double span = cur_kf.travel_len - ki.travel_len;
        if (drift / span >= pgo_param_.ratio_drift) {
            Reject(cand.kf_id, cur_kf_id, "ratio_drift", drift / span);
            continue;
        }

        loop->from = cand.kf_id;
        loop->to = cur_kf_id;
        loop->T_from_to = T_from_to;
        loop->drift = drift;
        loop->span = span;
        loop->eigval = icp.eigval;
        loop->source = source;
        return true;
    }
    return false;
}

}  // namespace pgo
}  // namespace kuma_lio

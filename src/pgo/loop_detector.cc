#include "pgo/loop_detector.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

namespace faster_lio {
namespace pgo {
namespace {

/// Absolute heading difference between two poses, in degrees.
double YawDiffDeg(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
    const Mat3 Ra = a.linear(), Rb = b.linear();
    const double ya = std::atan2(Ra(1, 0), Ra(0, 0)), yb = std::atan2(Rb(1, 0), Rb(0, 0));
    double d = std::fmod(yb - ya + M_PI, 2.0 * M_PI);
    if (d < 0) d += 2.0 * M_PI;
    return std::abs(d - M_PI) * 180.0 / M_PI;
}

}  // namespace

LoopDetector::LoopDetector(const PgoParam& prm) : prm_(prm), btc_db_(prm) {}

void LoopDetector::Reject(int from, int to, const char* reason, double value, const IcpResult* icp) {
    // keep the dump bounded on a long sequence; the early rejections are the informative ones
    if (rejections_.size() >= 10000) return;
    LoopRejection r;
    r.from = from;
    r.to = to;
    r.reason = reason;
    r.value = value;
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
    return from + prm_.loop_min_kf_gap <= to && kfs[to].jour - kfs[from].jour >= prm_.loop_min_travel;
}

LoopDetector::Candidates LoopDetector::BtcCandidates(const KeyFrames& kfs, const std::vector<Vec3>& cloud,
                                                     const PlaneCells& cells) {
    const int cur_kf_id = static_cast<int>(kfs.size()) - 1;
    CHECK_EQ(btc_db_.size(), cur_kf_id) << "Detect must see every keyframe once, in order";
    BtcFrame btc_frame = BuildBtcFrame(cloud, cells, prm_);
    const auto btc_cands = btc_db_.Query(btc_frame, [&](int id) {
        return Eligible(kfs, id, cur_kf_id);
    });
    btc_db_.Add(std::move(btc_frame));  // after the query, so a keyframe never finds itself

    // paper VI-B: of the candidates RANSAC accepted, the one whose planes overlap best
    int best = -1;
    double best_overlap = 0.0;
    for (size_t i = 0; i < btc_cands.size(); ++i) {
        const double overlap = PlaneOverlap(kfs[cur_kf_id].planes, kfs[btc_cands[i].frame_id].planes, btc_cands[i].T, prm_);
        if (overlap > best_overlap) best_overlap = overlap, best = static_cast<int>(i);
    }
    if (best < 0) return {};
    if (best_overlap <= prm_.btc_min_overlap) {
        Reject(btc_cands[best].frame_id, cur_kf_id, "btc_overlap", best_overlap);
        return {};
    }
    return {{btc_cands[best].frame_id, btc_cands[best].T}};
}

LoopDetector::Candidates LoopDetector::RadiusCandidates(const KeyFrames& kfs) const {
    const int cur = static_cast<int>(kfs.size()) - 1;
    const KeyFrame& kc = kfs[cur];
    // A brute-force scan is cheaper here than a kd-tree: the candidate set is already cut down by
    // the index and travel gates, and rebuilding a tree on every keyframe costs more than the
    // few thousand distance tests it would save.
    std::vector<std::pair<double, int>> near;
    for (int i = 0; i < cur; ++i) {
        if (!Eligible(kfs, i, cur)) continue;
        const Vec3 d = kc.pose.translation() - kfs[i].pose.translation();
        if (std::abs(d.z()) > prm_.loop_max_dz) continue;
        const double r = d.norm();
        if (r > prm_.loop_radius) continue;
        if (YawDiffDeg(kfs[i].pose, kc.pose) > prm_.loop_max_yaw_diff) continue;
        near.emplace_back(r, i);
    }
    std::sort(near.begin(), near.end());
    if (static_cast<int>(near.size()) > prm_.loop_max_candidates) near.resize(prm_.loop_max_candidates);

    Candidates out;
    // seeded with what the odometry currently believes the relative pose is
    for (const auto& n : near) out.push_back({n.second, kfs[n.second].pose.inverse() * kc.pose});
    return out;
}

bool LoopDetector::Detect(const KeyFrames& kfs, const std::vector<Vec3>& cloud, const PlaneCells& cells,
                          LoopEdge* loop) {
    const int cur_kf_id = static_cast<int>(kfs.size()) - 1;
    const KeyFrame& cur_kf = kfs[cur_kf_id];
    const Candidates cands = prm_.loop_search == LoopSearch::kBtc ? BtcCandidates(kfs, cloud, cells) : RadiusCandidates(kfs);

    for (const Candidate& cand : cands) {
        const KeyFrame& ki = kfs[cand.id];
        Eigen::Isometry3d T_from_to = cand.T_from_to;
        const IcpResult icp = IcpNormal(cur_kf.planes, ki.planes, prm_, &T_from_to);
        if (!icp.ok) {
            // Three quite different failures, kept apart because they call for different fixes:
            // starved means the seed pose was too far off for the gates to find anything,
            // unsettled means the Gauss-Newton never took a small enough step, and the last two
            // mean it aligned but the result is not trustworthy.
            if (icp.starved) {
                Reject(cand.id, cur_kf_id, "icp_starved", icp.matches, &icp);
            } else if (!icp.converged) {
                Reject(cand.id, cur_kf_id, "icp_unsettled", icp.step, &icp);
            } else if (icp.matches < prm_.icp_min_match) {
                Reject(cand.id, cur_kf_id, "icp_matches", icp.matches, &icp);
            } else {
                Reject(cand.id, cur_kf_id, "icp_eigval", icp.eigval, &icp);
            }
            continue;
        }

        // how far the loop would move the current pose, against how far we drove to get here
        const Vec3 p_loop = ki.pose * T_from_to.translation();
        const double drift = (p_loop - cur_kf.pose.translation()).norm();
        const double span = cur_kf.jour - ki.jour;
        if (drift / span >= prm_.ratio_drift) {
            Reject(cand.id, cur_kf_id, "ratio_drift", drift / span);
            continue;
        }

        loop->from = cand.id;
        loop->to = cur_kf_id;
        loop->T_from_to = T_from_to;
        loop->drift = drift;
        loop->span = span;
        loop->eigval = icp.eigval;
        return true;
    }
    return false;
}

}  // namespace pgo
}  // namespace faster_lio

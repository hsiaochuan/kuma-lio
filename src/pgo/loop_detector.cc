#include "pgo/loop_detector.h"

#include <algorithm>
#include <cmath>

#include "pgo/icp_normal.h"
#include "pgo/plane_cloud.h"

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

std::vector<PlanePoint> LoopDetector::SubmapPlanes(const KeyFrames& kfs, int center, int lo, int hi) const {
    if (lo == center && hi == center) return kfs[center].planes;
    const Eigen::Isometry3d inv = kfs[center].pose.inverse();
    std::vector<Vec3> merged;
    for (int k = lo; k <= hi; ++k) {
        const Eigen::Isometry3d d = inv * kfs[k].pose;
        for (const Vec3& p : kfs[k].cloud) merged.push_back(d * p);
    }
    return ExtractPlanes(DownSample(merged, prm_.kf_ds_voxel), prm_);
}

bool LoopDetector::Detect(const KeyFrames& kfs, int cur, LoopEdge* out) {
    if (cur < prm_.loop_min_kf_gap) return false;
    const KeyFrame& kc = kfs[cur];

    // A brute-force scan is cheaper here than a kd-tree: the candidate set is already cut down by
    // the index and travel gates, and rebuilding a tree on every keyframe costs more than the
    // few thousand distance tests it would save.
    std::vector<std::pair<double, int>> cands;
    for (int i = 0; i + prm_.loop_min_kf_gap <= cur; ++i) {
        const KeyFrame& ki = kfs[i];
        if (kc.jour - ki.jour < prm_.loop_min_travel) continue;
        const Vec3 d = kc.pose.translation() - ki.pose.translation();
        if (std::abs(d.z()) > prm_.loop_max_dz) continue;
        const double r = d.norm();
        if (r > prm_.loop_radius) continue;
        if (YawDiffDeg(ki.pose, kc.pose) > prm_.loop_max_yaw_diff) continue;
        cands.emplace_back(r, i);
    }
    if (cands.empty()) return false;

    std::sort(cands.begin(), cands.end());
    const int n = std::min<int>(cands.size(), prm_.loop_max_candidates);
    const int K = std::max(0, prm_.loop_submap_kf);
    const std::vector<PlanePoint> src = SubmapPlanes(kfs, cur, std::max(0, cur - K), cur);
    for (int k = 0; k < n; ++k) {
        const int id = cands[k].second;
        const KeyFrame& ki = kfs[id];

        // seed with what the odometry currently believes the relative pose is
        Eigen::Isometry3d T_from_to = ki.pose.inverse() * kc.pose;
        // the old end may not reach into the stretch the current end's submap covers
        const int hi = std::min(id + K, std::max(id, cur - K - 1));
        const std::vector<PlanePoint> tar = SubmapPlanes(kfs, id, std::max(0, id - K), hi);
        const IcpResult icp = IcpNormal(src, tar, prm_, &T_from_to);
        if (!icp.ok) {
            // Three quite different failures, kept apart because they call for different fixes:
            // starved means the seed pose was too far off for the gates to find anything,
            // unsettled means the Gauss-Newton never took a small enough step, and the last two
            // mean it aligned but the result is not trustworthy.
            if (icp.starved) {
                Reject(id, cur, "icp_starved", icp.matches, &icp);
            } else if (!icp.converged) {
                Reject(id, cur, "icp_unsettled", icp.step, &icp);
            } else if (icp.matches < prm_.icp_min_match) {
                Reject(id, cur, "icp_matches", icp.matches, &icp);
            } else {
                Reject(id, cur, "icp_eigval", icp.eigval, &icp);
            }
            continue;
        }

        // how far the loop would move the current pose, against how far we drove to get here
        const Vec3 p_loop = ki.pose * T_from_to.translation();
        const double drift = (p_loop - kc.pose.translation()).norm();
        const double span = kc.jour - ki.jour;
        if (drift / span >= prm_.ratio_drift) {
            Reject(id, cur, "ratio_drift", drift / span);
            continue;
        }

        out->from = id;
        out->to = cur;
        out->T_from_to = T_from_to;
        out->drift = drift;
        out->span = span;
        out->eigval = icp.eigval;
        return true;
    }
    return false;
}

}  // namespace pgo
}  // namespace faster_lio

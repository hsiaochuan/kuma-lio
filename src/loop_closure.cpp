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
#include "loop_closure.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

#include "so3_math.h"
#include "timer.h"

namespace kuma_lio {

LoopClosure::LoopClosure(const pgo::PgoParam& prm, LocalBA* lba, double fixed_var)
    : prm_(prm), lba_(lba), pgo_(prm), fixed_var_(fixed_var) {
    CHECK(prm.keep_kf_points) << "the coupled loop closure needs the keyframe clouds";
    LOG(INFO) << "loop closure coupled to the front end: map_kfs " << prm_.map_kfs << " reload_radius "
              << prm_.reload_radius << " kf_ds_voxel " << prm_.kf_ds_voxel;
}

bool LoopClosure::PushScan(const LocalBA::MargScan& scan, double time_s) {
    std::vector<Vec3> pts;
    pts.reserve(scan.points.size());
    for (const LbaPoint& pt : scan.points) pts.push_back(pt.p);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = scan.x.R;
    pose.translation() = scan.x.p;
    return pgo_.AddScan(time_s, pose, pts);
}

bool LoopClosure::AddScan(const LocalBA::MargScan& scan, double time_s, Eigen::Isometry3d* dx) {
    Timer timer;
    if (!PushScan(scan, time_s) || !pgo_.Solve(dx)) return false;
    Feedback(*dx);
    LOG(INFO) << "PGO: feedback dx t " << dx->translation().norm() << " m r "
              << LogMat(Mat3(dx->linear())).norm() * 180.0 / M_PI << " deg at scan " << scan.id << " in "
              << timer.ElapsedMiniSeconds() << " ms";
    return true;
}

void LoopClosure::Finish(const std::vector<LocalBA::MargScan>& window, double time_offset_s) {
    // no solve between them: the window's poses all share one frame, and the final solve covers them
    for (const LocalBA::MargScan& scan : window) PushScan(scan, time_offset_s + NS_TO_S(scan.time_ns));
    pgo_.Finish();
    LOG(INFO) << "PGO: finished with the " << window.size() << " window scans, " << pgo_.nodes().size() << " nodes";
}

void LoopClosure::MergeKfPoints(const pgo::KeyFrame& kf, std::vector<Vec3>* out_points) const {
    const Mat3 R = kf.pose.linear();
    const Vec3 t = kf.pose.translation();
    out_points->reserve(out_points->size() + kf.cloud.size());
    for (const Eigen::Vector3f& p : kf.cloud)
        out_points->push_back(R * p.cast<double>() + t);
}

void LoopClosure::Feedback(const Eigen::Isometry3d& dx) {
    // the window continues from the solved pose of the newest node
    lba_->ApplyCorrection(dx);

    // Voxel-SLAM: the newest init_num keyframes at their solved poses, plus the scans sealed into no
    // keyframe yet (nodes with solved poses too), make the new local map; the window re-enters on its own
    const pgo::KeyFrames& kfs = pgo_.keyframes();
    const pgo::ScanNodes& scans = pgo_.nodes();
    const int n_kf = static_cast<int>(kfs.size());
    const int history_end = std::max(0, n_kf - prm_.map_kfs);
    std::vector<Vec3> near_points;
    for (int i = history_end; i < n_kf; ++i)
        MergeKfPoints(kfs[i], &near_points);
    const auto& pending_scans = pgo_.pending_scans();
    for (size_t i = 0; i < pending_scans.size(); ++i) {
        const Eigen::Isometry3d& pose = scans[scans.size() - pending_scans.size() + i].pose;
        for (const Vec3& p : pending_scans[i])
            near_points.push_back(pose * p);
    }
    lba_->Rebuild(near_points, fixed_var_);

    // every older keyframe is out of the map again and may be reloaded near a revisit
    loaded_.assign(history_end, false);
}

void LoopClosure::ReloadKeyframe(const Vec3& pos) {
    const pgo::KeyFrames& kfs = pgo_.keyframes();
    int best = -1;
    double best_dist = prm_.reload_radius;
    for (int i = 0; i < static_cast<int>(loaded_.size()); ++i) {
        if (loaded_[i]) continue;
        const double dist = (kfs[i].pose.translation() - pos).norm();
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    if (best < 0) return;
    std::vector<Vec3> kf_points;
    MergeKfPoints(kfs[best], &kf_points);
    lba_->AddFixedPoints(kf_points, fixed_var_);
    loaded_[best] = true;
    LOG(INFO) << "PGO: keyframe " << best << " reloaded, " << kf_points.size() << " points at " << best_dist << " m";
}

void LoopClosure::SaveDebug(const std::string& dir) const { pgo_.SaveDebug(dir); }

}  // namespace kuma_lio

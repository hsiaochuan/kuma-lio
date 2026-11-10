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

#include <vector>

#include "pgo/btc.h"
#include "pgo/icp_normal.h"
#include "pgo/pgo_types.h"

namespace kuma_lio {
namespace pgo {

class LoopDetector {
   public:
    explicit LoopDetector(const PgoParam& prm);

    // Tries to close a loop at kfs.back(); call it once for every keyframe, in order.  `kf_points`
    // and `cell_pcs` are that keyframe's merged scan in its body frame and the point statistics of
    // its plane cells, which only BTC reads.  True when `loop` was filled.
    bool Detect(const KeyFrames& kfs, const std::vector<Vec3>& kf_points, const std::vector<PointCluster>& cell_pcs,
                LoopEdge* loop);

    const std::vector<LoopRejection>& rejections() const { return rejections_; }
    // BTC descriptors of keyframe `id`; null with btc_enable off.
    const BtcFrame* btc_frame(int id) const { return id < btc_db_.size() ? &btc_db_.frame(id) : nullptr; }

   private:
    struct Candidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int kf_id;
        Eigen::Isometry3d T_from_to;  // ICP seed
    };
    using Candidates = std::vector<Candidate, Eigen::aligned_allocator<Candidate>>;

    // Far enough back, by keyframes and by path length, to count as a loop.
    bool Eligible(const KeyFrames& kfs, int from, int to) const;
    Candidates BtcCandidates(const KeyFrames& kfs, const std::vector<Vec3>& kf_points,
                             const std::vector<PointCluster>& cell_pcs);
    Candidates RadiusCandidates(const KeyFrames& kfs) const;
    // Runs the ICP and drift checks on `cands` in order; true at the first one that passes.
    bool Verify(const KeyFrames& kfs, const Candidates& cands, LoopSource source, LoopEdge* loop);
    void Reject(int from, int to, const char* reason, double value, const IcpResult* icp = nullptr);

    PgoParam pgo_param_;
    BtcDatabase btc_db_;
    std::vector<LoopRejection> rejections_;
    LoopSource source_ = LoopSource::kBtc;  // layer being verified, stamped on each rejection
};

}  // namespace pgo
}  // namespace kuma_lio

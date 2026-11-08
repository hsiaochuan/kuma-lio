//
// Loop closure at the newest keyframe, in two layers: BTC candidates first and, when none of them
// is accepted (or btc_enable is off), radius-search candidates.  Each is verified by IcpNormal on
// the two keyframes' plane cells and the paper's drift-over-travel ratio; the first candidate that
// passes both becomes the loop.
//

#ifndef FASTER_LIO_PGO_LOOP_DETECTOR_H
#define FASTER_LIO_PGO_LOOP_DETECTOR_H

#include <vector>

#include "pgo/btc.h"
#include "pgo/icp_normal.h"
#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

class LoopDetector {
   public:
    explicit LoopDetector(const PgoParam& prm);

    /// Tries to close a loop at kfs.back(); call it once for every keyframe, in order.  `cloud` and
    /// `cells` are that keyframe's merged scan and plane cells in its body frame, which only BTC
    /// reads.  True when `out` was filled.
    bool Detect(const KeyFrames& kfs, const std::vector<Vec3>& kf_points, const PlaneCells& cells, LoopEdge* loop);

    const std::vector<LoopRejection>& rejections() const { return rejections_; }
    /// BTC descriptors of keyframe `id`; null with btc_enable off.
    const BtcFrame* btc_frame(int id) const { return id < btc_db_.size() ? &btc_db_.frame(id) : nullptr; }

   private:
    struct Candidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id;
        Eigen::Isometry3d T_from_to;  ///< ICP seed
    };
    using Candidates = std::vector<Candidate, Eigen::aligned_allocator<Candidate>>;

    /// Far enough back, by keyframes and by path length, to count as a loop.
    bool Eligible(const KeyFrames& kfs, int from, int to) const;
    Candidates BtcCandidates(const KeyFrames& kfs, const std::vector<Vec3>& kf_points, const PlaneCells& cells);
    Candidates RadiusCandidates(const KeyFrames& kfs) const;
    /// Runs the ICP and drift checks on `cands` in order; true at the first one that passes.
    bool Verify(const KeyFrames& kfs, const Candidates& cands, LoopSource source, LoopEdge* loop);
    void Reject(int from, int to, const char* reason, double value, const IcpResult* icp = nullptr);

    PgoParam pgo_param_;
    BtcDatabase btc_db_;
    std::vector<LoopRejection> rejections_;
    LoopSource source_ = LoopSource::kBtc;  ///< layer being verified, stamped on each rejection
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_LOOP_DETECTOR_H

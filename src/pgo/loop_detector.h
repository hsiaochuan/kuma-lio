//
// Loop candidates from a radius search over the keyframe positions, verified by IcpNormal and the
// paper's drift-over-travel ratio.  This replaces BTC: no descriptor, no retrieval database, so a
// candidate only ever comes from the current pose estimate being close to an old keyframe.
//

#ifndef FASTER_LIO_PGO_LOOP_DETECTOR_H
#define FASTER_LIO_PGO_LOOP_DETECTOR_H

#include <vector>

#include "pgo/icp_normal.h"
#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

class LoopDetector {
   public:
    explicit LoopDetector(const PgoParam& prm) : prm_(prm) {}

    /// Tries to close a loop at keyframe `cur`; true when `out` was filled.
    bool Detect(const KeyFrames& kfs, int cur, LoopEdge* out);

    const std::vector<LoopRejection>& rejections() const { return rejections_; }

   private:
    void Reject(int from, int to, const char* reason, double value, const IcpResult* icp = nullptr);
    /// Planes of keyframes [lo, hi] merged into keyframe `center`'s frame; just its own planes when
    /// the range is that keyframe alone.
    std::vector<PlanePoint> SubmapPlanes(const KeyFrames& kfs, int center, int lo, int hi) const;

    PgoParam prm_;
    std::vector<LoopRejection> rejections_;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_LOOP_DETECTOR_H

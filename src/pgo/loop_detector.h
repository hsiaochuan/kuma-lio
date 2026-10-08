#ifndef FASTER_LIO_PGO_LOOP_DETECTOR_H
#define FASTER_LIO_PGO_LOOP_DETECTOR_H

#include <string>
#include <vector>

#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

/// Radius search over the OPTIMIZED keyframe positions, verified by a
/// coarse-to-fine PCL GICP against a submap built around the historic
/// keyframe.  All gating happens before a factor is ever created.
class LoopDetector {
   public:
    struct Rejection {
        int id_from = -1;
        int id_to = -1;
        std::string reason;
        double fitness = 0.0;
        double overlap = 0.0;
        double corr_trans = 0.0;
        double corr_rot = 0.0;
    };

    explicit LoopDetector(const PgoParam& prm) : prm_(prm) {}

    /// Tries to close a loop at keyframe `cur_id`; true when `edge` was filled.
    /// `kfs` must already carry up-to-date pose_opt values.  Meant to be called
    /// with increasing `cur_id`: GICP is attempted at most once every
    /// loop_kf_gap keyframes, whether or not the attempt succeeds, because a
    /// rejected candidate costs the same as an accepted one.
    bool Detect(const std::vector<KeyFrame::Ptr>& kfs, int cur_id, LoopEdge* edge);

    const std::vector<Rejection>& rejections() const { return rejections_; }

   private:
    std::vector<int> FindCandidates(const std::vector<KeyFrame::Ptr>& kfs, int cur_id) const;
    PgoCloudPtr BuildSubmap(const std::vector<KeyFrame::Ptr>& kfs, int center_id) const;
    bool Verify(const PgoCloudPtr& src_world, const PgoCloudPtr& tgt_world,
                Eigen::Isometry3d* T_icp, double* fitness, double* overlap) const;

    PgoParam prm_;
    int last_attempt_id_ = -1000000;
    std::vector<Rejection> rejections_;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_LOOP_DETECTOR_H

//
// Point-to-plane Gauss-Newton between two plane clouds, after Voxel-SLAM's icp_normal
// (loop_refine.hpp).  Not PCL ICP and not a bundle adjustment: the correspondences are plane
// centres, and what decides acceptance is the paper's appendix B degeneration criterion on
// sum(n n^T) rather than a residual score, which is what keeps perceptual aliasing out.
//

#ifndef FASTER_LIO_PGO_ICP_NORMAL_H
#define FASTER_LIO_PGO_ICP_NORMAL_H

#include <vector>

#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

struct IcpResult {
    bool ok = false;         ///< converged through both phases, enough matches, non-degenerate
    bool converged = false;  ///< reached the tight correspondence phase and settled there
    int matches = 0;
    double eigval = 0.0;  ///< smallest eigenvalue of sum(n n^T) over the matched normals
    int iters = 0;        ///< iterations actually run
    double step = 0.0;    ///< norm of the last translation update [m]
    bool starved = false;  ///< bailed out because a pass found fewer than six correspondences
    double nn_median = 0.0;  ///< median ungated 1-NN distance on the first pass [m]
    int src_n = 0, tar_n = 0;
    double seed_t = 0.0;  ///< translation of the seed transform [m]
};

/// Refines `T_tar_src` so that `src` lands on `tar`, both in their own keyframe frames.
IcpResult IcpNormal(const std::vector<PlanePoint>& src, const std::vector<PlanePoint>& tar,
                    const PgoParam& prm, Eigen::Isometry3d* T_tar_src);

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_ICP_NORMAL_H

//
// BTC place recognition (Yuan et al., "BTC: A Binary and Triangle Combined Descriptor for 3D Place
// Recognition", T-RO 2024), rewritten from hku-mars/btc_descriptor for the offline pose graph.
//
//   keyframe cloud --> plane cells --> merged reference planes --> height-encoded image per plane
//     --> keypoints (5x5 block maxima, line filter, 3D NMS) --> triangles of near keypoints
//     --> hash table keyed by the quantized side lengths
//
// A query votes for the old keyframes whose triangles have nearly the same sides and nearly the
// same vertex bit strings, then RANSACs a rigid transform over the triangle vertex pairs of each
// top-voted keyframe, and finally scores it by plane overlap.  Everything is in the keyframe's own
// body frame, so the transform is T_old_cur, the same seed the loop ICP takes.
//
// Deliberate differences from upstream, none of which changes what the method is:
//  - a reference plane is oriented to face the sensor, not to point along +z, so the height band
//    lies on the sensor's side for any body axis convention (i2Nav's body is z-down);
//  - plane cells are merged by union-find instead of two passes of pairwise id propagation;
//  - non-maximum suppression breaks a tie by list order, upstream dropped both keypoints;
//  - the RANSAC transform is refit on all inlier vertex pairs instead of taken from one triangle;
//  - no fixed-size vote array (upstream overflowed past 20000 frames), no ROS, no Ceres.
//

#ifndef FASTER_LIO_PGO_BTC_H
#define FASTER_LIO_PGO_BTC_H

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "common_lib.h"
#include "pgo/plane_cloud.h"
#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

/// A keypoint and its binary descriptor: which height layers above the reference plane hold points.
struct BtcKeypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 p = Vec3::Zero();  ///< body frame of its keyframe
    uint64_t bits = 0;      ///< one bit per layer of btc_layer_height
    int count = 0;          ///< popcount(bits), the pixel intensity
};

/// Three keypoints, sides sorted l1 <= l2 <= l3; v[k] is the vertex opposite side 2-k, so the
/// vertices of two triangles with the same sides correspond index by index.
struct BtcTriangle {
    Eigen::Vector3f side = Eigen::Vector3f::Zero();  ///< [m]
    std::array<int, 3> v{{0, 0, 0}};                 ///< indices into the frame's keypoints
};

struct BtcFrame {
    std::vector<BtcKeypoint, Eigen::aligned_allocator<BtcKeypoint>> keypoints;
    std::vector<BtcTriangle> triangles;
};

/// Builds the descriptors of one keyframe from its merged cloud and the plane cells of that cloud.
BtcFrame BuildBtcFrame(const std::vector<Vec3>& cloud, const PlaneCells& cells, const PgoParam& prm);

/// Fraction of `src` planes that, moved by T_tar_src, land on a `tar` plane (paper eq. 8-9).
double PlaneOverlap(const std::vector<PlanePoint>& src, const std::vector<PlanePoint>& tar,
                    const Eigen::Isometry3d& T_tar_src, const PgoParam& prm);

struct BtcCandidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int frame_id = -1;
    int votes = 0;    ///< triangles that matched by sides and bit strings
    int inliers = 0;  ///< of those, how many agree with the RANSAC transform
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  ///< T_frame_query
};

class BtcDatabase {
   public:
    explicit BtcDatabase(const PgoParam& prm) : prm_(prm) {}

    /// Top-voted old frames for which `eligible(frame)` holds, best first, each with a transform.
    std::vector<BtcCandidate, Eigen::aligned_allocator<BtcCandidate>> Query(
        const BtcFrame& q, const std::function<bool(int)>& eligible) const;

    /// Appends frame number size(); call it after querying with the same frame.
    void Add(BtcFrame f);

    int size() const { return static_cast<int>(btc_frames_.size()); }
    const BtcFrame& frame(int id) const { return btc_frames_[id]; }

   private:
    struct Entry {
        int frame;
        int tri;  ///< index into frames_[frame].triangles
    };

    PgoParam prm_;
    std::vector<BtcFrame> btc_frames_;
    std::unordered_map<VOXEL_LOCATION, std::vector<Entry>> table_;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_BTC_H

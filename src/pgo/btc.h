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

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "common_lib.h"
#include "pgo/plane_cloud.h"
#include "pgo/pgo_types.h"

namespace kuma_lio {
namespace pgo {

// A keypoint and its binary descriptor: which height layers above the reference plane hold points.
struct BtcKeypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 p = Vec3::Zero();  // body frame of its keyframe
    uint64_t bits = 0;      // one bit per layer of btc_layer_height
    int count = 0;          // popcount(bits), the pixel intensity
};

// Three keypoints, sides sorted l1 <= l2 <= l3; v[k] is the vertex opposite side 2-k, so the
// vertices of two triangles with the same sides correspond index by index.
struct BtcTriangle {
    Eigen::Vector3f side_lens = Eigen::Vector3f::Zero();  // [m]
    std::array<int, 3> v{{0, 0, 0}};                 // indices into the frame's keypoints
};

struct BtcFrame {
    std::vector<BtcKeypoint, Eigen::aligned_allocator<BtcKeypoint>> keypoints;
    std::vector<BtcTriangle> triangles;
};

// Builds the descriptors of one keyframe from its merged cloud and the plane cells of that cloud,
// with their point statistics `cell_pcs` (ExtractPlaneCells).
BtcFrame BuildBtcFrame(const std::vector<Vec3>& kf_points, const PlaneCells& cells,
                       const std::vector<PointCluster>& cell_pcs, const PgoParam& prm);

// Fraction of `src` planes that, moved by T_tar_src, land on a `tar` plane (paper eq. 8-9).
double PlaneOverlap(const PlaneCells& src, const PlaneCells& tar, const Eigen::Isometry3d& T_tar_src,
                    const PgoParam& prm);

struct BtcCandidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int frame_id = -1;
    int votes = 0;    // triangles that matched by sides and bit strings
    int inliers = 0;  // of those, how many agree with the RANSAC transform
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // T_frame_query
};

class BtcDatabase {
   public:
    explicit BtcDatabase(const PgoParam& prm) : prm_(prm) {}

    // Top-voted old frames for which `eligible(frame)` holds, best first, each with a transform.
    std::vector<BtcCandidate, Eigen::aligned_allocator<BtcCandidate>> Query(
        const BtcFrame& q, const std::function<bool(int)>& eligible) const;

    // Appends frame number size(); call it after querying with the same frame.
    void Add(BtcFrame btc_f);

    int size() const { return static_cast<int>(btc_frames_.size()); }
    const BtcFrame& frame(int id) const { return btc_frames_[id]; }

   private:
    struct FrameTri {
        int frame;
        int tri;  // index into frames_[frame].triangles
    };

    PgoParam prm_;
    std::vector<BtcFrame> btc_frames_;
    std::unordered_map<VOXEL_LOCATION, std::vector<FrameTri>> sl_table_;
};

}  // namespace pgo
}  // namespace kuma_lio

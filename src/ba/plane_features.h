#ifndef FASTER_LIO_BA_PLANE_FEATURES_H
#define FASTER_LIO_BA_PLANE_FEATURES_H

//
// Plane association for the batch BA (run_ba).  The same three-layer grid and plane test as the local BA
// map in local_ba.cpp (RefitRoot): layer-0 voxels of voxel_size, each cut into 4 x 4 x 4 finest cells, planes
// found by walking the layers (voxel_size, /2, /4) from the top -- a node flat enough becomes a plane, one
// that is not is split, and the walk goes on only below the split nodes.
//
// What differs is that this map is batch only: every scan is in the "window", so there are no fixed
// (marginalized) points, no sliding and no point-to-plane queries.  What the BA needs is which scans see which
// plane, as body-frame point clusters (BALM3 eq. 2).  Raw points are never stored: a scan's points are folded
// into the cluster of the cell they land in as soon as they are added.
//

#include <omp.h>

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <vector>

#include "ba/ba_data.h"
#include "common_lib.h"
#include "point_cluster.h"

namespace faster_lio {
namespace ba {

struct VoxelMapOptions {
    double voxel_size = 1.0;         ///< [m] layer-0 voxel; layers 1 and 2 halve it
    double plane_eigen_ratio = 0.1;  ///< plane when lambda_0 / lambda_1 is below this; scale free
    double min_plane_size = 0.05;    ///< smallest in-plane standard deviation of a plane, as a fraction of the
                                     ///< cell edge of its own layer
    int min_points = 10;             ///< a node with this many points or fewer is neither a plane nor split
};

/// The points scan `xid` put on one plane, in that scan's body frame.
struct FrameCluster {
    IndexT xid = 0;  ///< pose id, a key of SfmData::poses (a dense index inside the BundleAdjuster)
    PointCluster clu;
};

/// One plane: the scans that see it.  Sorted by pose id, one entry per pose.
struct PlaneFeature {
    std::vector<FrameCluster> clus;
    int layer = 0;
    double coeff = 1.0;  ///< weight of the plane in the BA cost, set by BundleAdjuster::Load
};

class PlaneMap {
   public:
    static constexpr int kLayers = 3;  ///< fixed depth of the voxel grid
    static constexpr int kCells = 64;  ///< finest cells per layer-0 voxel, 4 per axis

    explicit PlaneMap(const VoxelMapOptions& param) : param_(param) {}

    /// Add the points of a scan at pose `id`, given in its body frame.
    void AddScan(IndexT id, const Mat3& R, const Vec3& p, const std::vector<Vec3>& body_pts) {
        for (const Vec3& pb : body_pts) {
            const Vec3 pw = R * pb + p;
            Cell& cell = CellAt(pw);
            if (cell.obs.empty() || cell.obs.back().xid != id) {
                cell.obs.emplace_back();
                cell.obs.back().xid = id;
            }
            cell.obs.back().clu.Push(pb);
        }
    }

    size_t NumVoxels() const { return map_.size(); }

    std::vector<PlaneFeature> ExtractPlanes(const Poses& poses) const {
        std::vector<const Root*> roots;
        roots.reserve(map_.size());
        for (const auto& kv : map_) roots.push_back(&kv.second);

        std::vector<std::vector<PlaneFeature>> roots_features(roots.size());
#pragma omp parallel for schedule(dynamic, 16)
        for (long i = 0; i < (long)roots.size(); i++) ExtractRoot(*roots[i], poses, &roots_features[i]);

        std::vector<PlaneFeature> features;
        for (auto& v : roots_features) std::move(v.begin(), v.end(), std::back_inserter(features));
        return features;
    }

   private:
    struct Cell {
        int idx = 0;                    ///< 16 ix + 4 iy + iz, each in [0, 4)
        std::vector<FrameCluster> obs;  ///< sorted by scan
    };
    struct Root {  ///< a layer-0 voxel
        std::vector<Cell> cells;
    };

    // the same indexing as LocalBA::NodeOf / LocalBA::CellIndex
    static int NodeOf(int idx, int layer) {
        const int s = kLayers - 1 - layer;
        return (((idx >> 4) >> s) << (2 * layer)) | ((((idx >> 2) & 3) >> s) << layer) | ((idx & 3) >> s);
    }
    int CellIndex(const Vec3& pw, const VOXEL_LOCATION& loc) const {
        const Vec3 corner(loc.x * param_.voxel_size, loc.y * param_.voxel_size, loc.z * param_.voxel_size);
        const Eigen::Vector3i c = ((pw - corner) * (4.0 / param_.voxel_size)).cast<int>().cwiseMax(0).cwiseMin(3);
        return 16 * c.x() + 4 * c.y() + c.z();
    }

    Cell& CellAt(const Vec3& pw) {
        const VOXEL_LOCATION loc(pw, param_.voxel_size);
        const int idx = CellIndex(pw, loc);
        std::vector<Cell>& cells = map_[loc].cells;
        auto it = std::find_if(cells.begin(), cells.end(), [idx](const Cell& c) { return c.idx == idx; });
        if (it == cells.end()) {
            cells.emplace_back();
            cells.back().idx = idx;
            return cells.back();
        }
        return *it;
    }

    void ExtractRoot(const Root& root, const Poses& poses, std::vector<PlaneFeature>* features) const {
        PointCluster tr_clu;
        bool split[kLayers][kCells] = {};
        for (int layer = 0; layer < kLayers; ++layer) {
            const int num_nodes = 1 << (3 * layer);
            PointCluster node_total_clu[kCells];                  // world frame
            std::vector<const FrameCluster*> node_clus[kCells];   // body frame
            bool touched[kCells] = {};
            bool any = false;
            for (const Cell& cell : root.cells) {
                if (layer > 0 && !split[layer - 1][NodeOf(cell.idx, layer - 1)]) continue;
                const int n = NodeOf(cell.idx, layer);
                touched[n] = any = true;
                for (const FrameCluster& cl : cell.obs) {
                    const Eigen::Isometry3d& pose = poses.at(cl.xid);
                    tr_clu.Transform(cl.clu, pose.linear(), pose.translation());
                    node_total_clu[n] += tr_clu;
                    node_clus[n].push_back(&cl);
                }
            }
            if (!any) break;

            for (int n = 0; n < num_nodes; ++n) {
                if (!touched[n] || node_total_clu[n].N <= param_.min_points) continue;
                const Eigen::SelfAdjointEigenSolver<Mat3> saes(node_total_clu[n].Cov());
                const Vec3& eig = saes.eigenvalues();
                if (eig[0] >= param_.plane_eigen_ratio * eig[1]) {  // lambda_0 / lambda_1
                    split[layer][n] = true;
                    continue;
                }
                const double extent = param_.min_plane_size * param_.voxel_size / (1 << layer);
                if (eig[1] < extent * extent) continue;

                std::stable_sort(node_clus[n].begin(), node_clus[n].end(),
                                 [](const FrameCluster* a, const FrameCluster* b) { return a->xid < b->xid; });
                PlaneFeature fea;
                fea.layer = layer;
                for (const FrameCluster* cl : node_clus[n]) {
                    if (fea.clus.empty() || fea.clus.back().xid != cl->xid) {
                        fea.clus.emplace_back();
                        fea.clus.back().xid = cl->xid;
                    }
                    fea.clus.back().clu += cl->clu;
                }
                features->push_back(std::move(fea));
            }
        }
    }

    VoxelMapOptions param_;
    std::unordered_map<VOXEL_LOCATION, Root> map_;
};

}  // namespace ba
}  // namespace faster_lio

#endif  // FASTER_LIO_BA_PLANE_FEATURES_H

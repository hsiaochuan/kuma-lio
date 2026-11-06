//
// Plane cells of a keyframe cloud, the same judgement the local BA applies to its own voxels: a
// cell is a plane when the smallest eigenvalue of its point covariance is small enough.  BTC
// builds these very cells for its plane overlap, and the loop ICP runs on their centres.
//

#ifndef FASTER_LIO_PGO_PLANE_CLOUD_H
#define FASTER_LIO_PGO_PLANE_CLOUD_H

#include <vector>

#include "pgo/pgo_types.h"
#include "point_cluster.h"

namespace faster_lio {
namespace pgo {

/// A flat `plane_voxel` cell: its point statistics, which BTC merges into reference planes, and
/// the centre + normal the loop ICP and the plane overlap use.
struct PlaneCell {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointCluster pc;
    PlanePoint plane;
};
using PlaneCells = std::vector<PlaneCell, Eigen::aligned_allocator<PlaneCell>>;

/// Hash `cloud` into `plane_voxel` cells and keep the flat ones.
PlaneCells ExtractPlaneCells(const std::vector<Vec3>& cloud, const PgoParam& prm);

std::vector<PlanePoint> PlanePoints(const PlaneCells& cells);

std::vector<Vec3> UniformSample(const std::vector<Vec3>& cloud, double leaf);

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_PLANE_CLOUD_H

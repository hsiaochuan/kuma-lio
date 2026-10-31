//
// Plane extraction for the loop ICP, standing in for the plane cloud BTC builds.  Same judgement
// the local BA applies to its own voxels: a cell is a plane when the smallest eigenvalue of its
// point covariance is small enough.
//

#ifndef FASTER_LIO_PGO_PLANE_CLOUD_H
#define FASTER_LIO_PGO_PLANE_CLOUD_H

#include <vector>

#include "pgo/pgo_types.h"

namespace faster_lio {
namespace pgo {

/// Hash `cloud` into `plane_voxel` cells and emit one centre + normal per cell that is flat.
std::vector<PlanePoint> ExtractPlanes(const std::vector<Vec3>& cloud, const PgoParam& prm);

/// Voxel-grid centroid downsampling, used for the stored keyframe cloud.
std::vector<Vec3> DownSample(const std::vector<Vec3>& cloud, double leaf);

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_PLANE_CLOUD_H

#include "pgo/plane_cloud.h"

#include <unordered_map>

#include "common_lib.h"
#include "point_cluster.h"

namespace faster_lio {
namespace pgo {

std::vector<Vec3> DownSample(const std::vector<Vec3>& cloud, double leaf) {
    // Eigen's fixed-size default constructor leaves the coefficients uninitialized, so an
    // accumulator reached through operator[] would start from garbage.  Seed it explicitly.
    struct Sum {
        Vec3 p = Vec3::Zero();
        int n = 0;
    };
    std::unordered_map<VOXEL_LOCATION, Sum> cells;
    cells.reserve(cloud.size());
    for (const Vec3& p : cloud) {
        if (!p.allFinite()) continue;
        Sum& cell = cells[VOXEL_LOCATION(p, leaf)];
        cell.p += p;
        cell.n++;
    }
    std::vector<Vec3> out;
    out.reserve(cells.size());
    for (const auto& kv : cells) out.push_back(kv.second.p / kv.second.n);
    return out;
}

std::vector<PlanePoint> ExtractPlanes(const std::vector<Vec3>& cloud, const PgoParam& prm) {
    std::unordered_map<VOXEL_LOCATION, PointCluster> cells;
    cells.reserve(cloud.size());
    // PointCluster zeroes itself on construction, so operator[] is safe here
    for (const Vec3& p : cloud) {
        if (p.allFinite()) cells[VOXEL_LOCATION(p, prm.plane_voxel)].Push(p);
    }

    std::vector<PlanePoint> out;
    out.reserve(cells.size());
    for (const auto& kv : cells) {
        const PointCluster& pc = kv.second;
        if (pc.N < prm.plane_min_points) continue;
        Eigen::SelfAdjointEigenSolver<Mat3> es(pc.Cov());
        if (es.eigenvalues()(0) >= prm.plane_min_eigen) continue;
        PlanePoint pp;
        pp.p = pc.Mean();
        pp.n = es.eigenvectors().col(0);
        if (!pp.p.allFinite() || !pp.n.allFinite()) continue;
        out.push_back(pp);
    }
    return out;
}

}  // namespace pgo
}  // namespace faster_lio

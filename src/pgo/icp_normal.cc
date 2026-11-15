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

#include "pgo/icp_normal.h"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <vector>

#include "so3_math.h"

namespace kuma_lio {
namespace pgo {

IcpResult IcpNormal(const PlaneCells& src, const PlaneCells& tar, const PgoParam& prm, Eigen::Isometry3d* T_tar_src) {
    IcpResult res;
    if (src.empty() || tar.size() < 4) return res;

    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    tar_xyz->reserve(tar.size());
    std::vector<const PlaneCell*> tar_ref;
    tar_ref.reserve(tar.size());
    for (const PlaneCell& pp : tar) {
        if (!pp.center.allFinite() || !pp.normal.allFinite()) continue;
        tar_xyz->push_back(pcl::PointXYZ(pp.center.x(), pp.center.y(), pp.center.z()));
        tar_ref.push_back(&pp);
    }
    if (tar_xyz->size() < 4) return res;
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    kd_tree.setInputCloud(tar_xyz);

    std::vector<int> idx(1);
    std::vector<float> dist(1);

    // the loose phase finds the basin, the tight one that follows decides whether it is real
    Vec4 paras(0.2, 0.2, 0.5, 3.0);

    Mat3 R = T_tar_src->linear();
    Vec3 t = T_tar_src->translation();
    Mat3 mat_norm = Mat3::Zero();
    int match_num = 0;
    bool& converged = res.converged;

    res.src_n = static_cast<int>(src.size());
    res.tar_n = static_cast<int>(tar_xyz->size());
    res.seed_t = T_tar_src->translation().norm();
    std::vector<double> nn0;
    nn0.reserve(src.size());

    for (int iter = 0; iter < prm.icp_max_iter; ++iter) {
        res.iters = iter + 1;
        Mat6 hess = Mat6::Zero();
        Vec6 jact = Vec6::Zero();
        mat_norm.setZero();
        match_num = 0;

        for (const PlaneCell& sp : src) {
            const Vec3 pi = R * sp.center + t;
            const Vec3 ni = R * sp.normal;
            // PCL asserts on a non-finite query rather than returning nothing, and an aborted
            // process is a far worse failure than a skipped correspondence
            if (!pi.allFinite() || !ni.allFinite()) continue;
            if (kd_tree.nearestKSearch(pcl::PointXYZ(pi.x(), pi.y(), pi.z()), 1, idx, dist) <= 0) continue;
            const PlaneCell& tp = *tar_ref[idx[0]];
            // how far the nearest plane centre is BEFORE any gate: says whether the two clouds
            // overlap at all under the seed, which no gated count can distinguish from a bad fit
            if (iter == 0) nn0.push_back((pi - tp.center).norm());
            // a plane normal has no sign, so either orientation is a match
            if ((ni - tp.normal).norm() >= paras[0] && (ni + tp.normal).norm() >= paras[1]) continue;
            const Vec3 d = pi - tp.center;
            const double point_to_plane = std::abs(tp.normal.dot(d));
            if (point_to_plane >= paras[2] || d.norm() >= paras[3]) continue;

            const double r = tp.normal.dot(d);
            Vec6 jac;
            jac.head<3>() = Hat(sp.center) * R.transpose() * tp.normal;
            jac.tail<3>() = tp.normal;
            hess += jac * jac.transpose();
            jact += jac * r;
            mat_norm += tp.normal * tp.normal.transpose();
            ++match_num;
        }

        // six correspondences is the bare minimum for the normal equations to have a solution at
        // all; below that ldlt() hands back noise that would walk the pose anywhere
        if (iter == 0 && !nn0.empty()) {
            std::nth_element(nn0.begin(), nn0.begin() + nn0.size() / 2, nn0.end());
            res.nn_median = nn0[nn0.size() / 2];
        }
        if (match_num < 6) {
            res.matches = match_num;
            res.starved = true;
            return res;
        }

        const Vec6 dxi = hess.ldlt().solve(-jact);
        if (!dxi.allFinite()) {
            res.matches = match_num;
            res.starved = true;
            return res;
        }
        res.step = dxi.tail<3>().norm();
        R = R * ExpMat(Vec3(dxi.head<3>()));
        t += dxi.tail<3>();

        if (dxi.head<3>().norm() < 1e-3 && dxi.tail<3>().norm() < 1e-3) {
            if (converged) break;
            paras << 0.1, 0.1, 0.1, 1.0;
            converged = true;
        }
    }

    Eigen::SelfAdjointEigenSolver<Mat3> es(mat_norm);
    res.eigval = es.eigenvalues()(0);
    res.matches = match_num;
    T_tar_src->linear() = R;
    T_tar_src->translation() = t;
    // appendix B: the smallest eigenvalue of sum(n n^T) says how well the weakest direction is
    // constrained, so this is an observability test, not a goodness-of-fit one
    res.ok = res.converged && res.eigval > prm.icp_eigval && match_num >= prm.icp_min_match;
    return res;
}

}  // namespace pgo
}  // namespace kuma_lio

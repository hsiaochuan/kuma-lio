#include "pgo/loop_detector.h"

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>

#include <algorithm>
#include <cmath>

namespace faster_lio {
namespace pgo {
namespace {

PgoCloudPtr VoxelDown(const PgoCloudPtr& in, double leaf) {
    if (leaf <= 0.0 || in->empty()) return in;
    PgoCloudPtr out(new PgoCloud);
    pcl::VoxelGrid<PgoPoint> vg;
    vg.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
    vg.setInputCloud(in);
    vg.filter(*out);
    return out;
}

double RotationAngleDeg(const Eigen::Matrix3d& R) {
    return Eigen::AngleAxisd(R).angle() * 180.0 / M_PI;
}

}  // namespace

std::vector<int> LoopDetector::FindCandidates(const std::vector<KeyFrame::Ptr>& kfs,
                                              int cur_id) const {
    std::vector<int> out;
    const int last_searchable = cur_id - prm_.loop_closest_id_th;
    if (last_searchable < 0) return out;

    const Eigen::Vector3d p_cur = kfs[cur_id]->pose_opt.translation();

    // 2-D (x-y) radius search: a revisit on another floor is rejected by dz,
    // not by the radius, so the tree deliberately flattens z.
    pcl::PointCloud<pcl::PointXYZ>::Ptr pos(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> pos_to_id;
    pos->reserve(last_searchable + 1);
    pos_to_id.reserve(last_searchable + 1);
    for (int i = 0; i <= last_searchable; ++i) {
        const Eigen::Vector3d& t = kfs[i]->pose_opt.translation();
        pcl::PointXYZ p;
        p.x = static_cast<float>(t.x());
        p.y = static_cast<float>(t.y());
        p.z = 0.0f;
        pos->push_back(p);
        pos_to_id.push_back(i);
    }
    if (pos->empty()) return out;

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(pos);

    pcl::PointXYZ q;
    q.x = static_cast<float>(p_cur.x());
    q.y = static_cast<float>(p_cur.y());
    q.z = 0.0f;

    std::vector<int> idx;
    std::vector<float> dist2;
    tree.radiusSearch(q, prm_.loop_search_radius, idx, dist2);
    if (idx.empty()) return out;

    // nearest first; keep the few best that also pass the height gate
    for (size_t k = 0; k < idx.size() && static_cast<int>(out.size()) < prm_.loop_max_candidates;
         ++k) {
        const int id = pos_to_id[idx[k]];
        if (prm_.loop_search_max_dz > 0.0) {
            const double dz =
                std::fabs(kfs[id]->pose_opt.translation().z() - p_cur.z());
            if (dz > prm_.loop_search_max_dz) continue;
        }
        // thin out candidates that sit on top of one another
        bool too_close = false;
        for (int already : out) {
            if (std::abs(already - id) < prm_.submap_half_size) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;
        out.push_back(id);
    }
    return out;
}

PgoCloudPtr LoopDetector::BuildSubmap(const std::vector<KeyFrame::Ptr>& kfs, int center_id) const {
    PgoCloudPtr submap(new PgoCloud);
    const int lo = std::max(0, center_id - prm_.submap_half_size);
    const int hi = std::min(static_cast<int>(kfs.size()) - 1, center_id + prm_.submap_half_size);
    const int stride = std::max(1, prm_.submap_stride);

    size_t reserve = 0;
    for (int i = lo; i <= hi; i += stride) reserve += kfs[i]->cloud ? kfs[i]->cloud->size() : 0;
    submap->reserve(reserve);

    PgoCloud tmp;
    for (int i = lo; i <= hi; i += stride) {
        const KeyFrame::Ptr& kf = kfs[i];
        if (!kf->cloud || kf->cloud->empty()) continue;
        tmp.clear();
        pcl::transformPointCloud(*kf->cloud, tmp, kf->pose_opt.matrix().cast<float>());
        submap->insert(submap->end(), tmp.begin(), tmp.end());
    }
    return submap;
}

bool LoopDetector::Verify(const PgoCloudPtr& src_world, const PgoCloudPtr& tgt_world,
                          Eigen::Isometry3d* T_icp, double* fitness, double* overlap) const {
    if (src_world->empty() || tgt_world->empty()) return false;

    using Gicp = pcl::GeneralizedIterativeClosestPoint<PgoPoint, PgoPoint>;
    const auto configure = [this](Gicp& gicp, double max_corr) {
        gicp.setMaxCorrespondenceDistance(max_corr);
        gicp.setMaximumIterations(prm_.loop_max_iterations);
        gicp.setTransformationEpsilon(1e-4);
        gicp.setEuclideanFitnessEpsilon(1e-3);
        gicp.setRANSACIterations(0);
        gicp.setCorrespondenceRandomness(20);
        gicp.setMaximumOptimizerIterations(20);
    };

    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
    PgoCloud aligned;

    // ---- coarse stage ----
    {
        PgoCloudPtr src = VoxelDown(src_world, prm_.loop_coarse_leaf);
        PgoCloudPtr tgt = VoxelDown(tgt_world, prm_.loop_coarse_leaf);
        if (src->empty() || tgt->empty()) return false;

        Gicp gicp;
        configure(gicp, prm_.loop_coarse_max_corr);
        gicp.setInputSource(src);
        gicp.setInputTarget(tgt);
        gicp.align(aligned, guess);
        if (!gicp.hasConverged()) return false;
        guess = gicp.getFinalTransformation();
    }

    // ---- fine stage ----
    PgoCloudPtr src = VoxelDown(src_world, prm_.loop_fine_leaf);
    PgoCloudPtr tgt = VoxelDown(tgt_world, prm_.loop_fine_leaf);
    if (src->empty() || tgt->empty()) return false;

    Gicp gicp;
    configure(gicp, prm_.loop_fine_max_corr);
    gicp.setInputSource(src);
    gicp.setInputTarget(tgt);
    gicp.align(aligned, guess);
    if (!gicp.hasConverged()) return false;

    const Eigen::Matrix4f Tf = gicp.getFinalTransformation();
    *T_icp = Eigen::Isometry3d::Identity();
    T_icp->linear() = Tf.block<3, 3>(0, 0).cast<double>();
    T_icp->translation() = Tf.block<3, 1>(0, 3).cast<double>();

    // getFitnessScore() with no argument uses an unbounded max range and so
    // folds in every spurious far-away correspondence; always bound it.
    *fitness = gicp.getFitnessScore(prm_.loop_fine_max_corr);

    // The fitness score is a mean over CORRESPONDING points only, so a
    // registration that matches 5% of the cloud can still score beautifully.
    // The inlier ratio is what catches that.
    pcl::KdTreeFLANN<PgoPoint> tree;
    tree.setInputCloud(tgt);
    const float max_corr2 =
        static_cast<float>(prm_.loop_fine_max_corr * prm_.loop_fine_max_corr);
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    size_t inliers = 0;
    for (const PgoPoint& p : aligned.points) {
        if (!std::isfinite(p.x)) continue;
        if (tree.nearestKSearch(p, 1, idx, d2) > 0 && d2[0] <= max_corr2) ++inliers;
    }
    *overlap = aligned.empty() ? 0.0
                              : static_cast<double>(inliers) / static_cast<double>(aligned.size());
    return true;
}

bool LoopDetector::Detect(const std::vector<KeyFrame::Ptr>& kfs, int cur_id, LoopEdge* edge) {
    if (cur_id < prm_.loop_min_kf_id) return false;
    if (cur_id - last_attempt_id_ < prm_.loop_kf_gap) return false;
    if (kfs[cur_id]->cloud->empty()) return false;

    const std::vector<int> candidates = FindCandidates(kfs, cur_id);
    if (candidates.empty()) return false;
    last_attempt_id_ = cur_id;

    PgoCloudPtr src_world(new PgoCloud);
    pcl::transformPointCloud(*kfs[cur_id]->cloud, *src_world,
                             kfs[cur_id]->pose_opt.matrix().cast<float>());

    for (int cand : candidates) {
        Rejection rej;
        rej.id_from = cand;
        rej.id_to = cur_id;

        PgoCloudPtr tgt_world = BuildSubmap(kfs, cand);
        if (tgt_world->size() < 1000) {
            rej.reason = "submap_too_small";
            rejections_.push_back(rej);
            continue;
        }

        Eigen::Isometry3d T_icp = Eigen::Isometry3d::Identity();
        double fitness = 0.0, overlap = 0.0;
        if (!Verify(src_world, tgt_world, &T_icp, &fitness, &overlap)) {
            rej.reason = "gicp_diverged";
            rejections_.push_back(rej);
            continue;
        }

        rej.fitness = fitness;
        rej.overlap = overlap;
        rej.corr_trans = T_icp.translation().norm();
        rej.corr_rot = RotationAngleDeg(T_icp.linear());

        if (fitness > prm_.loop_fitness_th) {
            rej.reason = "fitness";
            rejections_.push_back(rej);
            continue;
        }
        if (overlap < prm_.loop_min_overlap) {
            rej.reason = "overlap";
            rejections_.push_back(rej);
            continue;
        }
        // Gross-error gate: T_icp IS the disagreement between the registration
        // and the current estimate, so bounding it rejects the confident-but-
        // wrong matches that self-similar corridors and planar scenes produce.
        if (rej.corr_trans > prm_.loop_max_corr_trans) {
            rej.reason = "corr_trans";
            rejections_.push_back(rej);
            continue;
        }
        if (rej.corr_rot > prm_.loop_max_corr_rot) {
            rej.reason = "corr_rot";
            rejections_.push_back(rej);
            continue;
        }

        edge->id_from = cand;
        edge->id_to = cur_id;
        // T_icp corrects the current keyframe in the WORLD frame (left multiply).
        edge->T_from_to = kfs[cand]->pose_opt.inverse() * (T_icp * kfs[cur_id]->pose_opt);
        edge->fitness = fitness;
        edge->overlap = overlap;
        edge->corr_trans = rej.corr_trans;
        edge->corr_rot = rej.corr_rot;

        LOG(INFO) << "[pgo] loop " << cand << " -> " << cur_id << "  fitness " << fitness
                  << "  overlap " << overlap << "  dt " << edge->corr_trans << " m  dr "
                  << edge->corr_rot << " deg";
        return true;  // one loop per keyframe is plenty
    }
    return false;
}

}  // namespace pgo
}  // namespace faster_lio

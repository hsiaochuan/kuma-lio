#include "hash_octree.h"

#include <cmath>
#include <mutex>
using namespace faster_lio;
namespace {
int ComputeChildIndex(const Eigen::Vector3d &point, const double center[3], int (&xyz)[3])
{
    xyz[0] = point[0] > center[0] ? 1 : 0;
    xyz[1] = point[1] > center[1] ? 1 : 0;
    xyz[2] = point[2] > center[2] ? 1 : 0;
    return 4 * xyz[0] + 2 * xyz[1] + xyz[2];
}

OctoTree *EnsureChildNode(OctoTree *parent, int leaf_idx, const int (&xyz)[3])
{
    if (parent->leaves_[leaf_idx] != nullptr) return parent->leaves_[leaf_idx];

    OctoTree *child = new OctoTree(
        parent->max_layer_, parent->layer_ + 1,
        parent->layer_init_num_[parent->layer_ + 1],
        parent->max_points_num_, parent->planer_threshold_);
    child->layer_init_num_  = parent->layer_init_num_;
    child->voxel_center_[0] = parent->voxel_center_[0] + (2 * xyz[0] - 1) * parent->quater_length_;
    child->voxel_center_[1] = parent->voxel_center_[1] + (2 * xyz[1] - 1) * parent->quater_length_;
    child->voxel_center_[2] = parent->voxel_center_[2] + (2 * xyz[2] - 1) * parent->quater_length_;
    child->quater_length_   = parent->quater_length_ / 2.f;
    parent->leaves_[leaf_idx] = child;
    return child;
}

constexpr double kSigmaFloor = 1e-12;
} // namespace

// ============================================================================
// OctoTree
// ============================================================================

OctoTree::OctoTree(int max_layer, int layer,
                   int points_size_threshold, int max_points_num,
                   float planer_threshold)
    : max_layer_(max_layer),
      layer_(layer),
      points_size_threshold_(points_size_threshold),
      max_points_num_(max_points_num),
      planer_threshold_(planer_threshold)
{
    octo_state_           = 0;
    new_points_           = 0;
    update_size_threshold_ = 5;
    init_octo_            = false;
    update_enable_        = true;
    plane_ptr_            = new VoxelPlane;
    for (int i = 0; i < 8; i++) leaves_[i] = nullptr;
}

OctoTree::~OctoTree()
{
    for (int i = 0; i < 8; i++) delete leaves_[i];
    delete plane_ptr_;
}

// ---------------------------------------------------------------------------
// Fit a plane to a set of points and propagate covariance.
// Sets plane->is_plane = true when min eigenvalue < planer_threshold_.
// ---------------------------------------------------------------------------
void OctoTree::InitPlane(const std::vector<PointWithCov> &points, VoxelPlane *plane)
{
    plane->plane_var   = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance  = Eigen::Matrix3d::Zero();
    plane->center      = Eigen::Vector3d::Zero();
    plane->normal      = Eigen::Vector3d::Zero();
    plane->points_size = static_cast<int>(points.size());
    plane->radius      = 0.f;

    for (const auto &pv : points)
    {
        plane->covariance += pv.point * pv.point.transpose();
        plane->center     += pv.point;
    }
    plane->center    = plane->center    / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size
                       - plane->center * plane->center.transpose();

    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Vector3d evals_real = es.eigenvalues().real();
    Eigen::Matrix3d evecs_real = es.eigenvectors().real();

    Eigen::Index eval_min_idx, eval_max_idx;
    evals_real.rowwise().sum().minCoeff(&eval_min_idx);
    evals_real.rowwise().sum().maxCoeff(&eval_max_idx);
    int eval_mid_idx = 3 - static_cast<int>(eval_min_idx) - static_cast<int>(eval_max_idx);

    if (evals_real(eval_min_idx) < planer_threshold_)
    {
        // Propagate point covariances into plane uncertainty
        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / plane->points_size;
        for (size_t i = 0; i < points.size(); i++)
        {
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++)
            {
                if (m != static_cast<int>(eval_min_idx))
                {
                    Eigen::RowVector3d F_m =
                        (points[i].point - plane->center).transpose()
                        / (plane->points_size * (evals_real[eval_min_idx] - evals_real[m]))
                        * (evecs_real.col(m)           * evecs_real.col(eval_min_idx).transpose()
                         + evecs_real.col(eval_min_idx) * evecs_real.col(m).transpose());
                    F.row(m) = F_m;
                }
                else
                {
                    F.row(m) = Eigen::RowVector3d::Zero();
                }
            }
            Eigen::Matrix<double, 6, 3> J;
            J.block<3, 3>(0, 0) = evecs_real * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane->plane_var += J * points[i].cov * J.transpose();
        }

        plane->normal    = evecs_real.col(eval_min_idx);
        plane->y_normal  = evecs_real.col(eval_mid_idx);
        plane->x_normal  = evecs_real.col(eval_max_idx);
        plane->min_eigen_value = static_cast<float>(evals_real(eval_min_idx));
        plane->mid_eigen_value = static_cast<float>(evals_real(eval_mid_idx));
        plane->max_eigen_value = static_cast<float>(evals_real(eval_max_idx));
        plane->radius    = std::sqrt(static_cast<float>(evals_real(eval_max_idx)));
        plane->d         = static_cast<float>(
            -(plane->normal.dot(plane->center)));
        plane->is_plane  = true;
        //plane->is_update = true;
        plane->is_init   = true;
    }
    else
    {
        plane->is_plane  = false;
        //plane->is_update = true;
    }
}

// ---------------------------------------------------------------------------
// Try to initialise this node once enough points have been collected.
// ---------------------------------------------------------------------------
void OctoTree::InitOctoTree()
{
    if (static_cast<int>(temp_points_.size()) <= points_size_threshold_) return;

    InitPlane(temp_points_, plane_ptr_);

    if (plane_ptr_->is_plane)
    {
        octo_state_ = 0; // leaf
        if (static_cast<int>(temp_points_.size()) > max_points_num_)
        {
            update_enable_ = false;
            std::vector<PointWithCov>().swap(temp_points_);
            new_points_    = 0;
        }
    }
    else
    {
        octo_state_ = 1; // branch
        CutOctoTree();
    }
    init_octo_  = true;
    new_points_ = 0;
}

// ---------------------------------------------------------------------------
// Subdivide: redistribute buffered points into 8 child octants.
// ---------------------------------------------------------------------------
void OctoTree::CutOctoTree()
{
    if (layer_ >= max_layer_)
    {
        octo_state_ = 0;
        return;
    }

    for (const auto &pv : temp_points_)
    {
        int xyz[3] = {0, 0, 0};
        int leaf_idx = ComputeChildIndex(pv.point, voxel_center_, xyz);
        OctoTree *leaf = EnsureChildNode(this, leaf_idx, xyz);
        leaf->temp_points_.push_back(pv);
        leaf->new_points_++;
    }

    for (int i = 0; i < 8; i++)
    {
        OctoTree *leaf = leaves_[i];
        if (leaf == nullptr) continue;
        if (static_cast<int>(leaf->temp_points_.size()) <= leaf->points_size_threshold_) continue;

        InitPlane(leaf->temp_points_, leaf->plane_ptr_);
        if (leaf->plane_ptr_->is_plane)
        {
            leaf->octo_state_ = 0;
            if (static_cast<int>(leaf->temp_points_.size()) > leaf->max_points_num_)
            {
                leaf->update_enable_ = false;
                std::vector<PointWithCov>().swap(leaf->temp_points_);
                leaf->new_points_    = 0;
            }
        }
        else
        {
            leaf->octo_state_ = 1;
            leaf->CutOctoTree();
        }
        leaf->init_octo_  = true;
        leaf->new_points_ = 0;
    }
}

// ---------------------------------------------------------------------------
// Incrementally insert one point into this subtree.
// ---------------------------------------------------------------------------
void OctoTree::Update(const PointWithCov &pv)
{
    if (!init_octo_)
    {
        // Still accumulating: buffer the point, init when threshold reached
        new_points_++;
        temp_points_.push_back(pv);
        if (static_cast<int>(temp_points_.size()) > points_size_threshold_)
            InitOctoTree();
        return;
    }

    if (plane_ptr_->is_plane)
    {
        // Leaf with a valid plane: keep re-fitting until max_points_num_ reached
        if (!update_enable_) return;

        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
            InitPlane(temp_points_, plane_ptr_);
            new_points_ = 0;
        }
        if (static_cast<int>(temp_points_.size()) >= max_points_num_)
        {
            update_enable_ = false;
            std::vector<PointWithCov>().swap(temp_points_);
            new_points_    = 0;
        }
        return;
    }

    // Branch: route point to the correct child octant
    if (layer_ < max_layer_)
    {
        int xyz[3] = {0, 0, 0};
        int leaf_idx = ComputeChildIndex(pv.point, voxel_center_, xyz);
        OctoTree *leaf = EnsureChildNode(this, leaf_idx, xyz);
        leaf->Update(pv);
        return;
    }

    // At max layer, treat as leaf regardless
    if (!update_enable_) return;

    new_points_++;
    temp_points_.push_back(pv);
    if (new_points_ > update_size_threshold_)
    {
        InitPlane(temp_points_, plane_ptr_);
        new_points_ = 0;
    }
    if (static_cast<int>(temp_points_.size()) > max_points_num_)
    {
        update_enable_ = false;
        std::vector<PointWithCov>().swap(temp_points_);
        new_points_    = 0;
    }
}

// ============================================================================
// VoxelMap
// ============================================================================

HashOctree::HashOctree(const HashOctreeConfig &config) : config_(config) {}

HashOctree::~HashOctree()
{
    for (auto &kv : map_) delete kv.second;
}

// ---------------------------------------------------------------------------
// Convert a world-frame point to its integer voxel key.
// ---------------------------------------------------------------------------
VOXEL_LOCATION HashOctree::ToKey(const Eigen::Vector3d &p) const
{
    float s = static_cast<float>(config_.voxel_size);
    auto  coord = [&](double v) -> int64_t
    {
        float f = static_cast<float>(v) / s;
        if (f < 0.f) f -= 1.f;
        return static_cast<int64_t>(f);
    };
    return VOXEL_LOCATION(coord(p.x()), coord(p.y()), coord(p.z()));
}

// ---------------------------------------------------------------------------
// Look up or create the root octree node for a given key.
// ---------------------------------------------------------------------------
OctoTree *HashOctree::GetOrCreateNode(const VOXEL_LOCATION &key)
{
    auto it = map_.find(key);
    if (it != map_.end()) return it->second;

    float vs = static_cast<float>(config_.voxel_size);
    OctoTree *node = new OctoTree(
        config_.max_layer, 0,
        config_.layer_init_num[0],
        config_.max_points_num,
        static_cast<float>(config_.planer_threshold));
    node->layer_init_num_  = config_.layer_init_num;
    node->quater_length_   = vs / 4.f;
    node->voxel_center_[0] = (0.5 + key.x) * vs;
    node->voxel_center_[1] = (0.5 + key.y) * vs;
    node->voxel_center_[2] = (0.5 + key.z) * vs;
    map_[key]              = node;
    return node;
}

// ---------------------------------------------------------------------------
// AddPoints: insert a batch of world-frame points with covariance.
// ---------------------------------------------------------------------------
void HashOctree::AddPoints(const std::vector<PointWithCov> &points)
{
    for (const auto &pv : points)
    {
        VOXEL_LOCATION key  = ToKey(pv.point);
        OctoTree *node = GetOrCreateNode(key);
        node->Update(pv);
    }
}

// ---------------------------------------------------------------------------
// BuildSingleResidual: recursive descent to find the best plane match for pv.
// Selects the plane with the highest Mahalanobis-weighted probability; accepts
// only if the distance is within sigma_num standard deviations.
// ---------------------------------------------------------------------------
void HashOctree::BuildSingleResidual(const PointWithCov &pv,
                                   const OctoTree     *octo,
                                   int                 layer,
                                   bool               &success,
                                   double             &prob,
                                   PlaneMatch         &match)
{
    const double kRadiusScale = 3.0;
    const double sigma_num = config_.sigma_num;
    const int    max_layer = config_.max_layer;

    const Eigen::Vector3d &p_w = pv.point;

    if (octo->plane_ptr_->is_plane)
    {
        const VoxelPlane &plane = *octo->plane_ptr_;

        float dis_to_plane = static_cast<float>(
            plane.normal.dot(p_w) + plane.d);

        float dis_to_center = static_cast<float>((plane.center - p_w).squaredNorm());
        float range_dis     = std::sqrt(
            std::max(0.f, dis_to_center - dis_to_plane * dis_to_plane));

        if (range_dis > kRadiusScale * plane.radius) return;

        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = (p_w - plane.center).transpose();
        J_nq.block<1, 3>(0, 3) = -plane.normal.transpose();
        double sigma_l = (J_nq * plane.plane_var * J_nq.transpose())(0, 0)
                       + plane.normal.transpose() * pv.cov * plane.normal;
        sigma_l = std::max(sigma_l, kSigmaFloor);

        if (std::fabs(dis_to_plane) >= sigma_num * std::sqrt(sigma_l)) return;

        double this_prob = (1.0 / std::sqrt(sigma_l))
                         * std::exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);

        if (this_prob > prob)
        {
            success          = true;
            prob             = this_prob;
            match.point_w    = p_w;
            match.normal     = plane.normal;
            match.center     = plane.center;
            match.plane_var  = plane.plane_var;
            match.d          = plane.d;
            match.dis_to_plane = dis_to_plane;
            match.layer      = layer;
        }
        return;
    }

    if (layer < max_layer)
    {
        for (int i = 0; i < 8; i++)
        {
            if (octo->leaves_[i] != nullptr)
                BuildSingleResidual(pv, octo->leaves_[i], layer + 1, success, prob, match);
        }
    }
}

// ---------------------------------------------------------------------------
// Match: for each query point find its best matching plane in the map.
// Falls back to the adjacent voxel if the primary voxel yields no match.
// ---------------------------------------------------------------------------
std::vector<PlaneMatch> HashOctree::Match(const std::vector<PointWithCov> &query_points)
{
    const double vs = config_.voxel_size;

    std::vector<PlaneMatch> all_matches(query_points.size());
    std::vector<bool>       valid(query_points.size(), false);
    std::mutex              mtx;

    // Parallelisable with OpenMP if available; falls back to serial otherwise.
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < static_cast<int>(query_points.size()); i++)
    {
        const PointWithCov &pv  = query_points[i];
        VOXEL_LOCATION            key = ToKey(pv.point);

        auto it = map_.find(key);
        if (it == map_.end()) continue;

        PlaneMatch match;
        bool       success = false;
        double     prob    = 0.0;

        BuildSingleResidual(pv, it->second, 0, success, prob, match);

        // If no match in primary voxel, try the nearest neighbouring voxel
        if (!success)
        {
            const OctoTree *root = it->second;
            VOXEL_LOCATION near_key    = key;

            auto nudge = [&](int axis, double center, double quarter)
            {
                float lf = static_cast<float>(pv.point[axis] / vs);
                if (lf > center + quarter)
                {
                    if (axis == 0) near_key.x++;
                    if (axis == 1) near_key.y++;
                    if (axis == 2) near_key.z++;
                }
                else if (lf < center - quarter)
                {
                    if (axis == 0) near_key.x--;
                    if (axis == 1) near_key.y--;
                    if (axis == 2) near_key.z--;
                }
            };
            nudge(0, root->voxel_center_[0], root->quater_length_);
            nudge(1, root->voxel_center_[1], root->quater_length_);
            nudge(2, root->voxel_center_[2], root->quater_length_);

            auto it2 = map_.find(near_key);
            if (it2 != map_.end())
                BuildSingleResidual(pv, it2->second, 0, success, prob, match);
        }

        if (success)
        {
            std::lock_guard<std::mutex> lock(mtx);
            all_matches[i] = match;
            valid[i]       = true;
        }
    }

    // Collect valid matches
    std::vector<PlaneMatch> result;
    result.reserve(query_points.size());
    for (size_t i = 0; i < valid.size(); i++)
        if (valid[i]) result.push_back(all_matches[i]);

    return result;
}

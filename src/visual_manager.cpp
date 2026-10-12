//
// Created by hsiaochuan on 2026/05/11.
//

#include "visual_manager.h"

#include "local_ba.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
namespace faster_lio {
namespace {

float InterpolateMat8u(const cv::Mat& mat, float u, float v) {
    const int x = static_cast<int>(std::floor(u));
    const int y = static_cast<int>(std::floor(v));
    const float subpix_x = u - x;
    const float subpix_y = v - y;
    const float w00 = (1.0f - subpix_x) * (1.0f - subpix_y);
    const float w01 = (1.0f - subpix_x) * subpix_y;
    const float w10 = subpix_x * (1.0f - subpix_y);
    const float w11 = 1.0f - w00 - w01 - w10;
    const int stride = static_cast<int>(mat.step.p[0]);
    const unsigned char* ptr = mat.data + y * stride + x;
    return w00 * ptr[0] + w01 * ptr[stride] + w10 * ptr[1] + w11 * ptr[stride + 1];
}

float ShiTomasiScore(const cv::Mat& img, int u, int v) {
    float dXX = 0.0f, dYY = 0.0f, dXY = 0.0f;
    const int halfbox_size = 4;
    const int box_size = 2 * halfbox_size;
    const int box_area = box_size * box_size;
    const int x_min = u - halfbox_size;
    const int x_max = u + halfbox_size;
    const int y_min = v - halfbox_size;
    const int y_max = v + halfbox_size;
    if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1) return 0.0f;

    const int stride = static_cast<int>(img.step.p[0]);
    for (int y = y_min; y < y_max; ++y) {
        const uint8_t* ptr_left = img.data + stride * y + x_min - 1;
        const uint8_t* ptr_right = img.data + stride * y + x_min + 1;
        const uint8_t* ptr_top = img.data + stride * (y - 1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride * (y + 1) + x_min;
        for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
            const float dx = static_cast<float>(*ptr_right) - static_cast<float>(*ptr_left);
            const float dy = static_cast<float>(*ptr_bottom) - static_cast<float>(*ptr_top);
            dXX += dx * dx;
            dYY += dy * dy;
            dXY += dx * dy;
        }
    }
    // the smaller eigenvalue of the structure tensor
    dXX /= 2.0f * box_area;
    dYY /= 2.0f * box_area;
    dXY /= 2.0f * box_area;
    return 0.5f * (dXX + dYY - std::sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

float Mean(const float* patch, int n) { return std::accumulate(patch, patch + n, 0.0f) / static_cast<float>(n); }

}  // namespace

VisualManager::VisualManager(const VisualParam& prm, const CamModel::Ptr& cam) : prm_(prm), cam_(cam) {
    CHECK(cam_) << "the photometric update needs a camera model";
    CHECK(prm_.patch_size > 0 && prm_.patch_size % 2 == 0) << "vio.patch_size must be even";
    half_patch_ = prm_.patch_size / 2;
    patch_total_ = prm_.patch_size * prm_.patch_size;
    border_ = (half_patch_ + 1) * (1 << prm_.patch_pyramid_level);
    grid_w_ = static_cast<int>(std::ceil(static_cast<double>(cam_->w()) / prm_.grid_size));
    grid_h_ = static_cast<int>(std::ceil(static_cast<double>(cam_->h()) / prm_.grid_size));
    cell_type_.assign(grid_w_ * grid_h_, kUnknown);
}

cv::Mat VisualManager::ToGray(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        img.copyTo(gray);
    }
    return gray;
}

Eigen::Matrix<double, 2, 3> VisualManager::ProjectJacobian(const Vec3& pc) const {
    // Central differences on the camera model itself, so that a fisheye lens gets
    // its real Jacobian instead of the pinhole one evaluated at the same point.
    Eigen::Matrix<double, 2, 3> J;
    const double h = 1e-5 * pc.norm();
    for (int k = 0; k < 3; ++k) {
        Vec3 d = Vec3::Zero();
        d[k] = h;
        J.col(k) = (cam_->project(pc + d) - cam_->project(pc - d)) / (2.0 * h);
    }
    return J;
}

void VisualManager::PatchFromImage(const cv::Mat& gray, const Vec2& px, float* patch) const {
    const float u = static_cast<float>(px[0]);
    const float v = static_cast<float>(px[1]);
    const int u_i = static_cast<int>(std::floor(u));
    const int v_i = static_cast<int>(std::floor(v));
    const float sub_u = u - u_i;
    const float sub_v = v - v_i;
    const float w_tl = (1.0f - sub_u) * (1.0f - sub_v);
    const float w_tr = sub_u * (1.0f - sub_v);
    const float w_bl = (1.0f - sub_u) * sub_v;
    const float w_br = sub_u * sub_v;
    const int stride = static_cast<int>(gray.step.p[0]);
    for (int r = 0; r < prm_.patch_size; ++r) {
        const uint8_t* ptr = gray.data + (v_i - half_patch_ + r) * stride + (u_i - half_patch_);
        for (int c = 0; c < prm_.patch_size; ++c, ++ptr) {
            patch[r * prm_.patch_size + c] = w_tl * ptr[0] + w_tr * ptr[1] + w_bl * ptr[stride] + w_br * ptr[stride + 1];
        }
    }
}

Mat2 VisualManager::HomographyAffine(const Vec2& px_ref, const Vec3& xyz_ref, const Vec3& normal_ref,
                                     const Pose3& T_cur_ref) const {
    // plane-induced homography of the patch, linearized at the patch centre
    const Vec3 t = T_cur_ref.GetInverse().Trans();
    const Mat3 H_cur_ref = T_cur_ref.Mat3d() * (normal_ref.dot(xyz_ref) * Mat3::Identity() - t * normal_ref.transpose());
    const Vec3 f_du_ref = cam_->bearing(px_ref + Vec2(half_patch_, 0));
    const Vec3 f_dv_ref = cam_->bearing(px_ref + Vec2(0, half_patch_));
    const Vec2 px_cur = cam_->project(H_cur_ref * xyz_ref);
    const Vec2 px_du_cur = cam_->project(H_cur_ref * f_du_ref);
    const Vec2 px_dv_cur = cam_->project(H_cur_ref * f_dv_ref);
    Mat2 A_cur_ref;
    A_cur_ref.col(0) = (px_du_cur - px_cur) / half_patch_;
    A_cur_ref.col(1) = (px_dv_cur - px_cur) / half_patch_;
    return A_cur_ref;
}

void VisualManager::WarpAffine(const Mat2& A_cur_ref, const cv::Mat& img_ref, const Vec2& px_ref, int search_level,
                               int level, float* patch) const {
    const Mat2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    float* out = patch + patch_total_ * level;
    for (int r = 0; r < prm_.patch_size; ++r) {
        for (int c = 0; c < prm_.patch_size; ++c) {
            Vec2f d(c - half_patch_, r - half_patch_);
            d *= static_cast<float>(1 << (search_level + level));
            const Vec2f px = A_ref_cur * d + px_ref.cast<float>();
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
                out[r * prm_.patch_size + c] = 0.0f;
            } else {
                out[r * prm_.patch_size + c] = InterpolateMat8u(img_ref, px[0], px[1]);
            }
        }
    }
}

VisualManager::Observation::Ptr VisualManager::MakeObservation(const cv::Mat& gray, const Pose3& camera_from_world,
                                                              const Vec2& px, int level) const {
    Observation::Ptr obs = std::make_shared<Observation>();
    obs->img = gray;
    obs->px = px;
    obs->camera_from_world = camera_from_world;
    obs->level = level;
    obs->inv_expo = cur_inv_expo_;
    obs->patch.resize(patch_total_);
    PatchFromImage(gray, px, obs->patch.data());
    obs->mean = Mean(obs->patch.data(), patch_total_);
    return obs;
}

VisualManager::Observation::Ptr VisualManager::SelectReference(const VisualPoint& vp) const {
    if (vp.observation.empty()) return nullptr;
    if (vp.observation.size() == 1) return vp.observation.front();
    Observation::Ptr best = nullptr;
    double best_error = std::numeric_limits<double>::max();
    for (const Observation::Ptr& cand : vp.observation) {
        double error = 0.0;
        for (const Observation::Ptr& other : vp.observation) {
            if (other == cand) continue;
            for (int j = 0; j < patch_total_; ++j) {
                const double d = cand->inv_expo * cand->patch[j] - other->inv_expo * other->patch[j];
                error += d * d;
            }
        }
        if (error < best_error) {
            best_error = error;
            best = cand;
        }
    }
    return best;
}

void VisualManager::Retrieve(const cv::Mat& gray, const Pose3& camera_from_world,
                             const std::vector<PointWithNormal>& scan) {
    tracked_.clear();
    std::fill(cell_type_.begin(), cell_type_.end(), kUnknown);
    if (map_.empty()) return;

    // the voxels the LiDAR just saw, and a sparse depth image for the occlusion test
    std::unordered_set<VOXEL_LOCATION> scan_voxels;
    cv::Mat depth = cv::Mat::zeros(gray.rows, gray.cols, CV_32FC1);
    for (const PointWithNormal& point : scan) {
        scan_voxels.insert(VOXEL_LOCATION(point.xyz, prm_.voxel_size));
        const Vec3 pc = camera_from_world * point.xyz;
        const auto px = cam_->project_and_valid(pc, border_);
        if (px) depth.at<float>(static_cast<int>(px->y()), static_cast<int>(px->x())) = static_cast<float>(pc.z());
    }

    // the nearest map point of every grid cell
    const Vec3 cam_pos = camera_from_world.GetInverse().Trans();
    std::vector<VisualPoint::Ptr> cell_point(cell_type_.size());
    std::vector<double> cell_dist(cell_type_.size(), std::numeric_limits<double>::max());
    for (const VOXEL_LOCATION& vox : scan_voxels) {
        const auto it = map_.find(vox);
        if (it == map_.end()) continue;
        for (const VisualPoint::Ptr& vp : it->second) {
            if (vp->observation.empty()) continue;
            const auto px = cam_->project_and_valid(camera_from_world * vp->xyz, border_);
            if (!px) continue;
            const int idx = CellIdx(*px);
            cell_type_[idx] = kMap;
            const double dist = (cam_pos - vp->xyz).norm();
            if (dist < cell_dist[idx]) {
                cell_dist[idx] = dist;
                cell_point[idx] = vp;
            }
        }
    }

    struct Candidate {
        Tracked tracked;
        std::vector<float> cur_patch;
    };
    std::vector<Candidate> candidates;
    for (size_t idx = 0; idx < cell_type_.size(); ++idx) {
        if (cell_type_[idx] != kMap) continue;
        const VisualPoint::Ptr& vp = cell_point[idx];
        const Vec3 pc = camera_from_world * vp->xyz;
        const Vec2 px = cam_->project(pc);

        // a LiDAR return at a clearly different depth inside the patch: the point
        // sits on a depth edge or is occluded
        bool depth_discontinuous = false;
        for (int dv = -half_patch_; dv <= half_patch_ && !depth_discontinuous; ++dv) {
            for (int du = -half_patch_; du <= half_patch_; ++du) {
                if (du == 0 && dv == 0) continue;
                const float d = depth.at<float>(static_cast<int>(px.y()) + dv, static_cast<int>(px.x()) + du);
                if (d == 0.0f) continue;
                if (std::abs(pc.z() - d) > 0.5) {
                    depth_discontinuous = true;
                    break;
                }
            }
        }
        if (depth_discontinuous) continue;

        if (!vp->ref_obs) vp->ref_obs = SelectReference(*vp);
        const Observation::Ptr& ref = vp->ref_obs;

        const Vec3 normal_ref = (ref->camera_from_world.Mat3d() * vp->normal).normalized();
        const Vec3 pos_ref = ref->camera_from_world * vp->xyz;
        if (prm_.max_view_angle > 0.0) {
            // the normal points away from the camera, so head-on is cos = 1
            const double min_cos = std::cos(prm_.max_view_angle * M_PI / 180.0);
            const Vec3 normal_cur = (camera_from_world.Mat3d() * vp->normal).normalized();
            if (pc.normalized().dot(normal_cur) < min_cos || pos_ref.normalized().dot(normal_ref) < min_cos) continue;
        }
        const Pose3 T_cur_ref = camera_from_world * ref->camera_from_world.GetInverse();
        const Mat2 A_cur_ref = HomographyAffine(ref->px, pos_ref, normal_ref, T_cur_ref);
        double D = A_cur_ref.determinant();
        if (!A_cur_ref.allFinite() || std::abs(D) < 1e-6) continue;

        Candidate cand;
        cand.tracked.vp = vp;
        cand.tracked.ref_inv_expo = ref->inv_expo;
        // pyramid level at which the warped patch is about one pixel per pixel
        while (D > 3.0 && cand.tracked.search_level < 2) {
            cand.tracked.search_level++;
            D *= 0.25;
        }
        cand.tracked.warp_patch.resize(patch_total_ * prm_.patch_pyramid_level);
        for (int level = 0; level < prm_.patch_pyramid_level; ++level) {
            WarpAffine(A_cur_ref, ref->img, ref->px, cand.tracked.search_level, level, cand.tracked.warp_patch.data());
        }
        cand.cur_patch.resize(patch_total_);
        PatchFromImage(gray, px, cand.cur_patch.data());

        candidates.push_back(std::move(cand));
    }

    for (Candidate& cand : candidates) {
        // Residuals are formed in the intensity units of the CURRENT image, so the
        // arbitrary overall scale of the exposure chain never reaches the weights.
        cand.tracked.gain = cand.tracked.ref_inv_expo / cur_inv_expo_;
        double error = 0.0;
        for (int j = 0; j < patch_total_; ++j) {
            const double d = cand.tracked.gain * cand.tracked.warp_patch[j] - cand.cur_patch[j];
            error += d * d;
        }
        if (error > prm_.outlier_threshold * patch_total_) continue;
        tracked_.push_back(std::move(cand.tracked));
    }
}

bool VisualManager::PatchResidual(int i, const cv::Mat& gray, const Pose3& camera_from_world, int level,
                                  double inv_expo, Eigen::Ref<VecX> res,
                                  Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 2>> G, Eigen::Ref<VecX> d_expo,
                                  Eigen::Matrix<double, 2, 3>* J_proj) const {
    const Tracked& t = tracked_[i];
    const int scale = 1 << (level + t.search_level);
    const Vec3 pc = camera_from_world * t.vp->xyz;
    if (pc.z() < 1e-3) return false;
    const Vec2 px = cam_->project(pc);
    if (!px.allFinite()) return false;

    // Sampling happens on the full-resolution image with a stride of `scale`.
    const int u_i = static_cast<int>(std::floor(px[0] / scale)) * scale;
    const int v_i = static_cast<int>(std::floor(px[1] / scale)) * scale;
    // the gradient reaches one stride beyond the patch on the low side and two on the high side
    const int reach = (half_patch_ + 1) * scale;
    if (u_i - reach < 0 || v_i - reach < 0 || u_i + reach >= gray.cols || v_i + reach >= gray.rows) return false;

    const float sub_u = static_cast<float>(px[0] - u_i) / scale;
    const float sub_v = static_cast<float>(px[1] - v_i) / scale;
    const float w_tl = (1.0f - sub_u) * (1.0f - sub_v);
    const float w_tr = sub_u * (1.0f - sub_v);
    const float w_bl = (1.0f - sub_u) * sub_v;
    const float w_br = sub_u * sub_v;
    const int stride = static_cast<int>(gray.step.p[0]);
    const auto sample = [&](const uint8_t* p) {
        return w_tl * p[0] + w_tr * p[scale] + w_bl * p[scale * stride] + w_br * p[scale * stride + scale];
    };

    *J_proj = ProjectJacobian(pc);
    const float* ref_patch = t.warp_patch.data() + patch_total_ * level;
    const double inv_scale = 1.0 / scale;
    const double gain = t.ref_inv_expo / inv_expo;
    for (int r = 0; r < prm_.patch_size; ++r) {
        const uint8_t* ptr = gray.data + (v_i + (r - half_patch_) * scale) * stride + (u_i - half_patch_ * scale);
        for (int c = 0; c < prm_.patch_size; ++c, ptr += scale) {
            const double du = 0.5 * (sample(ptr + scale) - sample(ptr - scale));
            const double dv = 0.5 * (sample(ptr + scale * stride) - sample(ptr - scale * stride));
            const int row = r * prm_.patch_size + c;
            res(row) = sample(ptr) - gain * ref_patch[row];
            d_expo(row) = gain * ref_patch[row] / inv_expo;
            G.row(row) = Eigen::Vector2d(du, dv).transpose() * inv_scale;
        }
    }
    return true;
}

void VisualManager::UpdateMap(const cv::Mat& gray, const Pose3& camera_from_world,
                              const std::vector<PointWithNormal>& scan, const LocalBA& map) {
    GeneratePoints(gray, camera_from_world, scan);
    UpdateObservations(gray, camera_from_world);
    UpdateReferences(map);
}

void VisualManager::GeneratePoints(const cv::Mat& gray, const Pose3& camera_from_world,
                                   const std::vector<PointWithNormal>& scan) {
    // per cell without a map point, the LiDAR point sitting on the best corner
    std::vector<int> cell_scan(cell_type_.size(), -1);
    std::vector<float> cell_score(cell_type_.size(), 0.0f);
    for (size_t i = 0; i < scan.size(); ++i) {
        if (scan[i].normal == Vec3::Zero()) continue;
        const auto px = cam_->project_and_valid(camera_from_world * scan[i].xyz, border_);
        if (!px) continue;
        const int idx = CellIdx(*px);
        if (cell_type_[idx] == kMap) continue;
        const float score = ShiTomasiScore(gray, static_cast<int>(px->x()), static_cast<int>(px->y()));
        if (score > cell_score[idx]) {
            cell_score[idx] = score;
            cell_scan[idx] = static_cast<int>(i);
            cell_type_[idx] = kPointCloud;
        }
    }

    last_added_ = 0;
    for (size_t idx = 0; idx < cell_type_.size(); ++idx) {
        if (cell_type_[idx] != kPointCloud) continue;
        const PointWithNormal& point = scan[cell_scan[idx]];
        const Vec3 pc = camera_from_world * point.xyz;
        const Vec2 px = cam_->project(pc);

        VisualPoint::Ptr vp = std::make_shared<VisualPoint>();
        vp->xyz = point.xyz;
        // keep the normal pointing away from the camera, as the homography assumes
        const Vec3 normal_c = camera_from_world.Mat3d() * point.normal;
        vp->normal = pc.normalized().dot(normal_c) < 0.0 ? Vec3(-point.normal) : point.normal;
        vp->observation.push_back(MakeObservation(gray, camera_from_world, px, 0));
        vp->ref_obs = vp->observation.back();
        map_[VOXEL_LOCATION(vp->xyz, prm_.voxel_size)].push_back(vp);
        last_added_++;
    }
}

void VisualManager::UpdateObservations(const cv::Mat& gray, const Pose3& camera_from_world) {
    for (Tracked& t : tracked_) {
        VisualPoint& vp = *t.vp;
        // a converged point only ever needs its reference patch
        if (vp.is_converged) {
            vp.observation.remove_if([&vp](const Observation::Ptr& obs) { return obs != vp.ref_obs; });
            continue;
        }
        const auto px = cam_->project_and_valid(camera_from_world * vp.xyz, border_);
        if (!px) continue;

        // a new observation is worth keeping once the view has changed enough
        const Observation::Ptr& last = vp.observation.back();
        const Pose3 delta = last->camera_from_world * camera_from_world.GetInverse();
        const double delta_p = delta.Trans().norm();
        const double trace = delta.Mat3d().trace();
        const double delta_theta = (trace > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (trace - 1.0));
        const bool add = delta_p > 0.5 || delta_theta > 0.3 || (*px - last->px).norm() > 40.0;
        if (!add) continue;

        if (vp.observation.size() >= 30) {
            const auto worst = std::min_element(
                vp.observation.begin(), vp.observation.end(),
                [](const Observation::Ptr& a, const Observation::Ptr& b) { return a->score < b->score; });
            if (*worst == vp.ref_obs) vp.ref_obs = nullptr;  // Retrieve() picks a new one
            vp.observation.erase(worst);
        }
        vp.observation.push_back(MakeObservation(gray, camera_from_world, *px, t.search_level));
        t.obs_added = true;
    }
}

void VisualManager::UpdateReferences(const LocalBA& map) {
    for (Tracked& t : tracked_) {
        VisualPoint& vp = *t.vp;
        if (vp.is_converged || !t.obs_added || vp.observation.size() <= 5) continue;

        Vec3 normal;
        if (map.PlaneNormal(vp.xyz, &normal)) {
            if (normal.dot(vp.normal) < 0.0) normal = -normal;
            if (prm_.converge_en && (normal - vp.normal).norm() < 1e-4 && vp.observation.size() > 10) {
                vp.is_converged = true;
            }
            vp.normal = normal;
        }

        // the reference is the patch that correlates best with all the others and
        // looks at the surface most head-on
        float best_score = -std::numeric_limits<float>::max();
        for (const Observation::Ptr& cand : vp.observation) {
            float ncc_sum = 0.0f;
            int count = 0;
            for (const Observation::Ptr& other : vp.observation) {
                if (other == cand) continue;
                float up = 0.0f, down1 = 0.0f, down2 = 0.0f;
                for (int j = 0; j < patch_total_; ++j) {
                    const float a = cand->patch[j] - cand->mean;
                    const float b = other->patch[j] - other->mean;
                    up += a * b;
                    down1 += a * a;
                    down2 += b * b;
                }
                ncc_sum += std::abs(up) / std::sqrt(down1 * down2 + 1e-10f);
                count++;
            }
            const Vec3 view = (cand->camera_from_world * vp.xyz).normalized();
            const Vec3 normal_c = cand->camera_from_world.Mat3d() * vp.normal;
            cand->score = ncc_sum / std::max(count, 1) + static_cast<float>(view.dot(normal_c));
            if (cand->score > best_score) {
                best_score = cand->score;
                vp.ref_obs = cand;
            }
        }
    }
}

}  // namespace faster_lio

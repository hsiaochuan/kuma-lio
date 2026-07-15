//
// Created by hsiaochuan on 2026/05/11.
//

#include "visual_manager.h"
#include <ros/param.h>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
void VisualManager::Run(cv::Mat& raw_im, std::vector<PointWithNormal>& scan) {
    // Reset per-frame containers/state before collecting new data.
    sparse_map.visual_points.clear();
    sparse_map.search_levels.clear();
    sparse_map.warp_patch.clear();
    std::fill(grid_states.begin(), grid_states.end(), UNKOWN);
    camera_from_world = (Pose3(state_point_->rot, state_point_->pos) * param->extrin_ic_).GetInverse();
    cv::Mat mono_im;
    if (raw_im.channels() == 3)
        cv::cvtColor(raw_im,mono_im,cv::COLOR_BGR2GRAY);
    else
        raw_im.copyTo(mono_im);
    RetrieveFromSparseMap(mono_im, scan);
    UpdateEKF(mono_im);
    GenerateVisualPoints(mono_im, scan);
    UpdateVisualPointObserv(mono_im);
    UpdateVisualPointNormal();
    UpdateReferencePatch();
}
void VisualManager::Initialize() {
    grid_n_h = ceil(static_cast<double>(param->camera_->h() / options_.grid_size));
    grid_n_w = ceil(static_cast<double>(param->camera_->w() / options_.grid_size));
    total_grids = grid_n_h * grid_n_w;
    grid_states.resize(grid_n_h * grid_n_w, UNKOWN);
    Jdphi_dR = param->extrin_ic_.Mat3d().transpose();
    Jdp_dR = -param->extrin_ic_.Mat3d().transpose() * Hat(param->extrin_ic_.Trans());
}
void VisualManager::RetrieveFromSparseMap(cv::Mat& img, std::vector<PointWithNormal>& scan) {
    // use scan to extract the visible voxels
    std::unordered_set<VOXEL_LOCATION> scan_voxels;
    for (auto& point : scan) {
        VOXEL_LOCATION vox_loc(point.xyz, options_.visual_voxel_res);
        scan_voxels.insert(vox_loc);
    }

    // construct the depth image
    MatX d_im;
    d_im.setConstant(param->camera_->h(), param->camera_->w(), std::numeric_limits<double>::max());
    for (auto& point : scan) {
        Vec3 pc = camera_from_world * point.xyz;
        auto p_im = param->camera_->project_and_valid(pc, options_.border);
        if (p_im) {
            Vec2i uv = p_im->cast<int>();
            d_im(uv.y(), uv.x()) = pc.z();
        }
    }

    // visible visual points save to grid
    std::vector<VisualPoint::Ptr> grid_corres_vpoints(total_grids);
    std::vector<double> grid_distances(total_grids, std::numeric_limits<double>::max());
    for (auto& vox : scan_voxels) {
        if (visual_points_map.find(vox) != visual_points_map.end()) {
            // this voxel is visible
            std::vector<VisualPoint::Ptr>& visual_points = visual_points_map[vox];
            for (int i = 0; i < visual_points.size(); ++i) {
                VisualPoint::Ptr vp = visual_points[i];
                Vec3 pc = camera_from_world * vp->xyz;
                auto p_im = param->camera_->project_and_valid(pc, options_.border);
                if (!p_im) continue;
                int grid_idx = GridIdx(*p_im);
                grid_states[grid_idx] = VISUAL_POINT;
                double point_to_grid_dist = (camera_from_world.GetInverse().Trans() - vp->xyz).norm();
                if (point_to_grid_dist < grid_distances[grid_idx]) {
                    grid_distances[grid_idx] = point_to_grid_dist;
                    grid_corres_vpoints[grid_idx] = vp;
                }
            }
        }
    }

    for (int i = 0; i < total_grids; i++) {
        if (grid_states[i] != VISUAL_POINT) continue;
        VisualPoint::Ptr vp = grid_corres_vpoints[i];
        Vec3 pc = camera_from_world * vp->xyz;
        auto p_im = param->camera_->project_and_valid(pc, options_.border);
        CHECK(p_im);

        // depth continuous check
        bool depth_discontinu = false;
        for (int u = -options_.half_p_size; u <= options_.half_p_size; ++u) {
            for (int v = -options_.half_p_size; v <= options_.half_p_size; ++v) {
                if (u == 0 && v == 0) continue;
                Vec2 uv(p_im->x() + u, p_im->y() + v);
                double d = d_im(uv.y(), uv.x());
                if (d == std::numeric_limits<double>::max()) continue;
                double delta_depth = std::abs(pc[2] - d);
                if (delta_depth > 0.5) {
                    depth_discontinu = true;
                    break;
                }
            }
            if (depth_discontinu) break;
        }
        if (depth_discontinu) continue;


        // get the affine with current and reference
        PatchObservation::Ptr ref_obs = vp->ref_obs;
        Vec3 normal_ref = ref_obs->camera_from_world.Mat3d() * vp->normal;
        Vec3 pos_ref = ref_obs->camera_from_world * vp->xyz;
        Pose3 T_cur_ref = camera_from_world * ref_obs->camera_from_world.GetInverse();
        Mat2 A_cur_ref = HomographyAffineMatrix(param->camera_, ref_obs->px, pos_ref, normal_ref, T_cur_ref, ref_obs->level);

        // determine the search level
        int search_level = 0;
        double D = A_cur_ref.determinant();
        while (D > 3.0 && search_level < 2) {
            search_level++;
            D *= 0.25;
        }

        // affine the reference patch
        std::vector<float> affine_ref_patch(options_.total_patch_size * options_.patch_pyrimid_level, 0.0f);
        for (int level = 0; level < options_.patch_pyrimid_level; level++) {
            WarpAffine(A_cur_ref, ref_obs->img, ref_obs->px, search_level, level, affine_ref_patch.data());
        }

        // get the current patch
        std::vector<float> cur_patch(options_.total_patch_size);
        PatchFromImage(img, *p_im, cur_patch.data(), 0);

        // calculate the error
        double error = 0.0;
        for (int j = 0; j < options_.total_patch_size; j++) {
            error += (cur_patch[j] - affine_ref_patch[j]) * (cur_patch[j] - affine_ref_patch[j]);
        }
        if (error > options_.outlier_thr * options_.outlier_thr) continue;
        sparse_map.visual_points.push_back(vp);
        sparse_map.search_levels.push_back(search_level);
        sparse_map.warp_patch.push_back(affine_ref_patch);
    }
    std::cout << "sparse map size: " << sparse_map.visual_points.size() << std::endl;
}

void VisualManager::UpdateEKF(cv::Mat& img) {
    if (sparse_map.visual_points.empty())
        return;
    for (int level = options_.patch_pyrimid_level - 1; level >= 0; level--) {
        UpdateEKFInLevel(img, level);
    }
    camera_from_world = (Pose3(state_point_->rot, state_point_->pos) * param->extrin_ic_).GetInverse();
}
static Eigen::Matrix<double,2,3> PinholeJaco(const Vec3& pc, CamModel::Ptr & cam) {
    Eigen::Matrix<double,2,3> J;
    double fx = 0.0;
    double fy = 0.0;
    if (cam->get_type() == PINHOLE) {
        PinholeCamera::Ptr ph_cam = std::dynamic_pointer_cast<PinholeCamera>(cam);
        CHECK(ph_cam);
        fx = ph_cam->fx_;
        fy = ph_cam->fy_;
    } else if (cam->get_type() == PINHOLE_RADIAL) {
        PinholeRadialCamera::Ptr ph_cam = std::dynamic_pointer_cast<PinholeRadialCamera>(cam);
        CHECK(ph_cam);
        fx = ph_cam->fx_;
        fy = ph_cam->fy_;
    } else if (cam->get_type() == PINHOLE_FISHEYE) {
        PinholeFisheyeCamera::Ptr ph_cam = std::dynamic_pointer_cast<PinholeFisheyeCamera>(cam);
        CHECK(ph_cam);
        fx = ph_cam->fx_;
        fy = ph_cam->fy_;
    } else {
        throw std::runtime_error("Unknown camera type");
    }
    const double x = pc[0];
    const double y = pc[1];
    const double z_inv = 1. / pc[2];
    const double z_inv_2 = z_inv * z_inv;
    J(0, 0) = fx * z_inv;
    J(0, 1) = 0.0;
    J(0, 2) = -fx * x * z_inv_2;
    J(1, 0) = 0.0;
    J(1, 1) = fy * z_inv;
    J(1, 2) = -fy * y * z_inv_2;
    return J;
}
void VisualManager::UpdateEKFInLevel(cv::Mat& img, int level) {
    StatePoint old_state = *state_point_;
    int H_DIM = sparse_map.visual_points.size() * options_.total_patch_size;
    VecX z;
    MatX H, HTRinv;
    MatX K;
    z.setZero(H_DIM);
    H.setZero(H_DIM, StatePoint::STATE_DOF);
    HTRinv.setZero(StatePoint::STATE_DOF, H_DIM);
    float last_error = std::numeric_limits<float>::max();
    for (int iteration = 0; iteration < options_.max_iterations; iteration++) {
        Pose3 cam_from_world = (Pose3(state_point_->rot, state_point_->pos) * param->extrin_ic_).GetInverse();
        Jdp_dt = param->extrin_ic_.Mat3d().transpose() * state_point_->rot.toRotationMatrix().transpose();
        float error = 0.0;
        for (int i = 0; i < sparse_map.visual_points.size(); i++) {
            Eigen::Matrix<double, 1, 2> Jimg;
            Eigen::Matrix<double, 2, 3> Jdpi;
            Eigen::Matrix<double, 1, 3> Jdphi, Jdp, JdR, Jdt;

            int scale = (1 << (level + sparse_map.search_levels[i]));
            float inv_scale = 1.0f / scale;

            VisualPoint::Ptr vp = sparse_map.visual_points[i];
            Vec3 p_c = cam_from_world * vp->xyz;
            Mat3 p_c_hat = Hat(p_c);
            auto p_im = param->camera_->project_and_valid(p_c, options_.border);
            if (!p_im)
                continue;
            Jdpi = PinholeJaco(p_c, param->camera_);

            const float u_ref = (*p_im)[0];
            const float v_ref = (*p_im)[1];
            const int u_ref_i = floorf(u_ref / scale) * scale;
            const int v_ref_i = floorf(v_ref / scale) * scale;
            const float subpix_u_ref = (u_ref - u_ref_i) / scale;
            const float subpix_v_ref = (v_ref - v_ref_i) / scale;
            const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
            const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;
            std::vector<float>& aff_ref_patch = sparse_map.warp_patch[i];

            // top-left point of reference patch
            const int base_u = u_ref_i - options_.half_p_size * scale;
            const int base_v = v_ref_i - options_.half_p_size * scale;

            auto pixel = [&](int v, int u) -> float {
                return (float)img.at<uint8_t>(v,u);
            };
            auto bilinear = [&](int v, int u) -> float {
                return w_ref_tl * pixel(v, u) + w_ref_tr * pixel(v, u + scale) +
                       w_ref_bl * pixel(v + scale, u) + w_ref_br * pixel(v + scale, u + scale);
            };
            auto patch_at = [&](int x, int y) -> float& {
                return aff_ref_patch[options_.total_patch_size * level + x * options_.patch_size + y];
            };

            for (int x = 0; x < options_.patch_size; x++) {
                const int v = base_v + x * scale;
                for (int y = 0; y < options_.patch_size; ++y) {
                    const int u = base_u + y * scale;
                    const float du = 0.5f * (bilinear(v, u + scale) - bilinear(v, u - scale));
                    const float dv = 0.5f * (bilinear(v + scale, u) - bilinear(v - scale, u));

                    Jimg << du, dv;
                    // Jimg = Jimg * state->inv_expo_time;
                    Jimg = Jimg * inv_scale;
                    Jdphi = Jimg * Jdpi * p_c_hat;
                    Jdp = -Jimg * Jdpi;
                    JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                    Jdt = Jdp * Jdp_dt;

                    const double cur_value = bilinear(v, u);
                    const double res = cur_value - patch_at(x, y);

                    const int row = i * options_.total_patch_size + x * options_.patch_size + y;
                    z(row) = res;
                    H.block<1, 3>(row, 0) = JdR;
                    H.block<1, 3>(row, 3) = Jdt;
                    HTRinv.col(row) = H.row(row).transpose() / options_.img_point_cov;
                } // for patch pixel
            } // for patch pixel
        } // for points

        error = z.dot(z);
        error = error / z.rows();
        if (error < last_error) {
            old_state = *state_point_;
            last_error = error;
            K = (HTRinv * H + state_point_->cov.inverse()).inverse() * HTRinv;
            StatePoint::VectorN vec = state_predict_ - *state_point_;
            StatePoint::VectorN solution = K * (-z - H * vec ) + vec;
            *state_point_ += solution;
        }else {
            *state_point_ = old_state;
        }
    } // for iter


    if (level == 0) {
        static StatePoint::MatrixN state_iden_mat = StatePoint::MatrixN::Identity();
        state_point_->cov = (state_iden_mat - K * H) * state_point_->cov;
    }
}
static float
interpolateMat_8u(const cv::Mat& mat, float u, float v)
{
    assert(mat.type()==CV_8U);
    int x = floor(u);
    int y = floor(v);
    float subpix_x = u-x;
    float subpix_y = v-y;

    float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
    float w01 = (1.0f-subpix_x)*subpix_y;
    float w10 = subpix_x*(1.0f-subpix_y);
    float w11 = 1.0f - w00 - w01 - w10;

    const int stride = mat.step.p[0];
    unsigned char* ptr = mat.data + y*stride + x;
    return w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride+1];
}
void VisualManager::WarpAffine(const Mat2& A_cur_ref, const cv::Mat& img_ref, const Vec2& px_ref,
                               const int search_level, int pyr_level, float* patch) {
    const int patch_size = options_.half_p_size * 2;
    const Mat2f A_ref_cur = A_cur_ref.inverse().cast<float>();

    float* patch_ptr = patch;
    for (int y = 0; y < patch_size; ++y) {
        for (int x = 0; x < patch_size; ++x)  //, ++patch_ptr)
        {
            Vec2f du(x - options_.half_p_size, y - options_.half_p_size);
            du *= (1 << search_level);
            du *= (1 << pyr_level);
            const Vec2f px(A_ref_cur * du + px_ref.cast<float>());
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                patch_ptr[options_.total_patch_size * pyr_level + y * patch_size + x] = 0;
            else
                patch_ptr[options_.total_patch_size * pyr_level + y * patch_size + x] =
                    (float)interpolateMat_8u(img_ref, px[0], px[1]);
        }
    }
}
Mat2 VisualManager::AffineMatrix(CamModel::Ptr& cam_model, const Vec2& px, const Vec3& xyz_ref, const Pose3& T_cur_ref,
                                 const int& level) {
    Vec2 pu = px + Vec2(options_.half_p_size, 0) * (1 << level);
    Vec2 pv = px + Vec2(0, options_.half_p_size) * (1 << level);
    Vec3 u_bearing = cam_model->bearing(pu);
    Vec3 v_bearing = cam_model->bearing(pv);
    Vec3 xyz_du = u_bearing * (xyz_ref[2] / u_bearing[2]);
    Vec3 xyz_dv = v_bearing * (xyz_ref[2] / v_bearing[2]);
    Vec2 px_cur = cam_model->project(T_cur_ref * xyz_ref);
    Vec2 px_du = cam_model->project(T_cur_ref * xyz_du);
    Vec2 px_dv = cam_model->project(T_cur_ref * xyz_dv);
    Mat2 A_cur_ref;
    A_cur_ref.col(0) = (px_du - px_cur) / options_.half_p_size;
    A_cur_ref.col(1) = (px_dv - px_cur) / options_.half_p_size;
    return A_cur_ref;
}
Mat2 VisualManager::HomographyAffineMatrix(CamModel::Ptr& cam_model, const Vec2& px, const Vec3& xyz_ref,
                                           const Vec3& normal_ref, const Pose3& T_cur_ref, const int& level) {
    Mat2 A_cur_ref = Mat2::Identity();
    const Vec3 t = T_cur_ref.GetInverse().Trans();
    const Eigen::Matrix3d H_cur_ref =
        T_cur_ref.Mat3d() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
    Vec3 f_du_ref = cam_model->bearing(px + Eigen::Vector2d(options_.half_p_size, 0) * (1 << level));
    Vec3 f_dv_ref = cam_model->bearing(px + Eigen::Vector2d(0, options_.half_p_size) * (1 << level));
    const Vec3 f_cur(H_cur_ref * xyz_ref);
    const Vec3 f_du_cur = H_cur_ref * f_du_ref;
    const Vec3 f_dv_cur = H_cur_ref * f_dv_ref;
    Vec2 px_cur(cam_model->project(f_cur));
    Vec2 px_du_cur(cam_model->project(f_du_cur));
    Vec2 px_dv_cur(cam_model->project(f_dv_cur));
    A_cur_ref.col(0) = (px_du_cur - px_cur) / options_.half_p_size;
    A_cur_ref.col(1) = (px_dv_cur - px_cur) / options_.half_p_size;
    return A_cur_ref;
}
static float
shiTomasiScore(const cv::Mat& img, int u, int v)
{
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2*halfbox_size;
    const int box_area = box_size*box_size;
    const int x_min = u-halfbox_size;
    const int x_max = u+halfbox_size;
    const int y_min = v-halfbox_size;
    const int y_max = v+halfbox_size;

    if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for( int y=y_min; y<y_max; ++y )
    {
        const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
        {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx*dx;
            dYY += dy*dy;
            dXY += dx*dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
}
void VisualManager::GenerateVisualPoints(cv::Mat& img, std::vector<PointWithNormal>& scan_points) {
    std::vector<PointWithNormal> grid_corres_points(total_grids);
    std::vector<float> grid_score(total_grids, std::numeric_limits<float>::min());
    for (int i = 0; i < scan_points.size(); i++) {
        if (scan_points[i].normal == Vec3::Zero())
            continue;
        Vec3 pw = scan_points[i].xyz;
        Vec3 pc = camera_from_world * pw;
        auto p_im = param->camera_->project_and_valid(pc, options_.border);
        if (!p_im) continue;
        int grid_idx = GridIdx(*p_im);
        if (grid_states[grid_idx] != VISUAL_POINT) {
            // only generate visual points in no visual points grid
            float score = shiTomasiScore(img, p_im->x(), p_im->y());
            if (score > grid_score[grid_idx]) {
                grid_score[grid_idx] = score;
                grid_corres_points[grid_idx] = scan_points[i];
                grid_states[grid_idx] = TYPE_POINTCLOUD;
            }
        }
    }
    int add_vp_count = 0;
    for (int i = 0; i < total_grids; i++) {
        if (grid_states[i] != TYPE_POINTCLOUD) continue;

        PointWithNormal &point = grid_corres_points[i];
        Vec3 pc = camera_from_world * point.xyz;
        auto p_im = param->camera_->project_and_valid(pc, options_.border);
        if (!p_im)
            continue;

        // create visual point
        VisualPoint::Ptr vp = std::make_shared<VisualPoint>();
        vp->xyz = point.xyz;
        vp->normal = point.normal;
        Vec3 normal_c = camera_from_world.Mat3d() * vp->normal;
        double cos_theta = pc.normalized().dot(normal_c);
        if (cos_theta < 0.0)
            vp->normal = -vp->normal;
        // create observation
        PatchObservation::Ptr new_obs = std::make_shared<PatchObservation>();
        new_obs->img = img;
        new_obs->camera_from_world = camera_from_world;
        new_obs->px = *p_im;
        new_obs->bearing = param->camera_->bearing(*p_im);
        new_obs->patch.resize(options_.total_patch_size);
        PatchFromImage(img, *p_im, new_obs->patch.data(), 0);
        float sum = std::accumulate(new_obs->patch.begin(), new_obs->patch.end(), 0.0f);
        new_obs->mean = sum / static_cast<float>(options_.total_patch_size);
        vp->observation.push_back(new_obs);
        vp->ref_obs = new_obs;

        VOXEL_LOCATION vox_loc(vp->xyz, options_.visual_voxel_res);
        visual_points_map[vox_loc].push_back(vp);
        add_vp_count++;
    }
    std::cout << "Generate " << add_vp_count << " visual points." << std::endl;
}
void VisualManager::UpdateVisualPointObserv(cv::Mat& img) {
    for (int i = 0; i < sparse_map.visual_points.size(); ++i) {
        VisualPoint::Ptr vp = sparse_map.visual_points[i];
        // converge, not add, delete other observations
        if (vp->is_converged) {
            for (auto it = vp->observation.begin(); it != vp->observation.end();) {
                if (*it == vp->ref_obs) {
                    ++it;
                } else {
                    it = vp->observation.erase(it);
                }
            }
            continue;
        }
        Vec3 pc = camera_from_world * vp->xyz;
        auto p_im = param->camera_->project_and_valid(pc, options_.border);
        if (!p_im)
            continue;
        bool add_flag = false;

        // pose delta is enough
        PatchObservation::Ptr last_obs = vp->observation.back();
        Pose3 last_cw = last_obs->camera_from_world;
        Pose3 curr_cw = camera_from_world;
        Pose3 delta = last_cw * curr_cw.GetInverse();
        double delta_p = delta.Trans().norm();
        double delta_theta = (delta.Mat3d().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta.Mat3d().trace() - 1));
        if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

        // pixel delta is enough
        Vec2 last_pixel = vp->observation.back()->px;
        double pixel_dist = (*p_im - last_pixel).norm();
        if (pixel_dist > 40.0)
            add_flag = true;

        // maintain the size
        if (vp->observation.size() >= 30) {
            // find the min score observation
            float min_score = std::numeric_limits<float>::max();
            PatchObservation::Ptr min_obs = *vp->observation.begin();
            for (auto it = vp->observation.begin(); it != vp->observation.end(); ++it) {
                if ((*it)->score < min_score) {
                    min_score = (*it)->score;
                    min_obs = *it;
                }
            }

            // delete the min score observation
            for (auto it = vp->observation.begin(); it != vp->observation.end();) {
                if ((*it) == min_obs) {
                    if (*it == vp->ref_obs)
                        vp->ref_obs.reset();
                    it = vp->observation.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (add_flag) {
            PatchObservation::Ptr new_obs = std::make_shared<PatchObservation>();
            new_obs->img = img;
            new_obs->camera_from_world = camera_from_world;
            new_obs->px = *p_im;
            new_obs->bearing = param->camera_->bearing(*p_im);
            new_obs->level = sparse_map.search_levels[i];
            new_obs->patch.resize(options_.total_patch_size);
            PatchFromImage(img,*p_im,new_obs->patch.data(), 0);
            vp->observation.push_back(new_obs);
        }
    }
}
void VisualManager::UpdateVisualPointNormal() {
    for (auto & vp : sparse_map.visual_points) {
        if (vp->is_converged)
            continue;
        PointVector points_near;
        Point point;
        point.getVector3fMap() = vp->xyz.cast<float>();
        ivox_->GetClosestPoint(point, points_near, options::NUM_MATCH_POINTS);
        if (points_near.size() >= options::MIN_NUM_MATCH_POINTS) {
            Vec4f plane_coeffs;
            bool est = esti_plane(plane_coeffs, points_near, param->esti_plane_thr);
            if (!est)
                continue;
            Vec3 lidar_normal = plane_coeffs.head<3>().cast<double>();
            if (lidar_normal.dot(vp->normal) < 0)
                lidar_normal = -lidar_normal;
            // TODO: use S2 to model the update
            double normal_update = (vp->normal - lidar_normal).norm();
            if (normal_update < 0.0001 && vp->observation.size() > 10)
                vp->is_converged = true;
            else
                vp->normal = lidar_normal;
        }
    }
}
void VisualManager::UpdateReferencePatch() {
    for (auto &vp : sparse_map.visual_points) {
        if (vp->is_converged)
            continue;
        double best_score = std::numeric_limits<float>::min();
        PatchObservation::Ptr best_obs = nullptr;
        for (auto it = vp->observation.begin(); it != vp->observation.end(); ++it) {
            PatchObservation::Ptr tmp_ref_obs = *it;
            float ncc_up = 0.0;
            float ncc_down1 = 0.0;
            float ncc_down2 = 0.0;
            float ncc = 0.0;
            float score = 0.0;
            int count = 0;

            Vec3 p_ref = tmp_ref_obs->camera_from_world * vp->xyz;
            Vec3 normal_ref = tmp_ref_obs->camera_from_world.Mat3d() * vp->normal;
            p_ref.normalize();
            double cos_angle = p_ref.dot(normal_ref);

            for (auto it2 = vp->observation.begin(); it2 != vp->observation.end(); ++it2) {
                PatchObservation::Ptr obs = *it2;
                if (obs == tmp_ref_obs) continue;

                for (int i = 0; i < options_.total_patch_size; ++i) {
                    ncc_up += (tmp_ref_obs->patch[i] - tmp_ref_obs->mean) * (obs->patch[i] - obs->mean);
                    ncc_down1 += (tmp_ref_obs->patch[i] - tmp_ref_obs->mean) * (tmp_ref_obs->patch[i] - tmp_ref_obs->mean);
                    ncc_down2 += (obs->patch[i] - obs->mean) * (obs->patch[i] - obs->mean);
                }
                ncc += fabs(ncc_up / sqrt(ncc_down1 * ncc_down2));
                count++;
            } // for obs
            ncc = ncc / count;
            score = ncc + cos_angle;
            tmp_ref_obs->score = score;
            if (score > best_score) {
                best_score = score;
                vp->ref_obs = tmp_ref_obs;
            }
        } // for obs
    } // for vp points
}
void VisualManager::PatchFromImage(cv::Mat& img, Vec2& px, float* patch, int level) {
    const float u_ref = px[0];
    const float v_ref = px[1];
    const int scale = (1 << level);
    const int u_ref_i = floorf(u_ref / scale) * scale;
    const int v_ref_i = floorf(v_ref / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    const int base_u = u_ref_i - options_.half_p_size * scale;
    const int base_v = v_ref_i - options_.half_p_size * scale;

    auto pixel = [&](int v, int u) -> float {
        return (float)img.at<uint8_t>(v,u);
    };
    auto bilinear = [&](int v, int u) -> float {
        return w_ref_tl * pixel(v, u) + w_ref_tr * pixel(v, u + scale) +
               w_ref_bl * pixel(v + scale, u) + w_ref_br * pixel(v + scale, u + scale);
    };

    for (int x = 0; x < options_.patch_size; x++) {
        const int v = base_v + x * scale;
        for (int y = 0; y < options_.patch_size; y++) {
            const int u = base_u + y * scale;
            patch[options_.total_patch_size * level + x * options_.patch_size + y] = bilinear(v, u);
        }
    }
}


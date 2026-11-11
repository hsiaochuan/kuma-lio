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

#include <yaml-cpp/yaml.h>

#include <string>
#include <type_traits>

#include "pgo/pgo_types.h"

namespace kuma_lio {
namespace pgo {

void LoadPgoParam(const std::string& file, PgoParam* p) {
    const YAML::Node y = YAML::LoadFile(file);
    auto get = [&](const char* key, auto* v) {
        if (y[key]) *v = y[key].as<std::decay_t<decltype(*v)>>();
    };
    get("kf_scans", &p->kf_scans);
    get("kf_min_angle", &p->kf_min_angle);
    get("kf_min_dist", &p->kf_min_dist);
    get("kf_ds_voxel", &p->kf_ds_voxel);
    get("plane_voxel", &p->plane_voxel);
    get("plane_min_points", &p->plane_min_points);
    get("plane_min_eigen", &p->plane_min_eigen);
    get("btc_enable", &p->btc_enable);
    get("loop_min_kf_gap", &p->loop_min_kf_gap);
    get("loop_min_travel", &p->loop_min_travel);
    get("loop_radius", &p->loop_radius);
    get("loop_max_dz", &p->loop_max_dz);
    get("loop_max_yaw_diff", &p->loop_max_yaw_diff);
    get("btc_merge_normal", &p->btc_merge_normal);
    get("btc_merge_dist", &p->btc_merge_dist);
    get("btc_ref_planes", &p->btc_ref_planes);
    get("btc_pixel", &p->btc_pixel_res);
    get("btc_layer_height", &p->btc_layer_height);
    get("btc_height_min", &p->btc_height_min);
    get("btc_height_max", &p->btc_height_max);
    get("btc_min_bits", &p->btc_min_bits);
    get("btc_line_filter", &p->btc_line_filter);
    get("btc_nms_radius", &p->btc_nms_radius);
    get("btc_max_keypoints", &p->btc_max_keypoints);
    get("btc_near_num", &p->btc_near_num);
    get("btc_side_min", &p->btc_side_len_min);
    get("btc_side_max", &p->btc_side_len_max);
    get("btc_side_res", &p->btc_side_len_res);
    get("btc_side_tol", &p->btc_side_tol);
    get("btc_similarity", &p->btc_similarity);
    get("btc_candidates", &p->btc_candidates);
    get("btc_min_votes", &p->btc_min_votes);
    get("btc_ransac_dist", &p->btc_ransac_dist);
    get("btc_min_inliers", &p->btc_min_inliers);
    get("btc_overlap_normal", &p->btc_overlap_normal);
    get("btc_overlap_dist", &p->btc_overlap_dist);
    get("btc_min_overlap", &p->btc_min_overlap);
    get("icp_max_iter", &p->icp_max_iter);
    get("icp_eigval", &p->icp_eigval);
    get("icp_min_match", &p->icp_min_match);
    get("ratio_drift", &p->ratio_drift);
    get("opt_cooldown_kf", &p->opt_cooldown_kf);
    get("opt_min_drift", &p->opt_min_drift);
    get("loop_var", &p->loop_var);
    get("prior_var", &p->prior_var);
    get("odom_var_rot", &p->odom_var_rot);
    get("odom_var_trans", &p->odom_var_trans);
    get("map_kfs", &p->map_kfs);
    get("reload_radius", &p->reload_radius);
    get("kf_voxel", &p->kf_voxel);
}


}  // namespace pgo
}  // namespace kuma_lio

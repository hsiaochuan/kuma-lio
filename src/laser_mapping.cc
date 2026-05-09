#include <execution>
#include <iomanip>
#include <sstream>

#include <boost/filesystem.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include "global_optimizor.h"
#include "laser_mapping.h"
#include "utils.h"

namespace fs = boost::filesystem;
namespace faster_lio {

void LaserMapping::Run() {
    // sync the lidar and imu data, if no data or not synced, return true
    if (!SyncPackages()) {
        return;
    }

    /// IMU process, kf prediction, undistortion
    PointCloud::Ptr scan_body(new PointCloud);
    pcl::transformPointCloud(*measures_.lidar_, *scan_body, param->extrin_il_.Mat4d());
    p_imu_->Process(measures_, kf_, scan_body, *scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }

    /// the first scan
    if (if_local_map_init_) {
        state_point_ = kf_.get_x();
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            scan_down_world_->at(i).getVector3fMap() =
                (state_point_.rot * scan_undistort_->at(i).getVector3fMap().cast<double>() + state_point_.pos)
                    .cast<float>();
        }
        ivox_->AddPoints(scan_down_world_->points);
        if_local_map_init_ = false;
        return;
    }

    /// downsample
    Timer::Evaluate(
        [&, this]() {
            scan_sampler_.setInputCloud(scan_undistort_);
            scan_sampler_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    residuals_.resize(cur_pts, 0);
    point_selected_surf_.resize(cur_pts, true);
    plane_coef_.resize(cur_pts, Vec4f::Zero());

    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            double solve_H_time = 0;
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            state_point_ = kf_.get_x();
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() {
        MapIncremental();
    }, "Incremental Mapping");

    LOG(INFO) << "Raw scan: " << scan_undistort_->points.size() << " downsample " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    PublishROSMsg();
    PostUpdate();
}
void LaserMapping::PostUpdate() {
    // save to trajectory
    Pose3 body_pose = Pose3(state_point_.rot, state_point_.pos);
    trajectory_.emplace_back(end_time_, body_pose.Isometry3d());

    // add scan frame to global optimize
    static scan_t scan_id = 1;
    ScanFrame::Ptr scan = std::make_shared<ScanFrame>(scan_id);
    std::stringstream stamp_string;
    stamp_string << std::setw(15) << std::setfill('0') << std::fixed << std::setprecision(8) << measures_.end_time_;
    scan->cloud_fname = output_dir + "/scans/" + stamp_string.str() + ".pcd";
    scan->world_from_body = Pose3(state_point_.rot, state_point_.pos);
    scan->timestamp = measures_.end_time_;
    mapper->AddScan(scan);
    scan_id++;

    // save the pcd
    if (param->image_save_en_ && !measures_.img_.empty()) {
        // construct the image
        image_t im_id = scan->scan_id;
        Image::Ptr im = std::make_shared<Image>();
        im->timestamp_ = measures_.end_time_;
        im->image_id_ = im_id;
        // the name not include the dir path, only the filename
        im->name_ = stamp_string.str() + ".jpg";
        im->cam_from_world_ = (scan->world_from_body * param->extrin_ic_).GetInverse();
        CHECK(sfm_data_.cameras_.size() == 1);
        im->camera_id_ = sfm_data_.cameras_.begin()->first;

        // add to sfm_data
        sfm_data_.images_[im->image_id_] = im;

        // save image
        static auto once = fs::create_directories(output_dir + "/images");
        cv::imwrite(output_dir + "/images/" + im->name_, measures_.img_);
    }
    if (param->pcd_save_en_) {
        static auto once = fs::create_directories(output_dir + "/scans");
        pcl::io::savePCDFileBinary(scan->cloud_fname, *scan_undistort_);
    }
    if (param->pcd_save_en_) {
        *pcl_wait_save_ += *scan_down_world_;
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (param->pcd_save_interval_ > 0 && scan_wait_num >= param->pcd_save_interval_) {
            static auto once = fs::create_directories(output_dir + "/maps");

            // sample
            scan_sampler_.setInputCloud(pcl_wait_save_);
            scan_sampler_.filter(*pcl_wait_save_);

            // load pcd
            std::ostringstream pcd_save_fname_ss;
            pcd_save_fname_ss << output_dir << "/maps/" << std::setw(6) << std::setfill('0') << pcd_idx << ".pcd";
            std::string pcd_save_fname(pcd_save_fname_ss.str());
            if (!pcl_wait_save_->empty())
                pcl::io::savePCDFileBinary(pcd_save_fname, *pcl_wait_save_);
            pcd_idx++;
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}
void LaserMapping::PublishROSMsg() {
    // publish
    if (pub_laser_cloud_world_)
        PublishFrameWorld();
    if (pub_path_)
        PublishPath();
    if (pub_odom_aft_mapped_)
        PublishOdometry();
    if (pub_laser_cloud_effect_world_)
        PublishFrameEffectWorld();
}
bool LaserMapping::SyncPackages() {
    if (points_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    if (param->camera_enable_ && image_buffer_.empty())
        return false;

    // set the measure end timestamp
    if (std::isnan(end_time_)) {
        // for first time
        if (param->camera_enable_) {
            end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        } else
            end_time_ = points_buffer_.front().timestamp + param->scan_interval_;
    } else if (measures_.end_time_ == end_time_) {
        // after the update, incre the end time
        if (param->camera_enable_) {
            end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        } else
            end_time_ = end_time_ + param->scan_interval_;
    } else {
        // the measure is not synced, no need to set the lidar end time
        end_time_ = end_time_;
    }

    if (imu_buffer_.back().timestamp < end_time_) return false;
    if (points_buffer_.back().timestamp < end_time_) return false;

    // push the imu data
    measures_.imu_.clear();
    while (imu_buffer_.front().timestamp < end_time_ && !imu_buffer_.empty()) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    // push the lidar points
    measures_.lidar_->clear();
    while (points_buffer_.front().timestamp < end_time_ && !points_buffer_.empty()) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }

    measures_.end_time_ = end_time_;
    if (measures_.lidar_->empty() || measures_.imu_.empty()) {
        std::cout << "Empty lidar or imu data, skip this measure" << std::endl;
        return false;
    }
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
              << ", off r: " << s.R_il.coeffs().transpose() << ", t: " << s.t_il.transpose();
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        scan_down_world_->at(i).getVector3fMap() =
            (state_point_.rot * scan_down_body_->at(i).getVector3fMap().cast<double>() + state_point_.pos).cast<float>();

        /* decide if need add to map */
        Point &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty()) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / param->map_filter_size_).array().floor() + 0.5) * param->map_filter_size_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * param->map_filter_size_ && fabs(dis_2_center.y()) > 0.5 * param->map_filter_size_ &&
                fabs(dis_2_center.z()) > 0.5 * param->map_filter_size_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = (point_world.getVector3fMap() - center).squaredNorm();
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if ((points_near[readd_i].getVector3fMap() - center).squaredNorm() < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}


static bool esti_plane(Eigen::Matrix<float, 4, 1> &pca_result, const PointVector &point, const float &threshold = 0.1f) {
    if (point.size() < options::MIN_NUM_MATCH_POINTS) {
        return false;
    }

    Eigen::Matrix<float, 3, 1> normvec;

    if (point.size() == options::NUM_MATCH_POINTS) {
        Eigen::Matrix<float, options::NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<float, options::NUM_MATCH_POINTS, 1> b;

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < options::NUM_MATCH_POINTS; j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        normvec = A.colPivHouseholderQr().solve(b);
    } else {
        Eigen::MatrixXd A(point.size(), 3);
        Eigen::VectorXd b(point.size(), 1);

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < point.size(); j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        Eigen::MatrixXd n = A.colPivHouseholderQr().solve(b);
        normvec(0, 0) = n(0, 0);
        normvec(1, 0) = n(1, 0);
        normvec(2, 0) = n(2, 0);
    }

    float n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (const auto &p : point) {
        Eigen::Matrix<float, 4, 1> temp = p.getVector4fMap();
        temp[3] = 1.0;
        if (fabs(pca_result.dot(temp)) > threshold) {
            return false;
        }
    }
    return true;
}
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot).cast<float>();
            auto t_wl = (s.pos).cast<float>();

            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Point &point_body = scan_down_body_->points[i];
                Point &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                Vec3f p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    /** Find the closest surfaces in the map **/
                    points_near.clear();
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] = esti_plane(plane_coef_[i], points_near, param->esti_plane_thr);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        point_selected_surf_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const Mat3f Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Vec3f point_this_be = corr_pts_[i].head<3>();
                Mat3f point_be_crossmat = Hat(point_this_be);
                Vec3f point_this = point_this_be;
                Mat3f point_crossmat = Hat(point_this);

                Vec3f norm_vec = corr_norm_[i].head<3>();
                Vec3f C(Rt * norm_vec);
                Vec3f A(point_crossmat * C);
                ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

}  // namespace faster_lio



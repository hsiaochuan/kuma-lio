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

void CalcBodyCov(Vec3 &pb, const float dist_noise, const float dir_degree, Mat3 &cov)
{
    float d = pb.norm();
    Vec3 w(pb);
    w.normalize();

    // sigma
    float sigma_d = dist_noise * dist_noise;
    float dir_rad = DEG2RAD(dir_degree);
    Eigen::Matrix2d sigma_w = Eigen::Matrix2d::Identity() * dir_rad * dir_rad;

    // construct N
    Vec3 n1(1, 1, -(w(0) + w(1)) / w(2));
    n1.normalize();
    Vec3 n2 = n1.cross(w);
    n2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N.col(0) = n1;
    N.col(1) = n2;

    // cov
    Eigen::Matrix<double, 3, 2> dw = d * Hat(w) * N;
    cov = w * sigma_d * w.transpose() +
        dw * sigma_w * dw.transpose();
}
void LaserMapping::Run() {
    // sync the lidar and imu data, if no data or not synced, return true
    if (!SyncPackages()) {
        return;
    }

    if (!param->imu_enable_)
        throw std::runtime_error("disable the imu is not support");

    /// IMU process, kf prediction, undistortion
    PointCloud::Ptr scan_body(new PointCloud);
    pcl::transformPointCloud(*measures_.lidar_, *scan_body, param->extrin_il_.Mat4d());
    if (!p_imu_->inertial_initialized) {
        p_imu_->InertialInitialize(measures_, *state_point_);
        return;
    }

    Timer::Evaluate([&, this]() {
        p_imu_->Predict(measures_, *state_point_);
        p_imu_->UndistortPoints(*state_point_, scan_body, *scan_undistort_);
    }, "Undistort Pcl");

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return;
    }

    /// the first scan
    if (if_local_map_init_) {
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            scan_down_world_->at(i).getVector3fMap() =
                (state_point_->rot * scan_undistort_->at(i).getVector3fMap().cast<double>() + state_point_->pos)
                    .cast<float>();
        }
        ivox_->AddPoints(scan_down_world_->points);
        if_local_map_init_ = false;
        return;
    }

    /// downsample
    scan_sampler_.setInputCloud(scan_undistort_);
    scan_sampler_.filter(*scan_down_body_);


    if (scan_down_body_->size() < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return;
    }


    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            IESKF::IterativeUpdate(
            std::bind(&LaserMapping::BuildLidarObservation, this, std::placeholders::_1, std::placeholders::_2),
                param->max_iteraions, *state_point_);
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() {
        MapIncremental();
    }, "Incremental Mapping");

    LOG(INFO) << "Raw scan: " << scan_undistort_->points.size() << " downsample " << scan_down_body_->size()
              << " Map grid num: " << ivox_->grids_map_.size() << " effect num : " << eff_num_;

    PublishROSMsg();
    PostUpdate();
}
void LaserMapping::PostUpdate() {
    // save to trajectory
    Pose3 body_pose = Pose3(state_point_->rot, state_point_->pos);
    trajectory_.emplace_back(state_point_->timestamp, body_pose.Isometry3d());

    // add scan frame to global optimize
    static scan_t scan_id = 1;
    ScanFrame::Ptr scan = std::make_shared<ScanFrame>(scan_id);
    std::stringstream stamp_string;
    stamp_string << std::setw(15) << std::setfill('0') << std::fixed << std::setprecision(8) << measures_.end_time_;
    scan->cloud_fname = output_dir + "/scans/" + stamp_string.str() + ".pcd";
    scan->world_from_body = Pose3(state_point_->rot, state_point_->pos);
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
    if (points_buffer_.empty())
        return false;
    if (param->imu_enable_ && imu_buffer_.empty())
        return false;
    if (param->camera_enable_ && image_buffer_.empty())
        return false;

    // set the measure end timestamp
    if (std::isnan(state_point_->timestamp)) {
        // for first time
        if (param->camera_enable_) {
            measures_.end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        } else
            measures_.end_time_ = points_buffer_.front().timestamp + param->scan_interval_;
    } else if (measures_.end_time_ == state_point_->timestamp) {
        // after the update, incre the end time
        if (param->camera_enable_) {
            measures_.end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        } else
            measures_.end_time_ = measures_.end_time_ + param->scan_interval_;
    } else {
        // the measure is not synced, no need to set the lidar end time
        measures_.end_time_ = measures_.end_time_;
    }

    if (param->imu_enable_)
        if (imu_buffer_.back().timestamp < measures_.end_time_)
            return false;
    if (points_buffer_.back().timestamp < measures_.end_time_) return false;

    // push the imu data
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < measures_.end_time_) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    // push the lidar points
    measures_.lidar_->clear();
    while (!points_buffer_.empty() && points_buffer_.front().timestamp < measures_.end_time_) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }

    if (measures_.lidar_->empty() ||
        (param->imu_enable_ && measures_.imu_.empty())) {
        std::cout << "Empty lidar or imu data, skip this measure" << std::endl;
        return false;
    }
    return true;
}

void LaserMapping::PrintState(const StatePoint &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose();
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
            (state_point_->rot * scan_down_body_->at(i).getVector3fMap().cast<double>() + state_point_->pos)
                .cast<float>();

        /* decide if need add to map */
        Point &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty()) {
            const PointVector &points_near = nearest_points_[i];

            // get the vox
            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / param->map_filter_size_).array().floor() + 0.5) * param->map_filter_size_;

            // no near points in the vox, add this point
            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;
            if (fabs(dis_2_center.x()) > 0.5 * param->map_filter_size_ &&
                fabs(dis_2_center.y()) > 0.5 * param->map_filter_size_ &&
                fabs(dis_2_center.z()) > 0.5 * param->map_filter_size_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            // have near point in the vox, if the point is near to center, add the point
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
bool LaserMapping::BuildLidarObservation(const StatePoint &s, LidarObservation &obs) {
    eff_mask_.resize(scan_down_body_->size(), false);
    nearest_points_.resize(scan_down_body_->size());
    std::vector<Vec4f> plane_coeffs(scan_down_body_->size());
    std::vector<float> residuals_(scan_down_body_->size(), 0.0);
    std::vector<size_t> index(scan_down_body_->size());
    pcl::transformPointCloud(*scan_down_body_, *scan_down_world_, s.Isometry().cast<float>());

    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Point &point_body = scan_down_body_->points[i];
                Point &point_world = scan_down_world_->points[i];

                auto &points_near = nearest_points_[i];

                /** Find the closest surfaces in the map **/
                points_near.clear();
                ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                eff_mask_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                if (eff_mask_[i]) {
                    eff_mask_[i] = esti_plane(plane_coeffs[i], points_near, param->esti_plane_thr);
                }


                if (eff_mask_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coeffs[i].dot(temp);

                    bool valid_corr = point_body.getVector3fMap().norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        eff_mask_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        eff_mask_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    eff_num_ = 0;

    std::vector<Vec4f> match_point_;                              // inlier pts
    std::vector<Vec4f> match_plane_coeff_;                             // inlier plane norms
    match_point_.resize(scan_down_body_->size());
    match_plane_coeff_.resize(scan_down_body_->size());
    for (int i = 0; i < scan_down_body_->size(); i++) {
        if (eff_mask_[i]) {
            match_plane_coeff_[eff_num_] = plane_coeffs[i];
            match_point_[eff_num_] = scan_down_body_->points[i].getVector4fMap();
            match_point_[eff_num_][3] = residuals_[i];
            eff_num_++;
        }
    }
    match_point_.resize(eff_num_);
    match_plane_coeff_.resize(eff_num_);

    if (eff_num_ < 1) {
        obs.valid = false;
        LOG(WARNING) << "No Effective Points!";
        return false;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            obs.H = Eigen::MatrixXd::Zero(eff_num_, StatePoint::STATE_DOF);
            obs.r = Eigen::VectorXd::Zero(eff_num_);
            obs.HTRinv = Eigen::MatrixXd::Zero(StatePoint::STATE_DOF, eff_num_);
            index.resize(eff_num_);
            const Mat3f Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Vec3f point_this = match_point_[i].head<3>();

                Vec3f norm_vec = match_plane_coeff_[i].head<3>();
                Vec3f J_rot(Hat(point_this) * Rt * norm_vec);
                obs.H.block<1, 3>(i, StatePoint::POS) << norm_vec[0], norm_vec[1], norm_vec[2];
                obs.H.block<1, 3>(i, StatePoint::ROT) << J_rot[0], J_rot[1], J_rot[2];
                obs.HTRinv.col(i) = obs.H.row(i).transpose() / options::LASER_POINT_COV;
                obs.r(i) = -match_point_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
    obs.valid = true;
    return true;
}

}  // namespace faster_lio



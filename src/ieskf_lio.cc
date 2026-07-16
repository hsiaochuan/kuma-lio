#include <execution>

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>

#include "ieskf_lio.h"
#include "options.h"
#include "utils.h"

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

IeskfLio::IeskfLio() {
    p_imu_ = std::make_shared<ImuProcess>();
    state_point_ = std::make_shared<StatePoint>();
    visual_manager = std::make_shared<VisualManager>();
}

bool IeskfLio::Init(const std::shared_ptr<LaserMappingParam> &param_in) {
    param = param_in;

    if (param->ivox_nearby_type == 0) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (param->ivox_nearby_type == 6) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (param->ivox_nearby_type == 18) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (param->ivox_nearby_type == 26) {
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        param->ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    ivox_ = std::make_shared<IVoxType>(param->ivox_options_);

    scan_sampler_.setLeafSize(param->scan_filter_size, param->scan_filter_size, param->scan_filter_size);

    p_imu_->cov_gyr_ = Vec3(param->gyr_cov, param->gyr_cov, param->gyr_cov);
    p_imu_->cov_acc_ = Vec3(param->acc_cov, param->acc_cov, param->acc_cov);
    p_imu_->cov_bias_gyr_ = Vec3(param->b_gyr_cov, param->b_gyr_cov, param->b_gyr_cov);
    p_imu_->cov_bias_acc_ = Vec3(param->b_acc_cov, param->b_acc_cov, param->b_acc_cov);
    p_imu_->state_point_ = state_point_;

    if (param->camera_enable_) {
        visual_manager->state_point_ = state_point_;
        visual_manager->ivox_ = ivox_;
        visual_manager->param = param;
        visual_manager->Initialize();
    }

    return true;
}

bool IeskfLio::ProcessMeasure(MeasureGroup &measures) {
    /// IMU process, kf prediction, undistortion
    PointCloud::Ptr scan_body(new PointCloud);
    if (param->imu_enable_)
        pcl::transformPointCloud(*measures.lidar_, *scan_body, param->extrin_il_.Mat4d());
    else
        *scan_body = *measures.lidar_;

    if (!p_imu_->inertial_initialized) {
        if (param->imu_enable_)
            p_imu_->InertialInitialize(measures, *state_point_);
        else
            p_imu_->Initialize(measures, *state_point_);

        if (p_imu_->inertial_initialized && param->localization_enable_) {
            if (param->imu_enable_) {
                Vec3 g_map(0, 0, -GRAVITY_NORM);
                Eigen::Quaterniond q_prior = prior_init_rotation_.normalized();
                Vec3 acc_pred = q_prior * p_imu_->mean_acc_.normalized();
                Eigen::Quaterniond q_fix = Eigen::Quaterniond::FromTwoVectors(acc_pred, Vec3(0, 0, 1));

                state_point_->rot = q_fix * q_prior;
                state_point_->gravity = g_map;
            } else {
                state_point_->rot = prior_init_rotation_;
            }
            state_point_->pos = prior_init_position_;
        }
        return false;
    }

    Timer::Evaluate([&, this]() {
        if (param->imu_enable_) {
            p_imu_->Predict(measures);
            p_imu_->UndistortPoints(*state_point_, scan_body, *scan_undistort_);
        }else {
            p_imu_->PredictConstVel(measures);
            p_imu_->UndistortPointsConstVel(scan_body, *scan_undistort_);
        }
    }, "Undistort Pcl");

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return false;
    }

    /// the first scan
    if (if_local_map_init_ && !param->localization_enable_) {
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            scan_down_world_->at(i).getVector3fMap() =
                (state_point_->rot * scan_undistort_->at(i).getVector3fMap().cast<double>() + state_point_->pos)
                    .cast<float>();
        }
        ivox_->AddPoints(scan_down_world_->points);
        if_local_map_init_ = false;
        return false;
    }

    /// downsample
    scan_sampler_.setInputCloud(scan_undistort_);
    scan_sampler_.filter(*scan_down_body_);


    if (scan_down_body_->size() < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return false;
    }


    // ICP and iterated Kalman filter update
    StatePoint state_predict = *state_point_;
    Timer::Evaluate(
        [&, this]() {
            IESKF::IterativeUpdate(
            std::bind(&IeskfLio::BuildLidarObservation, this, std::placeholders::_1, std::placeholders::_2),
                param->max_iteraions, *state_point_);
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() {
        if (param->localization_enable_)
            return;
        MapIncremental();
    }, "Incremental Mapping");

    Timer::Evaluate([&, this]() {
        if (param->camera_enable_ && param->visual_update_ && !measures.img_.empty()) {
            std::vector<PointWithNormal> points_with_normals(scan_down_body_->size());
            for (int i = 0; i < scan_down_world_->size(); i++) {
                PointWithNormal point_with_normal;
                point_with_normal.xyz = scan_down_world_->points[i].getVector3fMap().cast<double>();
                if (eff_mask_[i])
                    point_with_normal.normal = plane_coeffs_[i].head<3>().cast<double>();
                else
                    point_with_normal.normal = Vec3::Zero();
                points_with_normals[i] = point_with_normal;
            }
            visual_manager->state_predict_ = state_predict;
            visual_manager->Run(measures.img_, points_with_normals);
        }
    }, "Visual IEKF solve and Update");

    LOG(INFO) << "Raw scan: " << scan_undistort_->points.size() << " downsample " << scan_down_body_->size()
              << " Map grid num: " << ivox_->grids_map_.size() << " effect num : " << eff_num_;
    return true;
}

void IeskfLio::PrintState(const StatePoint &s) {
    LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose();
}

void IeskfLio::MapIncremental() {
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

bool IeskfLio::BuildLidarObservation(const StatePoint &s, LidarObservation &obs) {
    eff_mask_.resize(scan_down_body_->size(), false);
    nearest_points_.resize(scan_down_body_->size());
    plane_coeffs_.resize(scan_down_body_->size());
    std::fill(eff_mask_.begin(), eff_mask_.end(), false);
    std::fill(nearest_points_.begin(), nearest_points_.end(), PointVector());
    std::fill(plane_coeffs_.begin(), plane_coeffs_.end(), Vec4f::Zero());
    std::vector<float> residuals_(scan_down_body_->size(), 0.0);
    std::vector<size_t> index(scan_down_body_->size());
    pcl::transformPointCloud(*scan_down_body_, *scan_down_world_, s.Isometry().cast<float>());

    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            if (param->localization_enable_) {
                // find the surface in the prior map
                std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                    Point &point_world = scan_down_world_->points[i];
                    PriorMapPoint search_point;
                    search_point.getVector3fMap() = point_world.getVector3fMap();
                    std::vector<int> nearest_indices;
                    std::vector<float> nearest_distances;
                    map_kd_tree_->nearestKSearch(search_point, 1, nearest_indices, nearest_distances);
                    if (nearest_indices.size() == 0) {
                        eff_mask_[i] = false;
                        return;
                    }
                    int nearest_index = nearest_indices[0];
                    Vec3f nearest_point = map_cloud_->at(nearest_index).getVector3fMap();
                    float distance = (nearest_point - point_world.getVector3fMap()).norm();
                    Vec3f normal = map_normals_[nearest_index];
                    if (distance < 0.2 && normal.hasNaN() == false) {
                        float d = -normal.dot(nearest_point);
                        plane_coeffs_[i] = Vec4f(normal[0], normal[1], normal[2], d);
                        residuals_[i] = plane_coeffs_[i].dot(point_world.getVector3fMap().homogeneous());
                        eff_mask_[i] = true;
                    } else {
                        eff_mask_[i] = false;
                    }
                });
                // no need to find point in ivox
                return;
            }
            /** closest surface search and residual computation **/
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Point &point_body = scan_down_body_->points[i];
                Point &point_world = scan_down_world_->points[i];


                /** Find the closest surfaces in the map **/
                auto &points_near = nearest_points_[i];
                points_near.clear();
                ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                eff_mask_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                if (eff_mask_[i]) {
                    eff_mask_[i] = esti_plane(plane_coeffs_[i], points_near, param->esti_plane_thr);
                }


                if (eff_mask_[i]) {
                    float residual = plane_coeffs_[i].dot(point_world.getVector3fMap().homogeneous());
                    bool valid_corr = point_body.getVector3fMap().norm() > 81 * residual * residual;
                    if (valid_corr) {
                        eff_mask_[i] = true;
                        residuals_[i] = residual;
                    } else {
                        eff_mask_[i] = false;
                    }
                }
            });
        },
        "ObsModel (Lidar Match)");

    eff_num_ = 0;

    std::vector<Vec4f> match_point_;                              // inlier pts
    std::vector<Vec4f> match_plane_coeff_;                             // inlier plane norms
    match_point_.resize(scan_down_body_->size());
    match_plane_coeff_.resize(scan_down_body_->size());
    for (int i = 0; i < scan_down_body_->size(); i++) {
        if (eff_mask_[i]) {
            match_plane_coeff_[eff_num_] = plane_coeffs_[i];
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

void IeskfLio::LoadPriorMap(const std::string &prior_map_fname) {
    if (!param->localization_enable_)
        return;

    pcl::PCLPointCloud2 cloud_blob;
    if (pcl::io::loadPCDFile(prior_map_fname, cloud_blob) < 0) {
        LOG(WARNING) << "failed to load prior map: " << prior_map_fname;
        return;
    }
    bool has_normals = false;
    for (const auto &field : cloud_blob.fields) {
        if (field.name == "normal_x") {
            has_normals = true;
            break;
        }
    }
    map_cloud_.reset(new pcl::PointCloud<PriorMapPoint>());

    if (has_normals) {
        LOG(INFO) << "prior map already contains normals, skip normal estimation";
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        pcl::fromPCLPointCloud2(cloud_blob, *cloud_with_normals);
        if (cloud_with_normals->empty()) {
            LOG(WARNING) << "no prior map found";
            return;
        }
        pcl::copyPointCloud(*cloud_with_normals, *map_cloud_);
        LOG(INFO) << "prior map loaded, size: " << map_cloud_->size();
        map_normals_.resize(cloud_with_normals->size());
        for (size_t i = 0; i < cloud_with_normals->size(); ++i) {
            map_normals_[i] = cloud_with_normals->points[i].getNormalVector3fMap();
        }
    } else {
        pcl::fromPCLPointCloud2(cloud_blob, *map_cloud_);
        if (map_cloud_->empty()) {
            LOG(WARNING) << "no prior map found";
            return;
        }
        LOG(INFO) << "prior map loaded, size: " << map_cloud_->size();
        // downsample
        double leaf_size = -0.1;
        if (leaf_size > 0) {
            pcl::VoxelGrid<PriorMapPoint> voxel_grid;
            voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
            voxel_grid.setInputCloud(map_cloud_);
            pcl::PointCloud<PriorMapPoint>::Ptr tmp_cloud(new pcl::PointCloud<PriorMapPoint>());
            voxel_grid.filter(*tmp_cloud);
            map_cloud_ = tmp_cloud;
        }
        LOG(INFO) << "prior map downsampled, size: " << map_cloud_->size();
        // estimate normal
        pcl::NormalEstimationOMP<PriorMapPoint, pcl::Normal> ne;
        ne.setNumberOfThreads(8);
        ne.setInputCloud(map_cloud_);
        pcl::search::KdTree<PriorMapPoint>::Ptr tree(new pcl::search::KdTree<PriorMapPoint>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(12);
        ne.compute(*cloud_normals);
        LOG(INFO) << "prior map normals estimated";
        map_normals_.resize(map_cloud_->size());
        for (size_t i = 0; i < map_cloud_->size(); ++i) {
            map_normals_[i] = cloud_normals->points[i].getNormalVector3fMap();
        }
    }

    // build kd tree
    map_kd_tree_.reset(new pcl::KdTreeFLANN<PriorMapPoint>());
    map_kd_tree_->setInputCloud(map_cloud_);
    LOG(INFO) << "prior map kd tree built";
}

}  // namespace faster_lio

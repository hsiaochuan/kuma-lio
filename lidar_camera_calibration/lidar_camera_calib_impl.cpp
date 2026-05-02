#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include "lidar_camera_calib.hpp"
Calibration::Calibration(const std::string &image_file, const std::string &pcd_file,
                         const std::string &calib_config_file) {
    LoadCalibConfig(calib_config_file);
    LoadCameraConfig(calib_config_file);

    image_ = cv::imread(image_file, cv::IMREAD_UNCHANGED);
    if (!image_.data) {
        throw std::runtime_error("Failed to load image file");
    }
    std::cout << "Sucessfully load image file" << std::endl;

    // check rgb or gray
    if (image_.type() == CV_8UC1) {
        grey_image_ = image_;
    } else if (image_.type() == CV_8UC3) {
        cv::cvtColor(image_, grey_image_, cv::COLOR_BGR2GRAY);
    } else {
        throw std::runtime_error("Unsupported image type");
    }

    // detect image
    cv::Mat edge_image;
    EdgeDetector(rgb_canny_threshold_, rgb_edge_minLen_, grey_image_, edge_image, rgb_egde_cloud_);
        std::string msg = "Sucessfully extract edge from image, edge size:" + std::to_string(rgb_egde_cloud_->size());

    // load point cloud
    if (!pcl::io::loadPCDFile(pcd_file, *lidar_processor_.raw_lidar_cloud_))
        std::cout << "Successfully load the pointcloud" << std::endl;
    else
        throw std::runtime_error("Failed to load pointcloud");

    // extract lidar edge
    std::vector<VoxelGrid> voxel_list;
    std::unordered_map<VOXEL_LOC, Voxel *> voxel_map;
    std::cout << "Start init voxel" << std::endl;
    lidar_processor_.InitVoxel(lidar_processor_.raw_lidar_cloud_, lidar_processor_.voxel_size_, voxel_map);
    std::cout << "Start extract lidar edge" << std::endl;
    lidar_processor_.ExtractLidarEdge(voxel_map, lidar_processor_.ransac_dis_threshold_,
                                      lidar_processor_.plane_size_threshold_, lidar_processor_.plane_line_cloud_);
};

bool Calibration::LoadCameraConfig(const std::string &camera_file) {
    YAML::Node camera_config;
    try {
        camera_config = YAML::LoadFile(camera_file);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load camera config file at: " << camera_file << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        exit(-1);
    }

    float fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_;
    int width_, height_;
    std::vector<double> camera_matrix_;
    std::vector<double> dist_coeffs_;

    width_ = camera_config["camera"]["width"].as<double>();
    height_ = camera_config["camera"]["height"].as<double>();
    camera_matrix_ = camera_config["camera"]["camera_matrix"].as<std::vector<double>>();
    dist_coeffs_ = camera_config["camera"]["dist_coeffs"].as<std::vector<double>>();

    fx_ = camera_matrix_[0];
    fy_ = camera_matrix_[4];
    cx_ = camera_matrix_[2];
    cy_ = camera_matrix_[5];
    k1_ = dist_coeffs_[0];
    k2_ = dist_coeffs_[1];
    k3_ = dist_coeffs_[2];
    p1_ = dist_coeffs_[3];
    p2_ = dist_coeffs_[4];

    camera_ =
        std::make_shared<faster_lio::PinholeRadialCamera>(width_, height_, fx_, fy_, cx_, cy_, k1_, k2_, k3_, p1_, p2_);
    return true;
};

bool Calibration::LoadCalibConfig(const std::string &config_file) {
    cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << config_file << std::endl;
        exit(-1);
    } else {
        ROS_INFO("Sucessfully load calib config file");
    }
    fSettings["ExtrinsicMat"] >> init_extrinsic_;
    init_rotation_matrix_ << init_extrinsic_.at<double>(0, 0), init_extrinsic_.at<double>(0, 1),
        init_extrinsic_.at<double>(0, 2), init_extrinsic_.at<double>(1, 0), init_extrinsic_.at<double>(1, 1),
        init_extrinsic_.at<double>(1, 2), init_extrinsic_.at<double>(2, 0), init_extrinsic_.at<double>(2, 1),
        init_extrinsic_.at<double>(2, 2);
    init_translation_vector_ << init_extrinsic_.at<double>(0, 3), init_extrinsic_.at<double>(1, 3),
        init_extrinsic_.at<double>(2, 3);
    rgb_canny_threshold_ = fSettings["Canny.gray_threshold"];
    rgb_edge_minLen_ = fSettings["Canny.len_threshold"];
    lidar_processor_.voxel_size_ = fSettings["Voxel.size"];
    lidar_processor_.down_sample_size_ = fSettings["Voxel.down_sample_size"];
    lidar_processor_.plane_size_threshold_ = fSettings["Plane.min_points_size"];
    lidar_processor_.plane_max_size_ = fSettings["Plane.max_size"];
    lidar_processor_.ransac_dis_threshold_ = fSettings["Ransac.dis_threshold"];
    lidar_processor_.min_line_dis_threshold_ = fSettings["Edge.min_dis_threshold"];
    lidar_processor_.max_line_dis_threshold_ = fSettings["Edge.max_dis_threshold"];
    lidar_processor_.theta_min_ = fSettings["Plane.normal_theta_min"];
    lidar_processor_.theta_max_ = fSettings["Plane.normal_theta_max"];
    lidar_processor_.theta_min_ = cos(DEG2RAD(lidar_processor_.theta_min_));
    lidar_processor_.theta_max_ = cos(DEG2RAD(lidar_processor_.theta_max_));
    lidar_processor_.direction_theta_min_ = cos(DEG2RAD(30.0));
    lidar_processor_.direction_theta_max_ = cos(DEG2RAD(150.0));
    color_intensity_threshold_ = fSettings["Color.intensity_threshold"];
    return true;
};

// Detect edge by canny, and filter by edge length
void Calibration::EdgeDetector(const int &canny_threshold, const int &edge_threshold, const cv::Mat &src_img,
                               cv::Mat &edge_img, PointCloud::Ptr &edge_cloud) {
    int gaussian_size = 5;
    cv::GaussianBlur(src_img, src_img, cv::Size(gaussian_size, gaussian_size), 0, 0);
    cv::Mat canny_result = cv::Mat::zeros(camera_->h(), camera_->w(), CV_8UC1);
    cv::Canny(src_img, canny_result, canny_threshold, canny_threshold * 3, 3, true);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
    edge_img = cv::Mat::zeros(camera_->h(), camera_->w(), CV_8UC1);

    edge_cloud = PointCloud::Ptr(new PointCloud);
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() > edge_threshold) {
            cv::Mat debug_img = cv::Mat::zeros(camera_->h(), camera_->w(), CV_8UC1);
            for (size_t j = 0; j < contours[i].size(); j++) {
                pcl::PointXYZ p;
                p.x = contours[i][j].x;
                p.y = contours[i][j].y;
                p.z = 0;
                edge_img.at<uchar>(p.y, p.x) = 255;
            }
        }
    }
    for (int x = 0; x < edge_img.cols; x++) {
        for (int y = 0; y < edge_img.rows; y++) {
            if (edge_img.at<uchar>(y, x) == 255) {
                pcl::PointXYZ p;
                p.x = x;
                p.y = y;
                p.z = 0;
                edge_cloud->points.push_back(p);
            }
        }
    }
    edge_cloud->width = edge_cloud->points.size();
    edge_cloud->height = 1;
}

void Calibration::Projection(const Eigen::Matrix3d &rot, const Eigen::Vector3d &tran,
                             const pcl::PointCloud<LiDARPoint>::Ptr &lidar_cloud, cv::Mat &projection_img) {
    std::vector<Eigen::Vector3d> pts_3d;
    std::vector<float> intensity_list;

    for (size_t i = 0; i < lidar_cloud->size(); i++) {
        LiDARPoint point_3d = lidar_cloud->points[i];
        float depth = sqrt(pow(point_3d.x, 2) + pow(point_3d.y, 2) + pow(point_3d.z, 2));
        if (depth > min_depth_ && depth < max_depth_) {
            pts_3d.emplace_back(point_3d.x, point_3d.y, point_3d.z);
            intensity_list.emplace_back(lidar_cloud->points[i].intensity);
        }
    }

    cv::Mat image_project = cv::Mat::zeros(camera_->h(), camera_->w(), CV_16UC1);
    cv::Mat rgb_image_project = cv::Mat::zeros(camera_->h(), camera_->w(), CV_8UC3);
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        Eigen::Vector2d point_2d = camera_->project(rot * pts_3d[i] + tran);
        if (!camera_->valid(point_2d.cast<int>())) {
            continue;
        }

        float intensity = intensity_list[i];
        if (intensity > 100) {
            intensity = 65535;
        } else {
            intensity = intensity / 150.0 * 65535;
        }
        image_project.at<ushort>(point_2d.y(), point_2d.x()) = intensity;
    }
    cv::Mat grey_image_projection;
    cv::cvtColor(rgb_image_project, grey_image_projection, cv::COLOR_BGR2GRAY);
    image_project.convertTo(image_project, CV_8UC1, 1 / 256.0);

    projection_img = image_project.clone();
}


cv::Mat Calibration::GetConnectImg(const int dis_threshold, const PointCloud::Ptr &rgb_edge_cloud,
                                   const PointCloud::Ptr &depth_edge_cloud) {
    cv::Mat connect_img = cv::Mat::zeros(camera_->h(), camera_->w(), CV_8UC3);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
    PointCloud::Ptr search_cloud = PointCloud::Ptr(new PointCloud);
    PointCloud::Ptr tree_cloud = PointCloud::Ptr(new PointCloud);
    kdtree->setInputCloud(rgb_edge_cloud);
    tree_cloud = rgb_edge_cloud;
    for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
        cv::Point2d p2(depth_edge_cloud->points[i].x, depth_edge_cloud->points[i].y);
        if (CheckFov(p2)) {
            pcl::PointXYZ p = depth_edge_cloud->points[i];
            search_cloud->points.push_back(p);
        }
    }

    int line_count = 0;

    int K = 1;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
        pcl::PointXYZ searchPoint = search_cloud->points[i];
        if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            for (int j = 0; j < K; j++) {
                float distance = sqrt(pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
                                      pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
                if (distance < dis_threshold) {
                    cv::Scalar color = cv::Scalar(0, 255, 0);
                    line_count++;
                    if ((line_count % 3) == 0) {
                        cv::line(connect_img, cv::Point(search_cloud->points[i].x, search_cloud->points[i].y),
                                 cv::Point(tree_cloud->points[pointIdxNKNSearch[j]].x,
                                           tree_cloud->points[pointIdxNKNSearch[j]].y),
                                 color, 1);
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
        connect_img.at<cv::Vec3b>(rgb_edge_cloud->points[i].y, rgb_edge_cloud->points[i].x)[0] = 255;
        connect_img.at<cv::Vec3b>(rgb_edge_cloud->points[i].y, rgb_edge_cloud->points[i].x)[1] = 0;
        connect_img.at<cv::Vec3b>(rgb_edge_cloud->points[i].y, rgb_edge_cloud->points[i].x)[2] = 0;
    }
    for (size_t i = 0; i < search_cloud->size(); i++) {
        connect_img.at<cv::Vec3b>(search_cloud->points[i].y, search_cloud->points[i].x)[0] = 0;
        connect_img.at<cv::Vec3b>(search_cloud->points[i].y, search_cloud->points[i].x)[1] = 0;
        connect_img.at<cv::Vec3b>(search_cloud->points[i].y, search_cloud->points[i].x)[2] = 255;
    }
    int expand_size = 2;
    cv::Mat expand_edge_img;
    expand_edge_img = connect_img.clone();
    for (int x = expand_size; x < connect_img.cols - expand_size; x++) {
        for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
            if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
                for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
                    for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
                    }
                }
            } else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
                for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
                    for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
                    }
                }
            }
        }
    }
    return connect_img;
}

bool Calibration::CheckFov(const cv::Point2d &p) {
    if (p.x > 0 && p.x < camera_->w() && p.y > 0 && p.y < camera_->h()) {
        return true;
    }
    return false;
}

void Calibration::BuildVPnp(const Eigen::Matrix3d &rot, const Eigen::Vector3d &tran, const int dis_threshold,
                             cv::Mat& residual_img, const PointCloud::Ptr &cam_edge_cloud_2d,
                            const LiDARCloud::Ptr &lidar_edge_cloud_3d,
                            std::vector<VPnPData> &pnp_list) {
    pnp_list.clear();
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> pixel_corres_lps;
    for (int y = 0; y < camera_->h(); y++) {
        std::vector<std::vector<Eigen::Vector3d>> row_pts_container;
        for (int x = 0; x < camera_->w(); x++) {
            std::vector<Eigen::Vector3d> col_pts_container;
            row_pts_container.push_back(col_pts_container);
        }
        pixel_corres_lps.push_back(row_pts_container);
    }

    std::vector<Eigen::Vector3d> lidar_edge_ps_3d;
    for (size_t i = 0; i < lidar_edge_cloud_3d->size(); i++)
        lidar_edge_ps_3d.push_back(lidar_edge_cloud_3d->points[i].getVector3fMap().cast<double>());

    // project 3d-points into image view
    std::vector<Eigen::Vector2d> lidar_edge_ps_2d(lidar_edge_ps_3d.size());
    PointCloud::Ptr lidar_edge_cloud_2d(new PointCloud);
    for (size_t i = 0; i < lidar_edge_ps_2d.size(); i++) {
        lidar_edge_ps_2d[i] = camera_->project(rot * lidar_edge_ps_3d[i] + tran);
        if (!camera_->valid(lidar_edge_ps_2d[i].cast<int>()))
            continue;
        double x = lidar_edge_ps_2d[i].x();
        double y = lidar_edge_ps_2d[i].y();
        if (pixel_corres_lps[y][x].empty())
            lidar_edge_cloud_2d->points.emplace_back(x,y,0);
        pixel_corres_lps[y][x].push_back(lidar_edge_ps_3d[i]);
    }


    residual_img = GetConnectImg(dis_threshold, cam_edge_cloud_2d, lidar_edge_cloud_2d);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_cam(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(new pcl::search::KdTree<pcl::PointXYZ>());

    kdtree_cam->setInputCloud(cam_edge_cloud_2d);
    kdtree_lidar->setInputCloud(lidar_edge_cloud_2d);

    int K = 5;
    std::vector<int> knn_ids_cam(K);
    std::vector<float> knn_dis_cam(K);
    std::vector<int> knn_ids_lidar(K);
    std::vector<float> knn_dis_lidar(K);

    std::vector<cv::Point2d> lidar_2d_list;
    std::vector<cv::Point2d> cam_2d_list;
    std::vector<Eigen::Vector2d> cam_dir_list;
    std::vector<Eigen::Vector2d> lidar_dir_list;

    for (size_t i = 0; i < lidar_edge_cloud_2d->points.size(); i++) {
        pcl::PointXYZ l_search_p = lidar_edge_cloud_2d->points[i];
        if (kdtree_cam->nearestKSearch(l_search_p, K, knn_ids_cam, knn_dis_cam) > 0 &&
            kdtree_lidar->nearestKSearch(l_search_p, K, knn_ids_lidar, knn_dis_lidar) > 0) {

            // check all the distance
            bool dis_check = true;
            for (int j = 0; j < K; j++) {
                float distance = sqrt(pow(l_search_p.x - cam_edge_cloud_2d->points[knn_ids_cam[j]].x, 2) +
                                      pow(l_search_p.y - cam_edge_cloud_2d->points[knn_ids_cam[j]].y, 2));
                if (distance > dis_threshold) {
                    dis_check = false;
                }
            }
            if (dis_check) {
                cv::Point p_l_2d(lidar_edge_cloud_2d->points[i].x, lidar_edge_cloud_2d->points[i].y);
                cv::Point p_c_2d(cam_edge_cloud_2d->points[knn_ids_cam[0]].x, cam_edge_cloud_2d->points[knn_ids_cam[0]].y);
                Eigen::Vector2d direction_cam(0, 0);
                std::vector<Eigen::Vector2d> points_cam;
                for (size_t k = 0; k < knn_ids_cam.size(); k++) {
                    Eigen::Vector2d p(cam_edge_cloud_2d->points[knn_ids_cam[k]].x,
                                      cam_edge_cloud_2d->points[knn_ids_cam[k]].y);
                    points_cam.push_back(p);
                }
                CalcDirection(points_cam, direction_cam);
                Eigen::Vector2d direction_lidar(0, 0);
                std::vector<Eigen::Vector2d> points_lidar;
                for (size_t k = 0; k < knn_ids_cam.size(); k++) {
                    Eigen::Vector2d p(lidar_edge_cloud_2d->points[knn_ids_lidar[k]].x,
                                      lidar_edge_cloud_2d->points[knn_ids_lidar[k]].y);
                    points_lidar.push_back(p);
                }
                CalcDirection(points_lidar, direction_lidar);
                // direction.normalize();
                if (CheckFov(p_l_2d)) {
                    lidar_2d_list.push_back(p_l_2d);
                    cam_2d_list.push_back(p_c_2d);
                    cam_dir_list.push_back(direction_cam);
                    lidar_dir_list.push_back(direction_lidar);
                }
            }
        }
    } // for lidar 2d points
    for (size_t i = 0; i < lidar_2d_list.size(); i++) {
        int y = lidar_2d_list[i].y;
        int x = lidar_2d_list[i].x;
        int pixel_points_size = pixel_corres_lps[y][x].size();
        if (pixel_points_size > 0) {
            VPnPData pnp;
            pnp.lp = Eigen::Vector3d::Zero();
            pnp.u = cam_2d_list[i].x;
            pnp.v = cam_2d_list[i].y;
            for (size_t j = 0; j < pixel_points_size; j++) {
                pnp.lp += pixel_corres_lps[y][x][j];
            }
            pnp.lp = pnp.lp / pixel_points_size;

            pnp.direction = cam_dir_list[i];
            pnp.direction_lidar = lidar_dir_list[i];
            float theta = pnp.direction.dot(pnp.direction_lidar);
            if (theta > lidar_processor_.direction_theta_min_ || theta < lidar_processor_.direction_theta_max_) {
                pnp_list.push_back(pnp);
            }
        }
    }
}

void Calibration::CalcDirection(const std::vector<Eigen::Vector2d> &points, Eigen::Vector2d &direction) {
    lidar_processor_.CalcDirection(points, direction);
}

cv::Mat Calibration::FusedProjectionImage(const Eigen::Matrix3d &rot, const Eigen::Vector3d &tran) {
    cv::Mat proj_img;
    Projection(rot, tran, lidar_processor_.raw_lidar_cloud_, proj_img);
    cv::Mat proj_color_img = cv::Mat::zeros(camera_->h(), camera_->w(), CV_8UC3);
    for (int x = 0; x < proj_color_img.cols; x++) {
        for (int y = 0; y < proj_color_img.rows; y++) {
            uint8_t r, g, b;
            float norm = proj_img.at<uchar>(y, x) / 256.0;
            MapJet(norm, 0, 1, r, g, b);
            proj_color_img.at<cv::Vec3b>(y, x)[0] = b;
            proj_color_img.at<cv::Vec3b>(y, x)[1] = g;
            proj_color_img.at<cv::Vec3b>(y, x)[2] = r;
        }
    }
    cv::Mat merge_img;
    if (image_.type() == CV_8UC3) {
        merge_img = 0.5 * proj_color_img + 0.8 * image_;
    } else {
        cv::Mat rgb;
        cv::cvtColor(image_, rgb, cv::COLOR_GRAY2BGR);
        merge_img = 0.5 * proj_color_img + 0.8 * rgb;
    }
    return merge_img;
}

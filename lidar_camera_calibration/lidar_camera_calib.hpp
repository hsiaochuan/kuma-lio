#ifndef LIDAR_CAMERA_CALIB_HPP
#define LIDAR_CAMERA_CALIB_HPP

#include <Eigen/Core>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>

#include <string>

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "cameras/cameras.h"
#include "common.h"
#include "lidar_processor.hpp"
#include <boost/optional.hpp>
class Calibration {
   public:

    using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
    using LiDARPoint = pcl::PointXYZI;
    using LiDARCloud = pcl::PointCloud<LiDARPoint>;
    int rgb_edge_minLen_ = 200;
    int rgb_canny_threshold_ = 20;
    int min_depth_ = 2.5;
    int max_depth_ = 50;
    float detect_line_threshold_ = 0.02;
    int color_intensity_threshold_ = 5;
    Calibration(const std::string &image_file, const std::string &pcd_file, const std::string &calib_config_file);

    bool LoadCameraConfig(const std::string &camera_file);
    bool LoadCalibConfig(const std::string &config_file);
    bool CheckFov(const cv::Point2d &p);

    void EdgeDetector(const int &canny_threshold, const int &edge_threshold, const cv::Mat &src_img, cv::Mat &edge_img,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud);
    void Projection(const Eigen::Matrix3d &rot, const Eigen::Vector3d &tran,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud, cv::Mat &projection_img);

    void BuildVPnp(const Eigen::Matrix3d &rot, const Eigen::Vector3d &tran, const int dis_threshold,
                   cv::Mat& residual_img, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
                   const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_edge_cloud_3d, std::vector<VPnPData> &pnp_list);

    cv::Mat GetConnectImg(const int dis_threshold, const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud,
                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud);
    cv::Mat FusedProjectionImage(const Eigen::Matrix3d &rot, const Eigen::Vector3d &tran);

    void CalcDirection(const std::vector<Eigen::Vector2d> &points, Eigen::Vector2d &direction);

    std::shared_ptr<faster_lio::CameraBase> camera_;
    cv::Mat init_extrinsic_;


    cv::Mat rgb_image_;
    cv::Mat image_;
    cv::Mat grey_image_;

    LidarProcessor lidar_processor_;

    Eigen::Matrix3d init_rotation_matrix_;
    Eigen::Vector3d init_translation_vector_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_cloud_;
};

#endif

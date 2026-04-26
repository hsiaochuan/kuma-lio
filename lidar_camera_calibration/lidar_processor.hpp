#ifndef LIDAR_CAMERA_LIDAR_PROCESSOR_HPP
#define LIDAR_CAMERA_LIDAR_PROCESSOR_HPP

#include <pcl/ModelCoefficients.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

class LidarProcessor {
   public:
    using LiDARCloud = pcl::PointCloud<pcl::PointXYZI>;
    LidarProcessor();

    bool LoadPointCloud(const std::string &pcd_file);
    void InitVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, const float voxel_size,
                   std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map);
    void ExtractLidarEdge(const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map, const float ransac_dis_thre,
                          const int plane_size_threshold, pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d);
    void CalcLine(const std::vector<Plane> &plane_list, const double voxel_size, const Eigen::Vector3d origin,
                  std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list);
    void CalcDirection(const std::vector<Eigen::Vector2d> &points, Eigen::Vector2d &direction);

    float voxel_size_ = 1.0f;
    float down_sample_size_ = 0.02f;
    float ransac_dis_threshold_ = 0.02f;
    float plane_size_threshold_ = 60.0f;
    int plane_max_size_ = 5;
    float theta_min_ = 0.0f;
    float theta_max_ = 0.0f;
    float direction_theta_min_ = 0.0f;
    float direction_theta_max_ = 0.0f;
    float min_line_dis_threshold_ = 0.03f;
    float max_line_dis_threshold_ = 0.06f;

    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
};

#endif

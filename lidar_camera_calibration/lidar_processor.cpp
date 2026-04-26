#include "lidar_processor.hpp"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <ros/ros.h>
LidarProcessor::LidarProcessor()
    : raw_lidar_cloud_(new LiDARCloud), plane_line_cloud_(new LiDARCloud) {}

void LidarProcessor::InitVoxel(const LiDARCloud::Ptr &input_cloud, const float voxel_size,
                               std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map) {
    srand((unsigned)time(NULL));
    pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
    for (size_t i = 0; i < input_cloud->size(); i++) {
        const pcl::PointXYZI &p_c = input_cloud->points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_c.data[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end()) {
            voxel_map[position]->cloud->push_back(p_c);
            pcl::PointXYZRGB p_rgb;
            p_rgb.x = p_c.x;
            p_rgb.y = p_c.y;
            p_rgb.z = p_c.z;
            p_rgb.r = voxel_map[position]->voxel_color(0);
            p_rgb.g = voxel_map[position]->voxel_color(1);
            p_rgb.b = voxel_map[position]->voxel_color(2);
            test_cloud.push_back(p_rgb);
        } else {
            Voxel *voxel = new Voxel(voxel_size);
            voxel_map[position] = voxel;
            voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
            voxel_map[position]->voxel_origin[1] = position.y * voxel_size;
            voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
            voxel_map[position]->cloud->push_back(p_c);
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;
            voxel_map[position]->voxel_color << r, g, b;
        }
    }
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        if (iter->second->cloud->size() > 20) {
            DownSamplingVoxel(*(iter->second->cloud), down_sample_size_);
        }
    }
}

void LidarProcessor::ExtractLidarEdge(const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map,
                                      const float ransac_dis_thre, const int plane_size_threshold,
                                      LiDARCloud::Ptr &lidar_line_cloud_3d) {
    lidar_line_cloud_3d = LiDARCloud::Ptr(new LiDARCloud);
    plane_line_cloud_ = lidar_line_cloud_3d;
    plane_line_cloud_->clear();

    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        if (iter->second->cloud->size() < 50)
            continue;
        std::vector<Plane> plane_list;
        LiDARCloud::Ptr cloud_filter(new LiDARCloud);
        pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(ransac_dis_thre);
        pcl::PointCloud<pcl::PointXYZRGB> color_planner_cloud;

        while (cloud_filter->points.size() > 10) {
            LiDARCloud planner_cloud;
            pcl::ExtractIndices<pcl::PointXYZI> extract;
            seg.setInputCloud(cloud_filter);
            seg.setMaxIterations(500);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.size() == 0) {
                ROS_INFO_STREAM("Could not estimate a planner model for the given dataset");
                break;
            }
            extract.setIndices(inliers);
            extract.setInputCloud(cloud_filter);
            extract.filter(planner_cloud);

            // contruct a plane
            if (planner_cloud.size() > plane_size_threshold) {
                pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
                std::vector<unsigned int> colors;
                colors.push_back(static_cast<unsigned int>(rand() % 256));
                colors.push_back(static_cast<unsigned int>(rand() % 256));
                colors.push_back(static_cast<unsigned int>(rand() % 256));
                pcl::PointXYZ p_center(0, 0, 0);
                for (size_t i = 0; i < planner_cloud.points.size(); i++) {
                    pcl::PointXYZRGB p;
                    p.x = planner_cloud.points[i].x;
                    p.y = planner_cloud.points[i].y;
                    p.z = planner_cloud.points[i].z;
                    p_center.x += p.x;
                    p_center.y += p.y;
                    p_center.z += p.z;
                    p.r = colors[0];
                    p.g = colors[1];
                    p.b = colors[2];
                    color_cloud.push_back(p);
                    color_planner_cloud.push_back(p);
                }
                p_center.x = p_center.x / planner_cloud.size();
                p_center.y = p_center.y / planner_cloud.size();
                p_center.z = p_center.z / planner_cloud.size();
                Plane single_plane;
                single_plane.cloud = planner_cloud;
                single_plane.p_center = p_center;
                single_plane.normal << coefficients->values[0], coefficients->values[1], coefficients->values[2];
                plane_list.push_back(single_plane);
            }
            extract.setNegative(true);
            LiDARCloud cloud_f;
            extract.filter(cloud_f);
            *cloud_filter = cloud_f;
        }

        std::vector<LiDARCloud> line_cloud_list;
        CalcLine(plane_list, voxel_size_, iter->second->voxel_origin, line_cloud_list);
        if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 8) {
            for (size_t cloud_index = 0; cloud_index < line_cloud_list.size(); cloud_index++) {
                for (size_t i = 0; i < line_cloud_list[cloud_index].size(); i++) {
                    pcl::PointXYZI p = line_cloud_list[cloud_index].points[i];
                    plane_line_cloud_->points.push_back(p);
                }
            }
        }

    }  // for voxel
    lidar_line_cloud_3d = plane_line_cloud_;
}

void LidarProcessor::CalcLine(const std::vector<Plane> &plane_list, const double voxel_size,
                              const Eigen::Vector3d origin,
                              std::vector<LiDARCloud> &line_cloud_list) {
    if (plane_list.size() >= 2 && plane_list.size() <= static_cast<size_t>(plane_max_size_)) {
        LiDARCloud temp_line_cloud;
        for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1; plane_index1++) {
            for (size_t plane_index2 = plane_index1 + 1; plane_index2 < plane_list.size(); plane_index2++) {
                float a1 = plane_list[plane_index1].normal[0];
                float b1 = plane_list[plane_index1].normal[1];
                float c1 = plane_list[plane_index1].normal[2];
                float x1 = plane_list[plane_index1].p_center.x;
                float y1 = plane_list[plane_index1].p_center.y;
                float z1 = plane_list[plane_index1].p_center.z;
                float a2 = plane_list[plane_index2].normal[0];
                float b2 = plane_list[plane_index2].normal[1];
                float c2 = plane_list[plane_index2].normal[2];
                float x2 = plane_list[plane_index2].p_center.x;
                float y2 = plane_list[plane_index2].p_center.y;
                float z2 = plane_list[plane_index2].p_center.z;
                float theta = a1 * a2 + b1 * b2 + c1 * c2;
                float point_dis_threshold = 0.00;
                if (theta > theta_max_ && theta < theta_min_) {
                    if (plane_list[plane_index1].cloud.size() > 0 && plane_list[plane_index2].cloud.size() > 0) {
                        float matrix[4][5];
                        matrix[1][1] = a1;
                        matrix[1][2] = b1;
                        matrix[1][3] = c1;
                        matrix[1][4] = a1 * x1 + b1 * y1 + c1 * z1;
                        matrix[2][1] = a2;
                        matrix[2][2] = b2;
                        matrix[2][3] = c2;
                        matrix[2][4] = a2 * x2 + b2 * y2 + c2 * z2;
                        std::vector<Eigen::Vector3d> points;
                        Eigen::Vector3d point;
                        matrix[3][1] = 1;
                        matrix[3][2] = 0;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[0];
                        Calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold &&
                            point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold &&
                            point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold &&
                            point[2] <= origin[2] + voxel_size + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 1;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[1];
                        Calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold &&
                            point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold &&
                            point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold &&
                            point[2] <= origin[2] + voxel_size + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 0;
                        matrix[3][3] = 1;
                        matrix[3][4] = origin[2];
                        Calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold &&
                            point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold &&
                            point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold &&
                            point[2] <= origin[2] + voxel_size + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 1;
                        matrix[3][2] = 0;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[0] + voxel_size;
                        Calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold &&
                            point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold &&
                            point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold &&
                            point[2] <= origin[2] + voxel_size + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 1;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[1] + voxel_size;
                        Calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold &&
                            point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold &&
                            point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold &&
                            point[2] <= origin[2] + voxel_size + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 0;
                        matrix[3][3] = 1;
                        matrix[3][4] = origin[2] + voxel_size;
                        Calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold &&
                            point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold &&
                            point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold &&
                            point[2] <= origin[2] + voxel_size + point_dis_threshold) {
                            points.push_back(point);
                        }
                        if (points.size() == 2) {
                            LiDARCloud line_cloud;
                            pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);
                            pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);
                            float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                            int K = 1;
                            std::vector<int> pointIdxNKNSearch1(K);
                            std::vector<float> pointNKNSquaredDistance1(K);
                            std::vector<int> pointIdxNKNSearch2(K);
                            std::vector<float> pointNKNSquaredDistance2(K);
                            pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(new pcl::search::KdTree<pcl::PointXYZI>());
                            pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZI>());
                            kdtree1->setInputCloud(plane_list[plane_index1].cloud.makeShared());
                            kdtree2->setInputCloud(plane_list[plane_index2].cloud.makeShared());
                            for (float inc = 0; inc <= length; inc += 0.005) {
                                pcl::PointXYZI p;
                                p.x = p1.x + (p2.x - p1.x) * inc / length;
                                p.y = p1.y + (p2.y - p1.y) * inc / length;
                                p.z = p1.z + (p2.z - p1.z) * inc / length;
                                p.intensity = 100;
                                if ((kdtree1->nearestKSearch(p, K, pointIdxNKNSearch1, pointNKNSquaredDistance1) > 0) &&
                                    (kdtree2->nearestKSearch(p, K, pointIdxNKNSearch2, pointNKNSquaredDistance2) > 0)) {
                                    float dis1 =
                                        pow(p.x - plane_list[plane_index1].cloud.points[pointIdxNKNSearch1[0]].x, 2) +
                                        pow(p.y - plane_list[plane_index1].cloud.points[pointIdxNKNSearch1[0]].y, 2) +
                                        pow(p.z - plane_list[plane_index1].cloud.points[pointIdxNKNSearch1[0]].z, 2);
                                    float dis2 =
                                        pow(p.x - plane_list[plane_index2].cloud.points[pointIdxNKNSearch2[0]].x, 2) +
                                        pow(p.y - plane_list[plane_index2].cloud.points[pointIdxNKNSearch2[0]].y, 2) +
                                        pow(p.z - plane_list[plane_index2].cloud.points[pointIdxNKNSearch2[0]].z, 2);
                                    if ((dis1 < min_line_dis_threshold_ * min_line_dis_threshold_ &&
                                         dis2 < max_line_dis_threshold_ * max_line_dis_threshold_) ||
                                        ((dis1 < max_line_dis_threshold_ * max_line_dis_threshold_ &&
                                          dis2 < min_line_dis_threshold_ * min_line_dis_threshold_))) {
                                        line_cloud.push_back(p);
                                    }
                                }
                            }
                            if (line_cloud.size() > 10) {
                                line_cloud_list.push_back(line_cloud);
                            }
                        }
                    }
                }
            }
        }
    }
}

void LidarProcessor::CalcDirection(const std::vector<Eigen::Vector2d> &points, Eigen::Vector2d &direction) {
    Eigen::Vector2d mean_point(0, 0);
    for (size_t i = 0; i < points.size(); i++) {
        mean_point(0) += points[i](0);
        mean_point(1) += points[i](1);
    }
    mean_point(0) = mean_point(0) / points.size();
    mean_point(1) = mean_point(1) / points.size();
    Eigen::Matrix2d S;
    S << 0, 0, 0, 0;
    for (size_t i = 0; i < points.size(); i++) {
        Eigen::Matrix2d s = (points[i] - mean_point) * (points[i] - mean_point).transpose();
        S += s;
    }
    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
    Eigen::MatrixXcd evecs = es.eigenvectors();
    Eigen::MatrixXcd evals = es.eigenvalues();
    Eigen::MatrixXd evalsReal;
    evalsReal = evals.real();
    Eigen::MatrixXf::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
}

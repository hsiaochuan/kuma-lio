#include <pcl/search/kdtree.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <unordered_set>
#include "common_lib.h"
#include "sfm_data.h"
#include "stamp_pose.h"

#define QUICKHULL_IMPLEMENTATION
#include <pcl/kdtree/kdtree_flann.h>

#include "qhull.h"
namespace po = boost::program_options;
using namespace faster_lio;
struct RenderOptions {
    int k_nb_for_point = 1;
};

int main(int argc, char **argv) {
    std::string lidar_points_fname;
    std::string color_points_fname;
    std::string images_dir;
    std::string colmap_result;
    po::options_description desc("points color options");
    // clang-format off
    desc.add_options()
        ("help,h", "Show help message")
        ("lidar_points_fname,l", po::value<std::string>(&lidar_points_fname)->required(), "LiDAR points map")
        ("color_points_fname,c", po::value<std::string>(&color_points_fname)->required(), "Output colored points map")
        ("images_dir,i", po::value<std::string>(&images_dir)->required(), "Directory containing images")
        ("colmap_result,f", po::value<std::string>(&colmap_result)->required(), "Colmap result directory")
    ;
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    RenderOptions options;

    // load images
    sfm_data recon;
    recon.LoadFromCOLMAPResult(colmap_result);

    // load
    pcl::PointCloud<pcl::PointXYZI>::Ptr kf_positions(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto &[image_id, image] : recon.images_) {
        pcl::PointXYZI point;
        point.x = image->cam_from_world_.Trans().x();
        point.y = image->cam_from_world_.Trans().y();
        point.z = image->cam_from_world_.Trans().z();
        point.intensity = image_id;
        kf_positions->push_back(point);
    }
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(kf_positions);

    // load lidar points
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_points(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(lidar_points_fname, *lidar_points);

    // for every point, search the nb images
    std::unordered_map<size_t, image_t> point_to_images;
    for (size_t i = 0; i < lidar_points->points.size(); ++i) {
        pcl::PointXYZI &lidar_point = lidar_points->points[i];
        std::vector<int> knn_ids;
        std::vector<float> knn_dists;
        kdtree.nearestKSearch(lidar_point, options.k_nb_for_point, knn_ids, knn_dists);
        CHECK(knn_ids.size() == options.k_nb_for_point);
        pcl::PointXYZI knn_point = kf_positions->points[knn_ids[0]];
        point_to_images[i] = static_cast<image_t>(knn_point.intensity);
    }

    // for every image, get the corres point
    std::unordered_map<image_t, std::vector<size_t>> image_to_points;
    for (const auto &[point_id, image_id] : point_to_images) {
        image_to_points[image_id].push_back(point_id);
    }

    // render the points
    using ColorPoint = pcl::PointXYZRGB;
    pcl::PointCloud<ColorPoint>::Ptr color_points(new pcl::PointCloud<ColorPoint>);
    color_points->reserve(lidar_points->points.size());
    for (size_t i = 0; i < lidar_points->points.size(); ++i) {
        ColorPoint point;
        point.x = lidar_points->points[i].x;
        point.y = lidar_points->points[i].y;
        point.z = lidar_points->points[i].z;
        point.r = 0;
        point.g = 0;
        point.b = 0;
        color_points->push_back(point);
    }
    for (const auto &[image_id, point_ids] : image_to_points) {
        Image::Ptr image = recon.GetImage(image_id);
        image->image_data_ = cv::imread(images_dir + "/" + image->Name());

        image->image_data_.convertTo(image->image_data_, CV_32FC3);
        CamModel::Ptr camera = recon.GetCamera(image->CameraId());
        for (size_t point_id : point_ids) {
            ColorPoint &color_point = color_points->points[point_id];
            Eigen::Vector3d point_world = color_point.getVector3fMap().cast<double>();
            Eigen::Vector3d point_cam = image->CameraFromWorld() * point_world;
            Eigen::Vector2d uv = camera->project(point_cam);
            if (camera->valid(uv.cast<int>())) {
                int u = uv(0);
                int v = uv(1);
                cv::Vec3b color = image->image_data_.at<cv::Vec3b>(v, u);
                color_point.r = color[0];
                color_point.g = color[1];
                color_point.b = color[2];
            }
        }
        // for reduce the memory usage, we only render the points for one image at a time, and release the image data
        // after rendering
        recon.GetImage(image_id)->image_data_.release();
    }
    pcl::io::savePCDFileASCII(color_points_fname, *color_points);
    return 0;
}
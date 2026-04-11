#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "reconstruction.h"
#include "stamp_pose.h"

DEFINE_string(slam_out_dir, "","");
DEFINE_string(config_fname, "", "");
using namespace faster_lio;
namespace fs = boost::filesystem;
int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);
    std::string output_dir = FLAGS_slam_out_dir + "/color";
    fs::create_directories(output_dir);
    Reconstruction reconstruction;
    PointCloud::Ptr points(new PointCloud);

    // load pcd
    pcl::io::loadPCDFile(FLAGS_slam_out_dir + "/final_map.pcd", *points);
    if (points->size() <= 0) {
        LOG(ERROR) << "Empty points file";
        return EXIT_FAILURE;
    }
    LOG(INFO) << "Points size: " << points->size();
    // camera
    std::vector<double> param;
    auto yaml = YAML::LoadFile(FLAGS_config_fname);
    std::shared_ptr<CameraBase> camera = std::make_shared<PinholeRadialCamera>();
    auto pinhole_param = yaml["cam"]["pinhole_param"].as<std::vector<double>>();
    auto distortion_param = yaml["cam"]["distortion_param"].as<std::vector<double>>();
    param.insert(param.end(), pinhole_param.begin(), pinhole_param.end());
    param.insert(param.end(), distortion_param.begin(), distortion_param.end());
    camera->w_ = yaml["cam"]["resolution"][0].as<unsigned int>();
    camera->h_ = yaml["cam"]["resolution"][1].as<unsigned int>();
    camera->updateFromParams(param);
    LOG(INFO) << "Load camera";
    // load poses
    Trajectory stamped_poses = TrajectoryGenerator::load_from_tumtxt(FLAGS_slam_out_dir + "/cam_traj_log.txt");
    TrajectoryInterpolator interpolator(stamped_poses);
    LOG(INFO) << "Trajectory size: " << stamped_poses.size();
    // load images dir
    reconstruction.LoadFromImages(FLAGS_slam_out_dir + "/images");
    LOG(INFO) << "Load " << reconstruction.images_.size() << " images";
    // proj the points to image
    for (int i = 0; i < reconstruction.images_.size(); i++) {
        cv::Mat output_image = cv::Mat::zeros(camera->h(), camera->w(), CV_64F);
        Image::Ptr image = reconstruction.images_[i];

        Pose3 world_from_cam = interpolator.query(image->TryReadTimeFromName());
        image->camera_ = camera;

        PointCloud::Ptr points_cam(new PointCloud);
        pcl::transformPointCloud(*points, *points_cam, world_from_cam.GetInverse().GetMat4d());

        for (int j = 0; j < points_cam->size(); ++j) {
            Eigen::Vector3d point_cam = points_cam->points[j].getVector3fMap().cast<double>();
            if (point_cam.z() < 0)
                continue;
            Eigen::Vector2d point_img = image->camera_->project(point_cam);
            for (int u = (int)point_img.x() - 1; u < (int)point_img.x() + 2; u++) {
                for (int v = (int)point_img.y() - 1; v < (int)point_img.y() + 2; v++) {
                    Eigen::Vector2d uv(u, v);
                    if (!image->Camera()->valid(uv)) continue;

                    output_image.at<double>(v, u) = points_cam->points[j].intensity;
                }
            }
        }

        std::string image_fname = (fs::path(output_dir) / fs::path(image->name_).filename()).string();

        // cv::normalize(output_image, output_image, 0, 1, cv::NORM_MINMAX);
        // cv::Mat output_image_8u;
        // output_image.convertTo(output_image_8u, CV_8U);
        // cv::applyColorMap(output_image_8u,output_image_8u,cv::COLORMAP_JET);
        cv::imwrite(image_fname, output_image);
        LOG(INFO) << "Writing result image to " << image_fname;
    }
}
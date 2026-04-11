#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <limits>
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
    std::shared_ptr<CameraBase> camera = std::make_shared<PinholeRadialCamera>();
    try {
        std::vector<double> param;
        auto yaml = YAML::LoadFile(FLAGS_config_fname);
        auto pinhole_param = yaml["cam"]["pinhole_param"].as<std::vector<double>>();
        auto distortion_param = yaml["cam"]["distortion_param"].as<std::vector<double>>();
        param.insert(param.end(), pinhole_param.begin(), pinhole_param.end());
        param.insert(param.end(), distortion_param.begin(), distortion_param.end());
        camera->w_ = yaml["cam"]["resolution"][0].as<unsigned int>();
        camera->h_ = yaml["cam"]["resolution"][1].as<unsigned int>();
        camera->updateFromParams(param);
        LOG(INFO) << "Load camera";
    } catch (...) {
        LOG(ERROR) << "Exception occured while creating camera";
        return EXIT_FAILURE;
    }

    // load poses
    Trajectory stamped_poses = TrajectoryGenerator::load_from_tumtxt(FLAGS_slam_out_dir + "/cam_traj_log.txt");
    TrajectoryInterpolator interpolator(stamped_poses);
    LOG(INFO) << "Trajectory size: " << stamped_poses.size();

    // load images dir
    reconstruction.LoadFromImages(FLAGS_slam_out_dir + "/images");
    LOG(INFO) << "Load " << reconstruction.images_.size() << " images";

    // proj the points to image
    for (int i = 0; i < reconstruction.images_.size(); i++) {
        cv::Mat intensity_map = cv::Mat::zeros(camera->h(), camera->w(), CV_32F);
        cv::Mat valid_mask = cv::Mat::zeros(camera->h(), camera->w(), CV_8U);
        Image::Ptr image = reconstruction.images_[i];

        cv::Mat raw_image = cv::imread(image->name_, cv::IMREAD_COLOR);
        if (raw_image.empty()) {
            LOG(WARNING) << "Skip unreadable image: " << image->name_;
            continue;
        }

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

                    intensity_map.at<float>(v, u) = points_cam->points[j].intensity;
                    valid_mask.at<uint8_t>(v, u) = 255;
                }
            }
        }

        cv::Mat output_image = raw_image.clone();
        if (cv::countNonZero(valid_mask) > 0) {
            double min_intensity = std::numeric_limits<double>::max();
            double max_intensity = std::numeric_limits<double>::lowest();
            cv::minMaxLoc(intensity_map, &min_intensity, &max_intensity, nullptr, nullptr, valid_mask);

            cv::Mat intensity_u8 = cv::Mat::zeros(camera->h(), camera->w(), CV_8U);
            if (max_intensity > min_intensity) {
                for (int r = 0; r < intensity_map.rows; ++r) {
                    for (int c = 0; c < intensity_map.cols; ++c) {
                        if (!valid_mask.at<uint8_t>(r, c)) continue;
                        float value = intensity_map.at<float>(r, c);
                        float norm = static_cast<float>((value - min_intensity) / (max_intensity - min_intensity));
                        intensity_u8.at<uint8_t>(r, c) = static_cast<uint8_t>(std::round(255.0f * norm));
                    }
                }
            } else {
                intensity_u8.setTo(128, valid_mask);
            }

            cv::Mat pseudo_color;
            cv::applyColorMap(intensity_u8, pseudo_color, cv::COLORMAP_JET);

            cv::Mat blended;
            cv::addWeighted(raw_image, 0.55, pseudo_color, 0.45, 0.0, blended);
            blended.copyTo(output_image, valid_mask);
        }

        std::string image_fname = (fs::path(output_dir) / fs::path(image->name_).filename()).string();

        cv::imwrite(image_fname, output_image);
        LOG(INFO) << "Writing result image to " << image_fname;
    }
}
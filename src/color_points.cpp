#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "stamp_pose.h"
#include "reconstruction.h"
#include <yaml-cpp/yaml.h>
DEFINE_string(config_fname, "", "config fname contain the pose");
DEFINE_string(points_fname, "", "points fname");
DEFINE_string(images_dir, "", "images dir");
DEFINE_string(output_dir, "", "otuput dir");
DEFINE_string(stamp_poses_fname, "", "stamped poses fname");
using namespace faster_lio;
namespace fs = boost::filesystem;
int main() {
    fs::create_directories(FLAGS_output_dir);
    Reconstruction reconstruction;
    PointCloud::Ptr points(new PointCloud);

    // load pcd
    pcl::io::loadPCDFile(FLAGS_points_fname, *points);
    if (points->size() <= 0) {
        LOG(ERROR) << "Empty points file";
        return EXIT_FAILURE;
    }

    // extrin
    Pose3 extrin_il;
    auto yaml = YAML::LoadFile(FLAGS_config_fname);
    std::vector<double> extrin_R_il_param;
    std::vector<double> extrin_t_il_param;
    extrin_R_il_param = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();
    extrin_t_il_param = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
    extrin_il.Quat() = common::MatFromArray<double>(extrin_R_il_param);
    extrin_il.Trans() = common::VecFromArray<double>(extrin_t_il_param);

    // load poses
    Trajectory stamped_poses = TrajectoryGenerator::load_from_tumtxt(FLAGS_stamp_poses_fname);
    TrajectoryInterpolator interpolator(stamped_poses);

    // load images dir
    reconstruction.LoadFromImages(FLAGS_images_dir);

    // proj the points to image
    for (int i = 0; i < reconstruction.images_.size(); i++) {
        // output color image
        cv::Mat output_image;

        Image::Ptr image = reconstruction.images_[i];

        auto body_pose = interpolator.query(image->TryReadTimeFromName());
        image->cam_from_world_ = Pose3(body_pose * extrin_il.GetIsometry3d());
        Pose3 cam_from_world = image->Pose();
        PointCloud::Ptr points_cam(new PointCloud);
        std::string image_name = fs::path(image->name_).stem().string();
        pcl::transformPointCloud(*points, *points_cam, cam_from_world.GetMat4d());
        for (int j = 0; j < points_cam->size(); ++j) {
            Eigen::Vector3d point_cam = points_cam->points[j].getVector3fMap().cast<double>();
            Eigen::Vector2d point_img = image->Camera()->project(point_cam);
            if (!image->Camera()->valid(point_img)) continue;
            int u = (int)point_img.x();
            int v = (int)point_img.y();

            output_image.at<double>(v, u) = points_cam->points[j].intensity;
        }

        std::string image_fname = (fs::path(FLAGS_output_dir) / image->name_).string();
        cv::imwrite(image_fname, output_image);
    }
}
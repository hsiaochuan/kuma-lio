#include <tf/transform_broadcaster.h>

#include <array>
#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <sstream>
#include <cv_bridge/cv_bridge.h>
#include "global_optimizor.h"
#include "laser_mapping.h"

namespace fs = boost::filesystem;

namespace faster_lio {

void LaserMapping::PublishOdometry() {
    nav_msgs::Odometry body_odometry;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = state_point_->pos(0);
    pose_stamped.pose.position.y = state_point_->pos(1);
    pose_stamped.pose.position.z = state_point_->pos(2);
    pose_stamped.pose.orientation.x = state_point_->rot.coeffs()[0];
    pose_stamped.pose.orientation.y = state_point_->rot.coeffs()[1];
    pose_stamped.pose.orientation.z = state_point_->rot.coeffs()[2];
    pose_stamped.pose.orientation.w = state_point_->rot.coeffs()[3];
    pose_stamped.header.stamp = ros::Time().fromSec(state_point_->timestamp);
    pose_stamped.header.frame_id = "world";

    body_odometry.header.frame_id = "world";
    body_odometry.child_frame_id = "body";
    body_odometry.header.stamp = ros::Time().fromSec(state_point_->timestamp);
    body_odometry.pose.pose = pose_stamped.pose;
    auto P = state_point_->cov;
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        body_odometry.pose.covariance[i * 6 + 0] = P(k, 3);
        body_odometry.pose.covariance[i * 6 + 1] = P(k, 4);
        body_odometry.pose.covariance[i * 6 + 2] = P(k, 5);
        body_odometry.pose.covariance[i * 6 + 3] = P(k, 0);
        body_odometry.pose.covariance[i * 6 + 4] = P(k, 1);
        body_odometry.pose.covariance[i * 6 + 5] = P(k, 2);
    }
    // publish
    static nav_msgs::Path path;
    static int call_once = [&]() {
        path.header.frame_id = "world";
        path.header.stamp = ros::Time().fromSec(state_point_->timestamp);
        return 0;
    }();
    if (pub_odom_aft_mapped_)
        pub_odom_aft_mapped_.publish(body_odometry);
    path.poses.push_back(pose_stamped);
    if (pub_path_)
        pub_path_.publish(path);
    if (bag_.isOpen())
        bag_.write("/Odometry", ros::Time().fromSec(state_point_->timestamp), pose_stamped);
    if (bag_.isOpen())
        bag_.write("/path", ros::Time().fromSec(state_point_->timestamp), path);

    // transform broadcast
    if (ros::isInitialized()) {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        tf::Vector3 t;
        t.setX(state_point_->pos.x());
        t.setY(state_point_->pos.y());
        t.setZ(state_point_->pos.z());
        q.setX(state_point_->rot.x());
        q.setY(state_point_->rot.y());
        q.setZ(state_point_->rot.z());
        q.setW(state_point_->rot.w());
        transform.setOrigin(t);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, body_odometry.header.stamp, "world", "body"));
    }
}

void LaserMapping::PublishFrameWorld() {
    sensor_msgs::PointCloud2 scan_msg;
    pcl::toROSMsg(*color_scan_world_, scan_msg);
    scan_msg.header.stamp = ros::Time().fromSec(state_point_->timestamp);
    scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_world_)
        pub_laser_cloud_world_.publish(scan_msg);
    if (bag_.isOpen())
        bag_.write("/cloud_registered", ros::Time().fromSec(state_point_->timestamp), scan_msg);
}

void LaserMapping::PublishFrameEffectWorld() {
    PointCloud::Ptr eff_scan(new PointCloud);
    eff_scan->resize(eff_num_);
    int j =0;
    for (int i = 0; i < scan_down_world_->size(); i++) {
        if (eff_mask_[i]) {
            eff_scan->points[j] = scan_down_world_->points[i];
            j++;
        }
    }
    sensor_msgs::PointCloud2 eff_scan_msg;
    pcl::toROSMsg(*eff_scan, eff_scan_msg);
    eff_scan_msg.header.stamp = ros::Time().fromSec(state_point_->timestamp);
    eff_scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_effect_world_)
        pub_laser_cloud_effect_world_.publish(eff_scan_msg);
    if (bag_.isOpen())
        bag_.write("/cloud_registered_effect_world", ros::Time().fromSec(state_point_->timestamp), eff_scan_msg);
}
void LaserMapping::PublishImage() {
    if (!measures_.img_.empty()) {
        cv_bridge::CvImage image_msg;
        image_msg.header.stamp = ros::Time().fromSec(measures_.end_time_);
        image_msg.header.frame_id = "world";
        image_msg.encoding = measures_.img_.type() == CV_8UC1 ? "mono8" : "bgr8";
        image_msg.image = measures_.img_;
        if (pub_image_)
            pub_image_.publish(*image_msg.toImageMsg());
    }
}

void LaserMapping::PublishFrustrum() {
    if (!pub_frustrum_)
        return;

    // virtual pinhole camera, drawn at the body-to-camera extrinsic (identity if uncalibrated)
    constexpr double kDepth = 1.0;       // frustum depth, meters
    constexpr double kHalfFovH = 0.5236; // 60 deg horizontal half-FOV, rad
    constexpr double kHalfFovV = 0.3927; // 45 deg vertical half-FOV, rad
    const double half_w = kDepth * std::tan(kHalfFovH);
    const double half_h = kDepth * std::tan(kHalfFovV);

    const Pose3 &body_from_cam = param->extrin_ic_;
    const Eigen::Vector3d apex_cam(0, 0, 0);
    const std::array<Eigen::Vector3d, 4> corners_cam = {
        Eigen::Vector3d(-half_w, -half_h, kDepth),
        Eigen::Vector3d(half_w, -half_h, kDepth),
        Eigen::Vector3d(half_w, half_h, kDepth),
        Eigen::Vector3d(-half_w, half_h, kDepth),
    };

    const Eigen::Vector3d apex = body_from_cam * apex_cam;
    std::array<Eigen::Vector3d, 4> corners;
    for (int i = 0; i < 4; ++i)
        corners[i] = body_from_cam * corners_cam[i];

    visualization_msgs::Marker marker;
    marker.header.frame_id = "body";
    marker.header.stamp = ros::Time().fromSec(state_point_->timestamp);
    marker.ns = "frustrum";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.02;  // line width
    marker.color.r = 1.0;
    marker.color.g = 0.65;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    auto add_point = [&marker](const Eigen::Vector3d &p) {
        geometry_msgs::Point pt;
        pt.x = p.x();
        pt.y = p.y();
        pt.z = p.z();
        marker.points.push_back(pt);
    };

    // apex to each corner
    for (int i = 0; i < 4; ++i) {
        add_point(apex);
        add_point(corners[i]);
    }
    // rectangle at the far plane
    for (int i = 0; i < 4; ++i) {
        add_point(corners[i]);
        add_point(corners[(i + 1) % 4]);
    }

    pub_frustrum_.publish(marker);
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    TrajectoryGenerator::save_to_tumtxt(trajectory_, traj_file);

    Trajectory cam_traj;
    for (auto stamp_pose : trajectory_) {
        stamp_pose.pose = stamp_pose.pose * param->extrin_ic_.Isometry3d();
        cam_traj.emplace_back(stamp_pose);
    }
    std::string cam_traj_file = fs::path(traj_file).parent_path().string() + "/cam_traj_log.txt";
    TrajectoryGenerator::save_to_tumtxt(cam_traj, cam_traj_file);
    if (cam_traj.empty()) {
        std::cout << "No camera trajectory to save." << std::endl;
        return;
    }
    TrajectoryGenerator::save_to_pcd(cam_traj, fs::path(traj_file).parent_path().string() + "/cam_traj_log.ply");
}

void LaserMapping::Finish() {
    bag_.close();
    if (param->pcd_save_interval_ > 0) {
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
    }

    mapper->ScanFilter();
    boost::filesystem::create_directories(output_dir + "/global/");
    mapper->ExportMap(output_dir + "/init.pcd");
    TrajectoryGenerator::save_to_tumtxt(mapper->ExportStampedPoses(), output_dir + "/init.txt");
    std::unordered_map<ScanPair, PairData> loops;
    if (mapper->options_.lc_enable) {
        loops = mapper->DetectLoopClosure();
        mapper->SaveLoopToPcd(output_dir + "/global/loops.pcd");
    }
    if (!loops.empty()) {
        mapper->PoseGraphOptimize();
        TrajectoryGenerator::save_to_tumtxt(mapper->ExportStampedPoses(), output_dir + "/global/pgo.txt");
        mapper->ExportMap(output_dir + "/global/pgo.pcd");
    }
    if (mapper->options_.ba_enable) {
        for (int i = 0; i < mapper->options_.ba_iters; ++i) {
            mapper->BundleAdjustment();
        }
    }

    // export the poses in body frame
    std::cout << "Exporting final map and trajectory..." << std::endl;
    mapper->ExportMap(output_dir + "/final.pcd");
    TrajectoryGenerator::save_to_tumtxt(mapper->ExportStampedPoses(), output_dir + "/final.txt");

    // export COLMAP
    if (param->image_save_en_) {
        // only for the keyscan, erase others
        for (auto it = sfm_data_.images_.begin(); it != sfm_data_.images_.end();) {
            if (mapper->keyscans_.count(it->first) > 0) {
                Image::Ptr im = it->second;
                im->cam_from_world_ = (mapper->keyscans_[it->first]->world_from_body * param->extrin_ic_).GetInverse();
                ++it;
            } else
                it = sfm_data_.images_.erase(it);
        }
        // write image list txt
        std::ofstream ofs(output_dir + "/images.txt");
        for (const auto &[im_id, im] : sfm_data_.images_) {
            ofs << im->name_ << std::endl;
        }
        ofs.close();

        // write colmap
        std::string colmap_dir = output_dir + "/colmap_result/";
        fs::create_directories(colmap_dir);
        LOG(INFO) << "Exporting COLMAP result to " << colmap_dir;
        sfm_data_.WriteCOLMAPText(colmap_dir);
    }

}

}  // namespace faster_lio


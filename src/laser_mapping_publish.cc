#include <tf/transform_broadcaster.h>

#include <array>
#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <sstream>
#include <cv_bridge/cv_bridge.h>
#include "laser_mapping.h"
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
namespace fs = boost::filesystem;

namespace faster_lio {

void LaserMapping::PublishOdometry() {
    auto interp_pose = resple_lio_->spl.Interpolate(resple_lio_->spl.maxTimeNs() - 1);
    nav_msgs::Odometry body_odometry;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = interp_pose.translation().x();
    pose_stamped.pose.position.y = interp_pose.translation().y();
    pose_stamped.pose.position.z = interp_pose.translation().z();
    Eigen::Quaterniond q(interp_pose.linear());
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    pose_stamped.header.stamp = ros_stamp;
    pose_stamped.header.frame_id = "world";

    body_odometry.header.frame_id = "world";
    body_odometry.child_frame_id = "body";
    body_odometry.header.stamp = ros_stamp;
    body_odometry.pose.pose = pose_stamped.pose;

    // publish
    static nav_msgs::Path path;
    static int call_once = [&]() {
        path.header.frame_id = "world";
        path.header.stamp = ros_stamp;
        return 0;
    }();
    if (pub_odom_aft_mapped_)
        pub_odom_aft_mapped_.publish(body_odometry);
    path.poses.push_back(pose_stamped);
    if (pub_path_)
        pub_path_.publish(path);
    if (bag_.isOpen())
        bag_.write("/Odometry", ros_stamp, pose_stamped);
    if (bag_.isOpen())
        bag_.write("/path", ros_stamp, path);

    // transform broadcast
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion tf_q;
    tf::Vector3 t;
    t.setX(interp_pose.translation().x());
    t.setY(interp_pose.translation().y());
    t.setZ(interp_pose.translation().z());
    tf_q.setX(q.x());
    tf_q.setY(q.y());
    tf_q.setZ(q.z());
    tf_q.setW(q.w());
    transform.setOrigin(t);
    transform.setRotation(tf_q);
    br.sendTransform(tf::StampedTransform(transform, body_odometry.header.stamp, "world", "body"));

}

void LaserMapping::PublishFrameWorld() {
    if (resple_lio_->pt_meas.empty())
        return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr points_world(new PointCloud);
    points_world->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        faster_lio::Point point{};
        point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
        points_world->push_back(point);
    }
    sensor_msgs::PointCloud2 scan_msg;
    pcl::toROSMsg(*points_world, scan_msg);
    scan_msg.header.stamp = ros_stamp;
    scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_world_)
        pub_laser_cloud_world_.publish(scan_msg);
    if (bag_.isOpen())
        bag_.write("/cloud_registered", ros_stamp, scan_msg);
}

void LaserMapping::PublishFrameEffectWorld() {
    if (resple_lio_->pt_meas.empty())
        return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr eff_points(new PointCloud);
    eff_points->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        if (!resple_lio_->pt_meas[i].if_valid)
            continue;
        faster_lio::Point point{};
        point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
        eff_points->push_back(point);
    }
    sensor_msgs::PointCloud2 eff_scan_msg;
    pcl::toROSMsg(*eff_points, eff_scan_msg);
    eff_scan_msg.header.stamp = ros_stamp;
    eff_scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_effect_world_)
        pub_laser_cloud_effect_world_.publish(eff_scan_msg);
    if (bag_.isOpen())
        bag_.write("/cloud_registered_effect_world", ros_stamp, eff_scan_msg);
}
void LaserMapping::PublishImage() {
    // if (!measures_.img_.empty()) {
    //     cv_bridge::CvImage image_msg;
    //     image_msg.header.stamp = ros::Time().fromSec(measures_.end_time_);
    //     image_msg.header.frame_id = "world";
    //     image_msg.encoding = measures_.img_.type() == CV_8UC1 ? "mono8" : "bgr8";
    //     image_msg.image = measures_.img_;
    //     if (pub_image_)
    //         pub_image_.publish(*image_msg.toImageMsg());
    // }
}

void LaserMapping::PublishFrustrum() {
    if (!pub_frustrum_)
        return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
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

    marker.header.stamp = ros_stamp;
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

void LaserMapping::Savetrajectory() {
    // save body pose
    TrajectoryGenerator::save_to_tumtxt(trajectory_, (fs::path(output_dir) / "traj_log.txt").string());
    TrajectoryGenerator::save_to_pcd(trajectory_, (fs::path(output_dir) / "traj.pcd").string());

    // save camera pose
    Trajectory cam_traj;
    for (auto stamp_pose : trajectory_) {
        stamp_pose.pose = stamp_pose.pose * param->extrin_ic_.Isometry3d();
        cam_traj.emplace_back(stamp_pose);
    }
    TrajectoryGenerator::save_to_tumtxt(cam_traj, (fs::path(output_dir) / "cam_traj_log.txt").string());
    TrajectoryGenerator::save_to_pcd(cam_traj, (fs::path(output_dir) / "cam_traj.pcd").string());
}

void LaserMapping::Finish() {
    bag_.close();
    if (param->pcd_save_en_ && !wait_save_points_->empty()) {
        double time_s = global_offset_time + wait_save_points_->front().GetTimeSec();
        std::stringstream stamp_string;
        stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s;

        pcl::UniformSampling<Point> sampler;
        sampler.setInputCloud(wait_save_points_);
        sampler.setRadiusSearch(param->ds_scan_voxel);
        sampler.filter(*wait_save_points_);
        fs::create_directories(fs::path(output_dir) / "map" );
        pcl::io::savePCDFileBinary((fs::path(output_dir) / "map" / stamp_string.str()).string(), *wait_save_points_);
    }
}

}  // namespace faster_lio


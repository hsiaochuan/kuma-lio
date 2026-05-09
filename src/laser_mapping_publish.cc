#include <tf/transform_broadcaster.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <sstream>

#include "global_optimizor.h"
#include "laser_mapping.h"

namespace fs = boost::filesystem;

namespace faster_lio {

void LaserMapping::PublishPath() {
    geometry_msgs::PoseStamped msg_body_pose;
    msg_body_pose.pose.position.x = state_point_.pos(0);
    msg_body_pose.pose.position.y = state_point_.pos(1);
    msg_body_pose.pose.position.z = state_point_.pos(2);
    msg_body_pose.pose.orientation.x = state_point_.rot.coeffs()[0];
    msg_body_pose.pose.orientation.y = state_point_.rot.coeffs()[1];
    msg_body_pose.pose.orientation.z = state_point_.rot.coeffs()[2];
    msg_body_pose.pose.orientation.w = state_point_.rot.coeffs()[3];
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time_);
    msg_body_pose.header.frame_id = "world";

    /*** if path is too large, the rvis will crash ***/
    path_.poses.push_back(msg_body_pose);
    pub_path_.publish(path_);
}

void LaserMapping::PublishOdometry() {
    nav_msgs::Odometry odom_aft_mapped;
    odom_aft_mapped.header.frame_id = "world";
    odom_aft_mapped.child_frame_id = "body";
    odom_aft_mapped.header.stamp = ros::Time().fromSec(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time_);
    odom_aft_mapped.pose.pose.position.x = state_point_.pos(0);
    odom_aft_mapped.pose.pose.position.y = state_point_.pos(1);
    odom_aft_mapped.pose.pose.position.z = state_point_.pos(2);
    odom_aft_mapped.pose.pose.orientation.x = state_point_.rot.coeffs()[0];
    odom_aft_mapped.pose.pose.orientation.y = state_point_.rot.coeffs()[1];
    odom_aft_mapped.pose.pose.orientation.z = state_point_.rot.coeffs()[2];
    odom_aft_mapped.pose.pose.orientation.w = state_point_.rot.coeffs()[3];
    pub_odom_aft_mapped_.publish(odom_aft_mapped);
    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom_aft_mapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odom_aft_mapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odom_aft_mapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odom_aft_mapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odom_aft_mapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odom_aft_mapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped.pose.pose.position.x, odom_aft_mapped.pose.pose.position.y,
                                    odom_aft_mapped.pose.pose.position.z));
    q.setW(odom_aft_mapped.pose.pose.orientation.w);
    q.setX(odom_aft_mapped.pose.pose.orientation.x);
    q.setY(odom_aft_mapped.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped.header.stamp, "world", "body"));
}

void LaserMapping::PublishFrameWorld() const {
    PointCloud::Ptr scan_world;
    scan_world = scan_down_world_;

    sensor_msgs::PointCloud2 scan_msg;
    pcl::toROSMsg(*scan_world, scan_msg);
    scan_msg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    scan_msg.header.frame_id = "world";
    pub_laser_cloud_world_.publish(scan_msg);
}

void LaserMapping::PublishFrameEffectWorld() {
    PointCloud::Ptr laser_cloud(new PointCloud);
    laser_cloud->resize(corr_pts_.size());
    for (int i = 0; i < corr_pts_.size(); i++) {
        laser_cloud->at(i).getVector3fMap() =
            (state_point_.rot * corr_pts_[i].head<3>().cast<double>() + state_point_.pos).cast<float>();
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laser_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
    laserCloudmsg.header.frame_id = "world";
    pub_laser_cloud_effect_world_.publish(laserCloudmsg);
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
    TrajectoryGenerator::save_to_pcd(cam_traj, fs::path(traj_file).parent_path().string() + "/cam_traj_log.ply");
}

void LaserMapping::Finish() {
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


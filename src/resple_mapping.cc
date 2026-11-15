// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "resple_mapping.h"

#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>
#include <boost/filesystem.hpp>
#include <unordered_map>

namespace fs = boost::filesystem;

namespace kuma_lio {
double global_offset_time = std::numeric_limits<double>::quiet_NaN();

namespace {
void SortByTime(const PointCloud::Ptr& scan) {
    std::sort(scan->points.begin(), scan->points.end(),
              [](const Point& a, const Point& b) { return a.GetTimeNs() < b.GetTimeNs(); });
}
}  // namespace

bool RespleMapping::Init(const std::string& config_fname) {
    LOG(INFO) << "init resple mapping from " << config_fname;
    Eigen::setNbThreads(1);
    param = std::make_shared<RespleParam>();
    param->LoadFromYaml(config_fname);
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->param = param->Preprocess();

    resple_lio_ = std::make_shared<RESPLE_LIO>();
    resple_lio_->param = param;
    resple_lio_->Init();

    if (param->bag_save_en_) {
        bag_.open((fs::path(output_dir) / "dump.bag").string(), rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }
    return true;
}

void RespleMapping::InitialPublishers(ros::NodeHandle& nh) {
    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
}

void RespleMapping::AddScanToBuffer(const PointCloud::Ptr& scan) {
    if (scan->empty()) {
        LOG(INFO) << "empty in scan, no points pushed into buffer";
        return;
    }
    int skip_scan_points = 0;
    for (int i = 0; i < scan->size(); ++i) {
        if (points_buffer_.empty()) {
            points_buffer_.push_back(scan->at(i));
            continue;
        }
        if (points_buffer_.back().GetTimeNs() <= scan->at(i).GetTimeNs()) {
            points_buffer_.push_back(scan->at(i));
        } else {
            skip_scan_points++;
        }
    }

    if (skip_scan_points != 0)
        LOG(INFO) << "skip " << skip_scan_points << " scan points for timestamp of point error, " << scan->size()
                  << " points total";
}

void RespleMapping::PushScan(const PointCloud::Ptr& scan) {
    if (!scan) return;
    if (param->ds_scan_voxel > 0) DownsampleScan(scan);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    AddScanToBuffer(scan);
}

void RespleMapping::DownsampleScan(const PointCloud::Ptr& scan) const {
    // One point per voxel of the whole scan (lidar frame): the centroid of its returns, with their mean time, as
    // RESPLE's VoxelGrid.  Per scan, not per 10 ms knot window: the knot windows would keep ~1.7x the points.
    if (scan->empty()) return;
    struct Cell {
        Vec3 p = Vec3::Zero();
        double dt = 0.0;  // [ns] summed time after the first point
        double intensity = 0.0;
        int n = 0;
    };
    std::unordered_map<VOXEL_LOCATION, Cell> cells;
    std::vector<VOXEL_LOCATION> order;  // first-seen order, so the result does not depend on the hash
    const int64_t t0 = scan->points.front().GetTimeNs();
    for (const Point& pt : scan->points) {
        const Vec3 p = pt.getVector3fMap().cast<double>();
        const VOXEL_LOCATION key(p, param->ds_scan_voxel);
        auto it = cells.find(key);
        if (it == cells.end()) {
            it = cells.emplace(key, Cell()).first;
            order.push_back(key);
        }
        Cell& c = it->second;
        c.p += p;
        c.dt += static_cast<double>(pt.GetTimeNs() - t0);
        c.intensity += pt.intensity;
        c.n++;
    }
    PointCloud::Ptr out(new PointCloud);
    out->reserve(order.size());
    for (const VOXEL_LOCATION& key : order) {
        const Cell& c = cells[key];
        Point q{};
        q.getVector3fMap() = (c.p / c.n).cast<float>();
        q.intensity = static_cast<float>(c.intensity / c.n);
        q.SetTimeNs(t0 + std::llround(c.dt / c.n));
        out->push_back(q);
    }
    scan->swap(*out);
    SortByTime(scan);
}

PointCloud::Ptr RespleMapping::ScanFromPCL(const sensor_msgs::PointCloud2::ConstPtr& msg) const {
    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return nullptr;
    timestamp = timestamp - global_offset_time;
    PointCloud::Ptr scan(new PointCloud());
    switch (LidarTypeFromString(param->lidar_type)) {
        case LidarType::OUSTER:
            scan = preprocess_->OusterHandler(msg, S_TO_NS(timestamp));
            break;
        case LidarType::HESAI:
            scan = preprocess_->HesaiHandler(msg, S_TO_NS(timestamp));
            break;
        case LidarType::VELODYNE_POINTCLOUD2:
            scan = preprocess_->VelodynePointsHandler(msg, S_TO_NS(timestamp));
            break;
        default:
            throw std::logic_error("unknown lidar type");
    }
    SortByTime(scan);
    return scan;
}

PointCloud::Ptr RespleMapping::ScanFromLivox(const livox_ros_driver::CustomMsg::ConstPtr& msg) const {
    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return nullptr;
    timestamp = timestamp - global_offset_time;
    PointCloud::Ptr scan = preprocess_->LivoxHandler(msg, S_TO_NS(timestamp));
    SortByTime(scan);
    return scan;
}

void RespleMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr& msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    double timestamp = msg_in->header.stamp.toSec();
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    Imu imu;
    imu.time_ns = S_TO_NS(timestamp);
    imu.angular_velocity.x() = msg_in->angular_velocity.x;
    imu.angular_velocity.y() = msg_in->angular_velocity.y;
    imu.angular_velocity.z() = msg_in->angular_velocity.z;
    imu.linear_acceleration.x() = msg_in->linear_acceleration.x;
    imu.linear_acceleration.y() = msg_in->linear_acceleration.y;
    imu.linear_acceleration.z() = msg_in->linear_acceleration.z;
    if (param->acc_ratio_) imu.linear_acceleration *= RESPLE_LIO::GRAVITY_NORM;
    imu_buffer_.emplace_back(imu);
}

void RespleMapping::Run() {
    // as RESPLE: gravity from 15 IMU samples, the map from the first 0.1 s of points
    static constexpr int INIT_IMU_COUNT = 15;
    static constexpr int64_t INIT_POINTS_SPAN = S_TO_NS(0.1);
    if (!if_localmap_initialized) {
        if (points_buffer_.empty() || imu_buffer_.empty()) return;

        // stationary init: gravity from the mean of the first IMU samples; the biases start at zero
        if (!if_inertial_initialized) {
            if (static_cast<int>(imu_buffer_.size()) < INIT_IMU_COUNT) {
                LOG(INFO) << "Imu buffer is not enough, wait";
                return;
            }
            Vec3 acc_sum = Vec3::Zero();
            for (int i = 0; i < INIT_IMU_COUNT; ++i) acc_sum += imu_buffer_[i].linear_acceleration;

            Vec3 mean_acc = acc_sum / INIT_IMU_COUNT;
            Vec3 gravity_body = mean_acc.normalized() * RESPLE_LIO::GRAVITY_NORM;
            Mat3 R0 = Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
            double yaw = std::atan2(R0(1, 0), R0(0, 0));
            R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;
            Eigen::Quaterniond q0(R0);
            resple_lio_->gravity = q0 * gravity_body;
            resple_lio_->spl.init(param->dt_ns, 0, q0);
            if_inertial_initialized = true;
            LOG(INFO) << "Inertial initialized, gravity: " << resple_lio_->gravity.transpose();
        }

        // the first INIT_POINTS_SPAN of points, at the initial pose, seed the ikd-tree
        std::int64_t first_point_time_ns = points_buffer_.front().GetTimeNs();
        std::int64_t end_time_ns = first_point_time_ns + INIT_POINTS_SPAN;
        if (points_buffer_.back().GetTimeNs() < end_time_ns || imu_buffer_.back().time_ns < end_time_ns) {
            LOG(INFO) << "buffer is not enough, wait";
            return;
        }
        PointCloud::Ptr init_points(new PointCloud);
        while (resple_lio_->spl.maxTimeNs() < end_time_ns) {
            resple_lio_->spl.addOneStateKnot(Vec3::Zero(), Vec3::Zero());
        }
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < resple_lio_->spl.maxTimeNs()) {
            init_points->emplace_back(points_buffer_.front());
            points_buffer_.pop_front();
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < resple_lio_->spl.maxTimeNs()) {
            imu_buffer_.pop_front();
        }

        if (init_points->size() < 100) {
            LOG(WARNING) << "only " << init_points->size() << " points for map init, waiting";
            return;
        }
        PointCloud::Ptr init_points_world(new PointCloud);
        Pose3 world_from_lidar(resple_lio_->spl.q_idle, Vec3::Zero());
        world_from_lidar = world_from_lidar * param->extrin_il_;
        pcl::transformPointCloud(*init_points, *init_points_world, world_from_lidar.Mat4d(), true);
        resple_lio_->kd_tree->Build(init_points_world->points);
        if_localmap_initialized = true;
        LOG(INFO) << "localmap initialized with " << init_points_world->size() << " points";
        return;
    }

    while (CollectMeasures(resple_lio_->spl.maxTimeNs() + param->dt_ns)) {
        resple_lio_->propRCP();
        resple_lio_->ProcessMeasurement(measures_);
        PublishROSMsg();
        PostUpdate();
    }
}

bool RespleMapping::CollectMeasures(const std::int64_t& meas_end) {
    if (points_buffer_.empty()) return false;
    if (imu_buffer_.empty()) return false;

    // buffer is not beyond the measures end
    if (imu_buffer_.back().time_ns < meas_end) return false;
    if (points_buffer_.back().GetTimeNs() < meas_end) return false;

    // push the imu data
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < meas_end) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    // push the lidar points data
    measures_.lidar_->clear();
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < meas_end) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }
    measures_.mea_end_ = meas_end;

    if (measures_.lidar_->empty()) {
        LOG(INFO) << "Empty lidar points in measures";
    }
    return true;
}

void RespleMapping::PostUpdate() {
    // the newest instant of the spline, as RESPLE logs it
    int64_t interp_time_ns = resple_lio_->spl.maxTimeNs() - 1;
    Eigen::Isometry3d body_pose = resple_lio_->spl.Interpolate(interp_time_ns);
    trajectory_.emplace_back(global_offset_time + interp_time_ns * 1e-9, body_pose);
}

void RespleMapping::PublishROSMsg() {
    PublishOdometry(resple_lio_->spl.Interpolate(resple_lio_->spl.maxTimeNs() - 1), resple_lio_->spl.maxTimeNs());
    PublishFrameWorld();
    PublishFrameEffectWorld();
}

void RespleMapping::PublishOdometry(const Eigen::Isometry3d& pub_pose, std::int64_t time_ns) {
    nav_msgs::Odometry body_odometry;
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = pub_pose.translation().x();
    pose_stamped.pose.position.y = pub_pose.translation().y();
    pose_stamped.pose.position.z = pub_pose.translation().z();
    Eigen::Quaterniond q(pub_pose.linear());
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(time_ns));
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
    if (pub_odom_aft_mapped_) pub_odom_aft_mapped_.publish(body_odometry);
    path.poses.push_back(pose_stamped);
    if (pub_path_) pub_path_.publish(path);
    if (bag_.isOpen()) bag_.write("/Odometry", ros_stamp, pose_stamped);
    if (bag_.isOpen()) bag_.write("/path", ros_stamp, path);

    // transform broadcast
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion tf_q;
    tf::Vector3 t;
    t.setX(pub_pose.translation().x());
    t.setY(pub_pose.translation().y());
    t.setZ(pub_pose.translation().z());
    tf_q.setX(q.x());
    tf_q.setY(q.y());
    tf_q.setZ(q.z());
    tf_q.setW(q.w());
    transform.setOrigin(t);
    transform.setRotation(tf_q);
    br.sendTransform(tf::StampedTransform(transform, body_odometry.header.stamp, "world", "body"));
}

void RespleMapping::PublishFrameWorld() {
    if (resple_lio_->pt_meas.empty()) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr points_world(new PointCloud);
    points_world->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        kuma_lio::Point point{};
        point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
        points_world->push_back(point);
    }
    sensor_msgs::PointCloud2 scan_msg;
    pcl::toROSMsg(*points_world, scan_msg);
    scan_msg.header.stamp = ros_stamp;
    scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_world_) pub_laser_cloud_world_.publish(scan_msg);
    if (bag_.isOpen()) bag_.write("/cloud_registered", ros_stamp, scan_msg);
}

void RespleMapping::PublishFrameEffectWorld() {
    if (resple_lio_->pt_meas.empty()) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr eff_points(new PointCloud);
    eff_points->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        if (!resple_lio_->pt_meas[i].if_valid) continue;
        kuma_lio::Point point{};
        point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
        eff_points->push_back(point);
    }
    sensor_msgs::PointCloud2 eff_scan_msg;
    pcl::toROSMsg(*eff_points, eff_scan_msg);
    eff_scan_msg.header.stamp = ros_stamp;
    eff_scan_msg.header.frame_id = "world";
    if (pub_laser_cloud_effect_world_) pub_laser_cloud_effect_world_.publish(eff_scan_msg);
    if (bag_.isOpen()) bag_.write("/cloud_registered_effect_world", ros_stamp, eff_scan_msg);
}

void RespleMapping::Savetrajectory() {
    LOG(INFO) << "Save states in imu frame";
    TrajectoryGenerator::save_to_tumtxt(trajectory_, (fs::path(output_dir) / "traj_log.txt").string());
}

void RespleMapping::Finish() { bag_.close(); }

}  // namespace kuma_lio

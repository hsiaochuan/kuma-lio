#include <iomanip>
#include <sstream>

#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/filesystem.hpp>
#include "laser_mapping.h"
namespace fs = boost::filesystem;
namespace faster_lio {
LaserMapping::LaserMapping() = default;
void LaserMapping::AddScanToBuffer(const PointCloud::Ptr &scan) {
    if (scan->empty()) {
        LOG(INFO) << "empty in scan, no points pushed into buffer";
        return;
    }
    std::sort(scan->points.begin(), scan->points.end(), [](const Point &a, const Point &b) {
        return a.GetTimeNs() < b.GetTimeNs();
    });
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
        LOG(INFO) << "skip " << skip_scan_points << " scan points for timestamp of point error, " << scan->size() << " points total";
}
void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;

    // loop
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
    }
    last_timestamp_lidar_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
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
    AddScanToBuffer(scan);
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;

    // loop
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
    }
    last_timestamp_lidar_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) {
        return;
    }
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan;
    scan = preprocess_->LivoxHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
}
void LaserMapping::VelodyneScanCallBack(const velodyne_msgs::VelodyneScan::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    CHECK(!msg->packets.empty()) << "VelodyneScan message is empty!";
    // the msg->header.stamp is 100 ms later than the msg->packets front
    double timestamp = msg->packets.front().stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;

    // loop
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
    }
    last_timestamp_lidar_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan(new PointCloud());
    scan = preprocess_->VelodyneScanHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    double timestamp = msg_in->header.stamp.toSec();

    // loop
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }
    last_timestamp_imu_ = timestamp;
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

void LaserMapping::ImageMsgCallBack(const sensor_msgs::Image::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        double timestamp = msg_in->header.stamp.toSec();
        timestamp += param->camera_time_offset_;
        if (timestamp < last_timestamp_camera_) {
            LOG(WARNING) << "image loop back, clear buffer";
            image_buffer_.clear();
        }
        last_timestamp_camera_ = timestamp;
        if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
        timestamp = timestamp - global_offset_time;
        Image image;
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        image.image_data_ = img;
        image.time_ns_ = S_TO_NS(timestamp);
        image_buffer_.emplace_back(image);
    }
    img_count++;
}

void LaserMapping::CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        double timestamp = msg_in->header.stamp.toSec();
        timestamp += param->camera_time_offset_;
        if (timestamp < last_timestamp_camera_) {
            LOG(WARNING) << "image loop back, clear buffer";
            image_buffer_.clear();
        }
        last_timestamp_camera_ = timestamp;
        if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
        timestamp = timestamp - global_offset_time;
        Image image;
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        image.image_data_ = img;
        image.time_ns_ = S_TO_NS(timestamp);
        image_buffer_.emplace_back(image);
    }
    img_count++;
}

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    if (!param->LoadFromYaml(config_fname)) return false;
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->param = param;

    // the estimation algorithm
    resple_lio_ = std::make_shared<RESPLE_LIO>();
    resple_lio_->param = param;
    resple_lio_->Init();

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}

void LaserMapping::InitialSubscribers(ros::NodeHandle &nh) {
    LidarType lidar_type = LidarTypeFromString(param->lidar_type);
    if (lidar_type == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000,
            [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else if (lidar_type == LidarType::VELODYNE_SCAN) {
        sub_pcl_ = nh.subscribe<velodyne_msgs::VelodyneScan>(
            param->lidar_topic_, 200000,
            [this](const velodyne_msgs::VelodyneScan::ConstPtr &msg) { VelodyneScanCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000,
            [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });
    bool compressed_image = false;
    if (!compressed_image) {
        sub_img_ = nh.subscribe<sensor_msgs::Image>(
            param->camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr &msg) { ImageMsgCallBack(msg); });
    } else {
        sub_img_ = nh.subscribe<sensor_msgs::CompressedImage>(
            param->camera_topic_, 200000,
            [this](const sensor_msgs::CompressedImage::ConstPtr &msg) { CompressedImageCallBack(msg); });
    }
}
void LaserMapping::InitialPublishers(ros::NodeHandle &nh) {
    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    if (param->camera_enable_) pub_image_ = nh.advertise<sensor_msgs::Image>("/image_raw", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
}

void LaserMapping::Run() {
    static constexpr int kInitImuNum = 20;
    static constexpr int64_t kInitMapSpanNs = S_TO_NS(0.3);
    if (!resple_lio_->if_localmap_initialized) {
        if (points_buffer_.empty()) return;

        std::int64_t first_point_time_ns = points_buffer_.front().GetTimeNs();

        if (!resple_lio_->if_inertial_initialized) {
            if (static_cast<int>(imu_buffer_.size()) < kInitImuNum) return;
            Vec3 acc_sum = Vec3::Zero();
            Vec3 acc_gyro = Vec3::Zero();
            for (int i = 0; i < kInitImuNum; ++i) {
                acc_sum += imu_buffer_[i].linear_acceleration;
                acc_gyro += imu_buffer_[i].angular_velocity;
            }

            Vec3 mean_acc = acc_sum / kInitImuNum;
            Vec3 gravity_body = mean_acc.normalized() * RESPLE_LIO::GRAVITY_NORM;
            Eigen::Matrix3d R0 =
                Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
            double yaw = std::atan2(R0(1, 0), R0(0, 0));
            R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;
            Eigen::Quaterniond q0(R0);
            resple_lio_->bg = acc_gyro / kInitImuNum;
            resple_lio_->gravity = q0 * gravity_body;
            resple_lio_->spl.init(param->dt_ns, 0, q0);
            resple_lio_->if_inertial_initialized = true;
            LOG(INFO) << " s, gravity " << resple_lio_->gravity.transpose();
        }

        if (points_buffer_.back().GetTimeNs() < first_point_time_ns + kInitMapSpanNs) return;
        std::int64_t end_time_ns = first_point_time_ns + kInitMapSpanNs;
        PointCloud::Ptr init_points(new PointCloud);
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < end_time_ns) {
            init_points->emplace_back(points_buffer_.front());
            points_buffer_.pop_front();
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < end_time_ns) {
            imu_buffer_.pop_front();
        }
        while (resple_lio_->spl.maxTimeNs() < end_time_ns) {
            resple_lio_->spl.addOneStateKnot(Vec3::Zero(), Vec3::Zero());
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
        resple_lio_->if_localmap_initialized = true;
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

void LaserMapping::PostUpdate() {
    int64_t t_ns = resple_lio_->spl.maxTimeNs() - param->dt_ns;
    Eigen::Isometry3d body_pose = resple_lio_->spl.Interpolate(t_ns);
    trajectory_.emplace_back(global_offset_time + t_ns * 1e-9, body_pose);

    if (param->pcd_save_en_) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "map");
            return 0;
        }();
        wait_save_points_->reserve(wait_save_points_->size() + resple_lio_->pt_meas.size());
        for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
            Point point{};
            point.getVector3fMap() = resple_lio_->pt_meas[i].point_world.cast<float>();
            point.intensity = resple_lio_->pt_meas[i].point.intensity;
            point.SetTimeNs(resple_lio_->pt_meas[i].time_ns);
            wait_save_points_->push_back(point);
        }

        if (wait_save_points_->size() >= 2 &&
            wait_save_points_->back().GetTimeNs() - wait_save_points_->front().GetTimeNs() >
                S_TO_NS(param->pcd_save_interval_)) {
            double time_s = global_offset_time + wait_save_points_->front().GetTimeSec();
            std::stringstream stamp_string;
            stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s;

            pcl::UniformSampling<Point> sampler;
            sampler.setInputCloud(wait_save_points_);
            sampler.setRadiusSearch(param->ds_scan_voxel);
            sampler.filter(*wait_save_points_);
            pcl::io::savePCDFileBinary((fs::path(output_dir) / "map" / stamp_string.str()).string(),
                                       *wait_save_points_);
            wait_save_points_->clear();
        }
    }
}
void LaserMapping::PublishROSMsg() {
    // publish
    PublishOdometry();
    PublishFrameWorld();
    PublishFrameEffectWorld();
    PublishImage();
    PublishFrustrum();
}
bool LaserMapping::CollectMeasures(const std::int64_t &meas_end) {
    if (points_buffer_.empty()) return false;
    if (param->imu_enable_ && imu_buffer_.empty()) return false;

    // imu buffer is not beyond the measures end
    if (imu_buffer_.back().time_ns < meas_end) return false;
    // points buffer is not beyond the measures end
    if (points_buffer_.back().GetTimeNs() < meas_end) return false;

    // push the imu data
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < meas_end) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    measures_.lidar_->clear();
    int num_collected = 0;
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < meas_end) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
        num_collected++;
    }

    if (measures_.lidar_->empty()) {
        LOG(INFO) << "Empty lidar points in measures";
    }
    return true;
}
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
    if (resple_lio_->pt_meas.empty()) return;
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
    if (pub_laser_cloud_world_) pub_laser_cloud_world_.publish(scan_msg);
    if (bag_.isOpen()) bag_.write("/cloud_registered", ros_stamp, scan_msg);
}

void LaserMapping::PublishFrameEffectWorld() {
    if (resple_lio_->pt_meas.empty()) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    PointCloud::Ptr eff_points(new PointCloud);
    eff_points->reserve(resple_lio_->pt_meas.size());
    for (int i = 0; i < resple_lio_->pt_meas.size(); ++i) {
        if (!resple_lio_->pt_meas[i].if_valid) continue;
        faster_lio::Point point{};
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
void LaserMapping::PublishImage() {}

void LaserMapping::PublishFrustrum() {
    if (!pub_frustrum_) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(resple_lio_->spl.maxTimeNs()));
    // virtual pinhole camera, drawn at the body-to-camera extrinsic (identity if uncalibrated)
    constexpr double kDepth = 1.0;        // frustum depth, meters
    constexpr double kHalfFovH = 0.5236;  // 60 deg horizontal half-FOV, rad
    constexpr double kHalfFovV = 0.3927;  // 45 deg vertical half-FOV, rad
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
    for (int i = 0; i < 4; ++i) corners[i] = body_from_cam * corners_cam[i];

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
        fs::create_directories(fs::path(output_dir) / "map");
        pcl::io::savePCDFileBinary((fs::path(output_dir) / "map" / stamp_string.str()).string(), *wait_save_points_);
    }
}
}  // namespace faster_lio

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

#include <iomanip>
#include <map>
#include <sstream>

#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/filesystem.hpp>
#include "laser_mapping.h"

#include <pcl/io/ply_io.h>

#include "rtklib_.h"
#include "rtklib_func.h"
namespace fs = boost::filesystem;

namespace kuma_lio {
double global_offset_time = std::numeric_limits<double>::quiet_NaN();
}  // namespace kuma_lio

int showmsg(const char* format, ...) { return 0; }
void settspan(gtime_t ts, gtime_t te) {}
void settime(gtime_t time) {}

namespace kuma_lio {
LaserMapping::LaserMapping() = default;
void LaserMapping::AddScanToBuffer(const PointCloud::Ptr& scan) {
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
namespace {
void SortByTime(const PointCloud::Ptr& scan) {
    std::sort(scan->points.begin(), scan->points.end(),
              [](const Point& a, const Point& b) { return a.GetTimeNs() < b.GetTimeNs(); });
}
}  // namespace

void LaserMapping::PushScan(const PointCloud::Ptr& scan) {
    if (!scan) return;
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    AddScanToBuffer(scan);
}

void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg) { PushScan(ScanFromPCL(msg)); }

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr& msg) { PushScan(ScanFromLivox(msg)); }

PointCloud::Ptr LaserMapping::ScanFromPCL(const sensor_msgs::PointCloud2::ConstPtr& msg) const {
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

PointCloud::Ptr LaserMapping::ScanFromLivox(const livox_ros_driver::CustomMsg::ConstPtr& msg) const {
    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return nullptr;
    timestamp = timestamp - global_offset_time;
    PointCloud::Ptr scan = preprocess_->LivoxHandler(msg, S_TO_NS(timestamp));
    SortByTime(scan);
    return scan;
}
void LaserMapping::ImageCallBack(const sensor_msgs::Image::ConstPtr& msg) {
    // time offset
    const double timestamp = msg->header.stamp.toSec() + param->image_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    // keep one image in img_filter
    static int img_count = 0;
    if (img_count++ % param->img_filter != 0) return;
    RgbImage image;
    if (!image.FromRaw(msg->data.data(), msg->width, msg->height, msg->step, msg->encoding)) {
        LOG(WARNING) << "image skipped, encoding " << msg->encoding;
        return;
    }
    image.time_ns = S_TO_NS(timestamp - global_offset_time);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    image_buff_.push_back(std::move(image));
}

void LaserMapping::CompressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    // time offset
    const double timestamp = msg->header.stamp.toSec() + param->image_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    // keep one image in img_filter
    static int img_count = 0;
    if (img_count++ % param->img_filter != 0) return;
    RgbImage image;
    // image_decode off: only the bytes are kept, decoding a 5 MP JPEG costs ~60 ms per frame
    if (!image.SetEncoded(msg->data.data(), msg->data.size(), msg->format) ||
        (false && !image.Decode(msg->data.data(), msg->data.size()))) {
        LOG(WARNING) << "image skipped, format " << msg->format;
        return;
    }
    image.time_ns = S_TO_NS(timestamp - global_offset_time);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    image_buff_.push_back(std::move(image));
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr& msg_in) {
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
    if (param->acc_ratio_) imu.linear_acceleration *= FastLio::GRAVITY_NORM;
    imu_buffer_.emplace_back(imu);
}

bool LaserMapping::ReadObsNav(const std::string& obs_fname, const std::string& nav_fname, const std::string& sol_fname) {
    gtime_t ts{}, te{};  // zero disable the time window filter

    // rcv index 1 for the observation file; the navigation file carries no receiver data.
    if (readrnxt(obs_fname.c_str(), 1, ts, te, 0.0, "", &obss, &nav, nullptr) <= 0) {
        LOG(ERROR) << "failed to read observation file: " << obs_fname;
        return false;
    }
    if (readrnxt(nav_fname.c_str(), 0, ts, te, 0.0, "", nullptr, &nav, nullptr) <= 0) {
        LOG(ERROR) << "failed to read navigation file: " << nav_fname;
        return false;
    }

    // RTKLIB expects observations sorted by time/sat and ephemerides uniquified before use.
    sortobs(&obss);
    uniqnav(&nav);
    if (fast_lio_) {
        fast_lio_->obss = obss;
        fast_lio_->nav = nav;
    }

    LOG(INFO) << "read " << obss.n << " observations, " << nav.n << " eph, " << nav.ng << " geph, " << nav.ns
              << " seph";
    if (obss.n <= 0 || nav.n + nav.ng <= 0) {
        LOG(ERROR) << "no usable GNSS data; check that RTKLIB is built with -DENAGLO/-DENAGAL/-DENACMP";
        return false;
    }
    CHECK(!std::isnan(global_offset_time));
    for (int i = 0; i < obss.n;) {
        obss_id_buff_.push_back(i);
        gtime_t utc = gpst2utc(obss.data[i].time);
        double utc_d = double(utc.time) + utc.sec;
        obss_time_ns_buff_.push_back(S_TO_NS(utc_d - global_offset_time));
        int j = i;
        while (std::abs(timediff(obss.data[j].time, obss.data[i].time)) < DTTOL && j < obss.n) ++j;
        i = j;
    }
    std::vector<std::string> files{obs_fname, nav_fname};
    std::vector<const char*> infiles;
    for (const auto& f : files) infiles.push_back(f.c_str());

    // position
    filopt_t filopt{};
    const int ret = postpos(ts, te, 0.0, 0.0, &param->prcopt, &param->solopt, &filopt, infiles.data(),
                            static_cast<int>(infiles.size()), sol_fname.c_str(), "", "");
    if (ret != 0) {
        LOG(ERROR) << "postpos failed with status " << ret;
        return false;
    }
    return true;
}

bool LaserMapping::Init(const std::string& config_fname, const std::string& camera_fname, const std::string& pgo_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    Eigen::setNbThreads(1);
    param = std::make_shared<LaserMappingParam>();
    param->LoadFromYaml(config_fname);
    if (!camera_fname.empty()) param->LoadCameraFromYaml(camera_fname);
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->param.livox_tag_mask = param->livox_tag_mask;
    preprocess_->param.point_filter_num = param->point_filter_num;
    preprocess_->param.blind = param->blind;
    preprocess_->param.det_range_ = param->det_range_;

    // the estimation algorithm
    fast_lio_ = std::make_shared<FastLio>();
    fast_lio_->param = param;
    fast_lio_->Init(pgo_fname);

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}

void LaserMapping::InitialSubscribers(ros::NodeHandle& nh) {
    LidarType lidar_type = LidarTypeFromString(param->lidar_type);
    if (lidar_type == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000,
            [this](const livox_ros_driver::CustomMsg::ConstPtr& msg) { LivoxPCLCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000,
            [this](const sensor_msgs::PointCloud2::ConstPtr& msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr& msg) { IMUCallBack(msg); });
    const std::string& img = param->img_topic_;
    if (img.size() >= 11 && img.compare(img.size() - 11, 11, "/compressed") == 0) {
        sub_img_ = nh.subscribe<sensor_msgs::CompressedImage>(
            img, 100, [this](const sensor_msgs::CompressedImage::ConstPtr& msg) { CompressedImageCallback(msg); });
    } else if (!img.empty()) {
        sub_img_ = nh.subscribe<sensor_msgs::Image>(
            img, 100, [this](const sensor_msgs::Image::ConstPtr& msg) { ImageCallBack(msg); });
    }
}
void LaserMapping::InitialPublishers(ros::NodeHandle& nh) {
    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
}

void LaserMapping::RunFastLio() {
    static constexpr int INIT_IMU_COUNT = 20;
    static constexpr int64_t INIT_POINTS_SPAN = S_TO_NS(0.3);
    if (!if_localmap_initialized) {
        if (points_buffer_.empty() || imu_buffer_.empty()) return;
        if (static_cast<int>(imu_buffer_.size()) < INIT_IMU_COUNT) return;

        const std::int64_t end_time_ns = points_buffer_.front().GetTimeNs() + INIT_POINTS_SPAN;
        if (points_buffer_.back().GetTimeNs() < end_time_ns || imu_buffer_.back().time_ns < end_time_ns) return;

        // standing still: gravity and the gyroscope bias come from the first samples
        Vec3 acc_sum = Vec3::Zero();
        Vec3 gyr_sum = Vec3::Zero();
        for (int i = 0; i < INIT_IMU_COUNT; ++i) {
            acc_sum += imu_buffer_[i].linear_acceleration;
            gyr_sum += imu_buffer_[i].angular_velocity;
        }
        const Vec3 gravity_body = (acc_sum / INIT_IMU_COUNT).normalized() * FastLio::GRAVITY_NORM;
        Mat3 R0 = Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
        const double yaw = std::atan2(R0(1, 0), R0(0, 0));
        R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;

        FastLioState state;
        state.R = R0;
        state.bg = gyr_sum / INIT_IMU_COUNT;

        PointCloud::Ptr init_points(new PointCloud);
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < end_time_ns) {
            init_points->emplace_back(points_buffer_.front());
            points_buffer_.pop_front();
        }
        Imu last_imu = imu_buffer_.front();
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < end_time_ns) {
            last_imu = imu_buffer_.front();
            imu_buffer_.pop_front();
        }
        while (!image_buff_.empty() && image_buff_.front().time_ns < end_time_ns) {
            image_buff_.pop_front();
        }
        if (init_points->size() < 100) {
            LOG(WARNING) << "only " << init_points->size() << " points for map init, waiting";
            return;
        }
        PointCloud::Ptr init_points_world(new PointCloud);
        pcl::transformPointCloud(*init_points, *init_points_world,
                                 (Pose3(Eigen::Quaterniond(R0), Vec3::Zero()) * param->extrin_il_).Mat4d(), true);
        std::vector<Vec3> init_ps;
        for (const Point& pt : init_points_world->points) init_ps.push_back(pt.getVector3fMap().cast<double>());
        fast_lio_->lba->InitMap(init_ps, param->lba_.range_err * param->lba_.range_err + param->lba_.pose_var);
        fast_lio_->Start(state, R0 * gravity_body, end_time_ns, last_imu);
        if_localmap_initialized = true;
        LOG(INFO) << "localmap initialized with " << init_points_world->size() << " points";
        return;
    }

    while (CollectMeasuresFastLio()) {
        fast_lio_->ProcessMeasurement(measures_);
        PublishROSMsgFastLio();
        PostUpdateFastLio();
    }
}

void LaserMapping::PostUpdateFastLio() {
    const std::int64_t time_ns = measures_.mea_end_;
    trajectory_.emplace_back(global_offset_time + time_ns * 1e-9, fast_lio_->x.Isometry());

    if (param->pcd_save_en_) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "scans");
            return 0;
        }();
        static Vec3 last_save_pos = Vec3::Zero();
        if (last_save_pos == Vec3::Zero() || (last_save_pos - fast_lio_->x.p).norm() > param->pcd_save_interval_) {
            last_save_pos = fast_lio_->x.p;
            PointCloud::Ptr save_points{new PointCloud()};
            save_points->reserve(fast_lio_->points.size());
            for (const FastLioPoint& pt : fast_lio_->points) {
                Point point{};
                point.getVector3fMap() = pt.body.cast<float>();  // with traj_log.txt, what run_pgo and run_clean read
                point.SetTimeNs(time_ns);
                save_points->push_back(point);
            }

            double time_s = global_offset_time + time_ns * 1e-9;
            std::stringstream stamp_string;
            stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s << ".ply";
            pcl::io::savePLYFileBinary((fs::path(output_dir) / "scans" / stamp_string.str()).string(),
                                       *save_points);
        }
    }

    if (param->image_save_en_ && !measures_.rgb_image_.Empty()) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "images");
            return 0;
        }();
        static Vec3 last_save_pos = Vec3::Zero();
        static Mat3 last_save_R = Mat3::Identity();
        constexpr double kMaxAngleChange = 20 * M_PI / 180;
        if (last_save_pos == Vec3::Zero() ||
            (last_save_pos - fast_lio_->x.p).norm() > param->image_save_interval_ ||
            LogMat(last_save_R.transpose() * fast_lio_->x.R).norm() > kMaxAngleChange) {
            last_save_pos = fast_lio_->x.p;
            last_save_R = fast_lio_->x.R;
            double time_s = global_offset_time + time_ns * 1e-9;
            std::stringstream stamp_string;
            stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << time_s;
            if (!measures_.rgb_image_.Save((fs::path(output_dir) / "images" / stamp_string.str()).string()))
                LOG(INFO) << "failed to save image at " << time_s;
        }
    }
}

void LaserMapping::PublishROSMsgFastLio() {
    const std::int64_t time_ns = measures_.mea_end_;
    PublishOdometry(fast_lio_->x.Isometry(), time_ns);
    if (!fast_lio_->points.empty()) {
        const ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(time_ns));
        PointCloud::Ptr points_world(new PointCloud);
        PointCloud::Ptr eff_points(new PointCloud);
        for (const FastLioPoint& pt : fast_lio_->points) {
            Point point{};
            point.getVector3fMap() = pt.world.cast<float>();
            points_world->push_back(point);
            if (pt.valid) eff_points->push_back(point);
        }
        sensor_msgs::PointCloud2 scan_msg;
        pcl::toROSMsg(*points_world, scan_msg);
        scan_msg.header.stamp = ros_stamp;
        scan_msg.header.frame_id = "world";
        if (pub_laser_cloud_world_) pub_laser_cloud_world_.publish(scan_msg);
        if (bag_.isOpen()) bag_.write("/cloud_registered", ros_stamp, scan_msg);

        sensor_msgs::PointCloud2 eff_scan_msg;
        pcl::toROSMsg(*eff_points, eff_scan_msg);
        eff_scan_msg.header.stamp = ros_stamp;
        eff_scan_msg.header.frame_id = "world";
        if (pub_laser_cloud_effect_world_) pub_laser_cloud_effect_world_.publish(eff_scan_msg);
        if (bag_.isOpen()) bag_.write("/cloud_registered_effect_world", ros_stamp, eff_scan_msg);
    }
    PublishFrustrum(time_ns);
}
void LaserMapping::ClearMeasuresBefore(const std::int64_t& time_ns) {
    while (!image_buff_.empty() && image_buff_.front().time_ns <= time_ns) {
        image_buff_.pop_front();
    }
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() <= time_ns) {
        points_buffer_.pop_front();
    }
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns <= time_ns) {
        imu_buffer_.pop_front();
    }
    while (!obss_time_ns_buff_.empty() && obss_time_ns_buff_.front() <= time_ns) {
        obss_time_ns_buff_.pop_front();
        obss_id_buff_.pop_front();
    }
}
bool LaserMapping::CollectMeasuresFastLio() {
    std::int64_t meas_end;
    ClearMeasuresBefore(fast_lio_->last_end_ns);
    if (param->fast_lio_.mode == FastLioMode::LIVO) {
        if (image_buff_.empty()) return false;
        meas_end = image_buff_.front().time_ns;
    } else if (param->fast_lio_.mode == FastLioMode::LIGO && !obss_time_ns_buff_.empty() &&
               obss_time_ns_buff_.front() - fast_lio_->last_end_ns <= S_TO_NS(1.5 * param->lba_.frame_dt)) {
        meas_end = obss_time_ns_buff_.front();
    } else {
        meas_end = fast_lio_->last_end_ns + S_TO_NS(param->lba_.frame_dt);
    }

    if (points_buffer_.empty() || imu_buffer_.empty()) return false;
    if (imu_buffer_.back().time_ns < meas_end) return false;
    if (points_buffer_.back().GetTimeNs() < meas_end) return false;

    measures_.mea_end_ = meas_end;
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().time_ns <= meas_end) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    measures_.lidar_->clear();
    while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() <= meas_end) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }
    measures_.obs_id = -1;
    if (!obss_time_ns_buff_.empty() && obss_time_ns_buff_.front() <= meas_end) {
        measures_.obs_id = obss_id_buff_.front();
        measures_.obs_time_ns = obss_time_ns_buff_.front();
        obss_id_buff_.pop_front();
        obss_time_ns_buff_.pop_front();
    }
    measures_.rgb_image_ = RgbImage();
    while (!image_buff_.empty() && image_buff_.front().time_ns <= meas_end) {
        measures_.rgb_image_ = std::move(image_buff_.front());
        image_buff_.pop_front();
    }
    return true;
}

void LaserMapping::PublishOdometry(const Eigen::Isometry3d& pub_pose, std::int64_t time_ns) {
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

void LaserMapping::PublishFrustrum(std::int64_t time_ns) {
    if (!pub_frustrum_) return;
    ros::Time ros_stamp = ros::Time().fromSec(NS_TO_S(time_ns));
    // virtual pinhole camera, drawn at the body-to-camera extrinsic (identity if uncalibrated)
    constexpr double kDepth = 1.0;        // frustum depth, meters
    constexpr double kHalfFovH = 0.5236;  // 60 deg horizontal half-FOV, rad
    constexpr double kHalfFovV = 0.3927;  // 45 deg vertical half-FOV, rad
    const double half_w = kDepth * std::tan(kHalfFovH);
    const double half_h = kDepth * std::tan(kHalfFovV);

    const Pose3& body_from_cam = param->extrin_ic_;
    const Vec3 apex_cam(0, 0, 0);
    const std::array<Vec3, 4> corners_cam = {
        Vec3(-half_w, -half_h, kDepth),
        Vec3(half_w, -half_h, kDepth),
        Vec3(half_w, half_h, kDepth),
        Vec3(-half_w, half_h, kDepth),
    };

    const Vec3 apex = body_from_cam * apex_cam;
    std::array<Vec3, 4> corners;
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

    auto add_point = [&marker](const Vec3& p) {
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
    LOG(INFO) << "Save states in imu frame";
    TrajectoryGenerator::save_to_tumtxt(trajectory_,
                                        (fs::path(output_dir) / "traj_log.txt").string());
}

void LaserMapping::Finish() {
    if (fast_lio_ && fast_lio_->loop) {
        fast_lio_->loop->Finish(fast_lio_->lba->WindowScans(), global_offset_time);
        const pgo::ScanNodes& nodes = fast_lio_->loop->backend().nodes();
        CHECK_EQ(nodes.size(), trajectory_.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            CHECK_LT(std::abs(nodes[i].time - trajectory_[i].time), 1e-6) << "node " << i << " is not frame " << i;
            trajectory_[i].pose = nodes[i].pose;
        }
        const fs::path dir = fs::path(output_dir) / "pgo";
        fs::create_directories(dir);
        fast_lio_->loop->SaveDebug(dir.string());
    }
    bag_.close();
}
}  // namespace kuma_lio

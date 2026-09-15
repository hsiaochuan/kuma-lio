#include <iomanip>
#include <sstream>

#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/filesystem.hpp>
#include "laser_mapping.h"
#include <opencv2/opencv.hpp>
namespace fs = boost::filesystem;


int showmsg(const char *format, ...) { return 0; }
void settspan(gtime_t ts, gtime_t te) {}
void settime(gtime_t time) {}

namespace faster_lio {
LaserMapping::LaserMapping() = default;
void LaserMapping::AddScanToBuffer(const PointCloud::Ptr& scan) {
    if (scan->empty()) {
        LOG(INFO) << "empty in scan, no points pushed into buffer";
        return;
    }
    std::sort(scan->points.begin(), scan->points.end(),
              [](const Point& a, const Point& b) { return a.GetTimeNs() < b.GetTimeNs(); });
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
void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
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

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) {
        return;
    }
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan;
    scan = preprocess_->LivoxHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
}
void LaserMapping::VelodyneScanCallBack(const velodyne_msgs::VelodyneScan::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    CHECK(!msg->packets.empty()) << "VelodyneScan message is empty!";
    // the msg->header.stamp is 100 ms later than the msg->packets front
    double timestamp = msg->packets.front().stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan(new PointCloud());
    scan = preprocess_->VelodyneScanHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
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
    if (param->acc_ratio_) imu.linear_acceleration *= RESPLE_LIO::GRAVITY_NORM;
    imu_buffer_.emplace_back(imu);
}

void LaserMapping::ImageMsgCallBack(const sensor_msgs::Image::ConstPtr& msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        double timestamp = msg_in->header.stamp.toSec();
        timestamp += param->camera_time_offset_;
        if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
        timestamp = timestamp - global_offset_time;
        RgbImage im;
        im.image_data_ = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        im.time_ns_ = S_TO_NS(timestamp);
        im.filename = std::to_string(im.time_ns_) + ".jpg";
        image_buffer_.emplace_back(im);
    }
    img_count++;
}

void LaserMapping::CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr& msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        double timestamp = msg_in->header.stamp.toSec();
        timestamp += param->camera_time_offset_;
        if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
        timestamp = timestamp - global_offset_time;
        RgbImage im;
        im.image_data_ = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        im.time_ns_ = S_TO_NS(timestamp);
        im.filename = std::to_string(im.time_ns_) + ".jpg";
        image_buffer_.emplace_back(im);
    }
    img_count++;
}
void LaserMapping::GnssSolutionCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    gtime_t gpst{};
    gpst.time = msg->header.stamp.sec;
    gpst.sec = NS_TO_S(msg->header.stamp.nsec);

    sol_t sol{};
    sol.time = gpst;
    const double pos[3] = {msg->latitude * D2R, msg->longitude * D2R, msg->altitude};
    pos2ecef(pos, sol.rr);
    const uint8_t fix_type = static_cast<uint8_t>(msg->status.status);
    const uint8_t carr_soln = static_cast<uint8_t>(msg->status.service);
    if (fix_type == 0 || fix_type == 5) {
        sol.stat = SOLQ_NONE;
    } else if (carr_soln == 2) {
        sol.stat = SOLQ_FIX;
    } else if (carr_soln == 1) {
        sol.stat = SOLQ_FLOAT;
    } else {
        sol.stat = SOLQ_SINGLE;
    }
    sol.type = 0;
}
bool LaserMapping::ReadObsNav(const std::string& obs_fname, const std::string& nav_fname, const string& sol_fname) {
    gtime_t ts{}, te{};  // zero epochs disable the time window filter

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
    resple_lio_->obss = obss;
    resple_lio_->nav = nav;

    LOG(INFO) << "read " << obss.n << " observations, " << nav.n << " eph, " << nav.ng << " geph, " << nav.ns
              << " seph";
    if (obss.n <= 0 || nav.n + nav.ng <= 0) {
        LOG(ERROR) << "no usable GNSS data; check that RTKLIB is built with -DENAGLO/-DENAGAL/-DENACMP";
        return false;
    }
    CHECK(!std::isnan(global_offset_time));
    for (int i = 0; i < obss.n;) {
        obss_id_buffer_.push_back(i);
        gtime_t utc = gpst2utc(obss.data[i].time);
        double utc_d = double(utc.time) + utc.sec;
        obss_time_buffer_.push_back(S_TO_NS(utc_d - global_offset_time));
        int j = i;
        while (std::abs(timediff(obss.data[j].time, obss.data[i].time)) < DTTOL && j < obss.n) ++j;
        i = j;
    }
    std::vector<std::string> files{obs_fname, nav_fname};
    std::vector<const char*> infile;
    for (const auto& f : files) infile.push_back(f.c_str());

    // position
    filopt_t filopt{};
    const int ret = postpos(ts, te, 0.0, 0.0, &param->prcopt, &param->solopt, &filopt, infile.data(),
                            static_cast<int>(infile.size()), sol_fname.c_str(), "", "");
    if (ret != 0) {
        LOG(ERROR) << "postpos failed with status " << ret;
        return false;
    }
    return true;
}

bool LaserMapping::Init(const std::string& config_fname, const std::string& camera_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    param->LoadFromYaml(config_fname);
    if (param->camera_enable_ && !camera_fname.empty()) {
        param->cam_model_ = ReadCameraFromCamerasText(camera_fname);
        if (!param->cam_model_) LOG(INFO) << "Null camera model";
    }
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

void LaserMapping::InitialSubscribers(ros::NodeHandle& nh) {
    LidarType lidar_type = LidarTypeFromString(param->lidar_type);
    if (lidar_type == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000,
            [this](const livox_ros_driver::CustomMsg::ConstPtr& msg) { LivoxPCLCallBack(msg); });
    } else if (lidar_type == LidarType::VELODYNE_SCAN) {
        sub_pcl_ = nh.subscribe<velodyne_msgs::VelodyneScan>(
            param->lidar_topic_, 200000,
            [this](const velodyne_msgs::VelodyneScan::ConstPtr& msg) { VelodyneScanCallBack(msg); });
    } else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000,
            [this](const sensor_msgs::PointCloud2::ConstPtr& msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr& msg) { IMUCallBack(msg); });
    bool compressed_image = false;
    if (!compressed_image) {
        sub_img_ = nh.subscribe<sensor_msgs::Image>(
            param->camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr& msg) { ImageMsgCallBack(msg); });
    } else {
        sub_img_ = nh.subscribe<sensor_msgs::CompressedImage>(
            param->camera_topic_, 200000,
            [this](const sensor_msgs::CompressedImage::ConstPtr& msg) { CompressedImageCallBack(msg); });
    }
}
void LaserMapping::InitialPublishers(ros::NodeHandle& nh) {
    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    if (param->camera_enable_) pub_image_ = nh.advertise<sensor_msgs::Image>("/image_raw", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
}

void LaserMapping::Run() {
    static constexpr int INIT_IMU_COUNT = 20;
    static constexpr int64_t INIT_POINTS_SPAN = S_TO_NS(0.3);
    static constexpr int64_t DYNAMIC_INIT_SPAN = S_TO_NS(1.0);
    if (!if_localmap_initialized) {
        if (points_buffer_.empty() || imu_buffer_.empty() || obss_id_buffer_.empty()) return;
        std::int64_t first_point_time_ns = points_buffer_.front().GetTimeNs();
        std::int64_t first_gnss_time_ns = obss_time_buffer_.front();
        std::int64_t first_imu_time_ns = imu_buffer_.front().time_ns;
        std::int64_t first_time_ns = std::max(first_point_time_ns, std::max(first_gnss_time_ns, first_imu_time_ns));
        std::int64_t end_time_ns = first_time_ns + DYNAMIC_INIT_SPAN;
        // clear the data before the first time
        while (!points_buffer_.empty() && points_buffer_.front().GetTimeNs() < first_time_ns) {
            points_buffer_.pop_front();
        }
        while (!imu_buffer_.empty() && imu_buffer_.front().time_ns < first_time_ns) {
            imu_buffer_.pop_front();
        }
        while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < first_time_ns) {
            obss_time_buffer_.pop_front();
            obss_id_buffer_.pop_front();
        }

    }  // if not init
    if (!if_localmap_initialized) {
        if (points_buffer_.empty() || imu_buffer_.empty()) return;

        if (!if_inertial_initialized) {
            if (static_cast<int>(imu_buffer_.size()) < INIT_IMU_COUNT) {
                LOG(INFO) << "Imu buffer is not enough, wait";
                return;
            }
            Vec3 acc_sum = Vec3::Zero();
            Vec3 acc_gyro = Vec3::Zero();
            for (int i = 0; i < INIT_IMU_COUNT; ++i) {
                acc_sum += imu_buffer_[i].linear_acceleration;
                acc_gyro += imu_buffer_[i].angular_velocity;
            }

            Vec3 mean_acc = acc_sum / INIT_IMU_COUNT;
            Vec3 gravity_body = mean_acc.normalized() * RESPLE_LIO::GRAVITY_NORM;
            Mat3 R0 = Eigen::Quaterniond::FromTwoVectors(gravity_body.normalized(), Vec3(0, 0, 1)).toRotationMatrix();
            double yaw = std::atan2(R0(1, 0), R0(0, 0));
            R0 = Eigen::AngleAxisd(-yaw, Vec3::UnitZ()).toRotationMatrix() * R0;
            Eigen::Quaterniond q0(R0);
            resple_lio_->bg = acc_gyro / INIT_IMU_COUNT;
            resple_lio_->gravity = q0 * gravity_body;
            resple_lio_->spl.init(param->dt_ns, 0, q0);
            if_inertial_initialized = true;
            LOG(INFO) << "Inertial initialized, gravity: " << resple_lio_->gravity.transpose();
        }

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
        while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < resple_lio_->spl.maxTimeNs()) {
            obss_time_buffer_.pop_front();
            obss_id_buffer_.pop_front();
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

void LaserMapping::PostUpdate() {
    int64_t interp_time_ns = resple_lio_->spl.maxTimeNs() - param->dt_ns;
    Eigen::Isometry3d body_pose = resple_lio_->spl.Interpolate(interp_time_ns);
    trajectory_.emplace_back(global_offset_time + interp_time_ns * 1e-9, body_pose);

    if (!measures_.images.empty()) {
        for (int i = 0; i < measures_.images.size(); ++i) {
            Eigen::Isometry3d cam_state = resple_lio_->spl.Interpolate(measures_.images[i].time_ns_);
            cam_states.emplace_back(measures_.images[i].time_ns_, Pose3(cam_state) * param->extrin_ic_);
        }
    }

    if (param->bag_save_en_ && !measures_.images.empty()) {
        static int call_once = [&]() {
            fs::create_directories(fs::path(output_dir) / "images");
            return 0;
        }();
        for (int i = 0; i < measures_.images.size(); ++i) {
            fs::path im_save_fname = fs::path(output_dir) / "images" / measures_.images[i].filename;
            cv::imwrite(im_save_fname.string(), measures_.images[i].image_data_);
        }
    }
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
bool LaserMapping::CollectMeasures(const std::int64_t& meas_end) {
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
    // process the raw measurement
    measures_.obs_id = -1;
    while (!obss_time_buffer_.empty() && obss_time_buffer_.front() < meas_end) {
        measures_.obs_id = obss_id_buffer_.front();
        measures_.obs_time_ns = obss_time_buffer_.front();
        obss_id_buffer_.pop_front();
        obss_time_buffer_.pop_front();
    }
    if (measures_.lidar_->empty()) {
        LOG(INFO) << "Empty lidar points in measures";
    }
    return true;
}
void LaserMapping::PublishOdometry() {
    Eigen::Isometry3d pub_pose;
    pub_pose = resple_lio_->spl.Interpolate(resple_lio_->spl.maxTimeNs() - 1);

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
    TrajectoryGenerator::save_to_tumtxt(trajectory_, (fs::path(output_dir) / "traj_log.txt").string());

    LOG(INFO) << "Save states in camera frame";
    std::ofstream cam_ofs((fs::path(output_dir) / "cam_log.txt").string());
    cam_ofs << std::fixed;
    for (auto& cam_state : cam_states) {
        int64_t& time_ns = cam_state.first;
        Pose3& cam_pose = cam_state.second;
        cam_ofs << std::to_string(time_ns) + ".jpg" << " ";
        cam_ofs << cam_pose.Trans().x() << " " << cam_pose.Trans().y() << " " << cam_pose.Trans().z() << " ";
        cam_ofs << cam_pose.Quat().x() << " " << cam_pose.Quat().y() << " " << cam_pose.Quat().z() << " "
                << cam_pose.Quat().w() << std::endl;
    }
    cam_ofs.close();
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

#include <iomanip>
#include <sstream>

#include <boost/filesystem.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include "global_optimizor.h"
#include "laser_mapping.h"
#include "utils.h"

namespace fs = boost::filesystem;
namespace faster_lio {

static Vec3f InterpolatedPixel(cv::Mat & img, Vec2 pc)
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]);
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref - u_ref_i);
    const float subpix_v_ref = (v_ref - v_ref_i);
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    auto pixel_tl = (cv::Vec3f)img.at<cv::Vec3b>(v_ref_i, u_ref_i);
    auto pixel_tr = (cv::Vec3f)img.at<cv::Vec3b>(v_ref_i, u_ref_i + 1);
    auto pixel_bl = (cv::Vec3f)img.at<cv::Vec3b>(v_ref_i + 1, u_ref_i);
    auto pixel_br = (cv::Vec3f)img.at<cv::Vec3b>(v_ref_i + 1, u_ref_i + 1);
    float B = w_ref_tl * pixel_tl[0] + w_ref_tr * pixel_tr[0] + w_ref_bl * pixel_bl[0] + w_ref_br * pixel_br[0];
    float G = w_ref_tl * pixel_tl[1] + w_ref_tr * pixel_tr[1] + w_ref_bl * pixel_bl[1] + w_ref_br * pixel_br[1];
    float R = w_ref_tl * pixel_tl[2] + w_ref_tr * pixel_tr[2] + w_ref_bl * pixel_bl[2] + w_ref_br * pixel_br[2];
    Vec3f pixel(B, G, R);
    return pixel;
}

LaserMapping::LaserMapping() = default;

bool LaserMapping::Init(const std::string &config_fname) {
    LOG(INFO) << "init laser mapping from " << config_fname;
    param = std::make_shared<LaserMappingParam>();
    if (!param->LoadFromYaml(config_fname))
        return false;
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    preprocess_->blind_ = param->blind;
    preprocess_->point_filter_num_ = param->point_filter_num;
    preprocess_->lidar_type_ = LidarTypeFromString(param->lidar_type);
    preprocess_->max_range = param->det_range_;

    // the estimation algorithm
    ieskf_lio = std::make_shared<IeskfLio>();
    ieskf_lio->SetPriorInitPose(prior_init_position_, prior_init_rotation_);
    if (!ieskf_lio->Init(param))
        return false;

    scan_sampler_.setLeafSize(param->scan_filter_size, param->scan_filter_size, param->scan_filter_size);

    mapper = std::make_shared<GlobalOptimizor>();
    GlobalOptimizor::Options global_options;
    global_options.LoadFromYaml(config_fname);
    mapper->options_ = global_options;
    mapper->output_dir = output_dir;

    if (param->bag_save_en_) {
        bag_.open(output_dir + "/dump.bag", rosbag::bagmode::Write);
        if (!bag_.isOpen()) {
            throw std::runtime_error("Could not open bag");
        }
    }

    return true;
}

void LaserMapping::LoadPriorMap(const std::string &prior_map_fname) {
    ieskf_lio->LoadPriorMap(prior_map_fname);
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    if (preprocess_->lidar_type_ == LidarType::LIVOX) {
        sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
            param->lidar_topic_, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
    } else if (preprocess_->lidar_type_ == LidarType::VELODYNE_SCAN) {
        sub_pcl_ = nh.subscribe<velodyne_msgs::VelodyneScan>(
            param->lidar_topic_, 200000, [this](const velodyne_msgs::VelodyneScan::ConstPtr &msg) { VelodyneScanCallBack(msg); });
    }else {
        sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
            param->lidar_topic_, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
    }

    sub_imu_ = nh.subscribe<sensor_msgs::Imu>(param->imu_topic_, 200000,
                                              [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });
    sub_img_ = nh.subscribe<sensor_msgs::Image>(param->camera_topic_, 200000, [this](const sensor_msgs::Image::ConstPtr &msg) {
        ImageMsgCallBack(msg);
    });

    pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world", 100000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
    if (param->camera_enable_)
        pub_image_ = nh.advertise<sensor_msgs::Image>("/image_raw", 100000);
    pub_frustrum_ = nh.advertise<visualization_msgs::Marker>("/frustrum", 100000);
    if (param->localization_enable_) {
        auto map_cloud = ieskf_lio->PriorMap();
        LOG(INFO) << "publish prior map " << map_cloud->size() << " points";
        auto pub_prior_map = nh.advertise<sensor_msgs::PointCloud2>("/prior_map", 100000);
        sensor_msgs::PointCloud2 prior_map_msg;
        pcl::toROSMsg(*map_cloud, prior_map_msg);
        prior_map_msg.header.frame_id = "world";
        prior_map_msg.header.stamp = ros::Time::now();
        while (1) {
            if (pub_prior_map.getNumSubscribers() > 0) {
                for (int i = 0; i < 3; ++i) {
                    pub_prior_map.publish(prior_map_msg);
                    LOG(INFO) << "publish prior map";
                    sleep(1);
                }
                break;
            }
        }
    }
}

void LaserMapping::Run() {
    // sync the lidar and imu data, if no data or not synced, return true
    if (!SyncPackages()) {
        return;
    }

    // run the estimation algorithm on the synced measure
    if (!ieskf_lio->ProcessMeasure(measures_)) {
        return;
    }

    BuildColorScan();
    PublishROSMsg();
    PostUpdate();
}
void LaserMapping::BuildColorScan() {
    auto state_point_ = ieskf_lio->State();
    auto scan_down_world_ = ieskf_lio->ScanDownWorld();
    if (param->camera_enable_) {
        color_scan_world_->reserve(scan_down_world_->size());
        Pose3 camera_from_world = (Pose3(state_point_->rot, state_point_->pos) * param->extrin_ic_).GetInverse();
        for (int i = 0; i < scan_down_world_->size(); i++) {
            Vec3 pc = camera_from_world * scan_down_world_->at(i).getVector3fMap().cast<double>();
            auto p_im = param->camera_->project_and_valid(pc,3);
            if (p_im) {
                ColorPoint color_point{};
                Vec3f color = InterpolatedPixel(measures_.img_, *p_im);
                color_point.rgb = RGBToU32(color[2], color[1], color[0]);
                color_point.getVector3fMap() = scan_down_world_->at(i).getVector3fMap();
                color_point.intensity = scan_down_world_->at(i).intensity;
                color_scan_world_->emplace_back(color_point);
            }
        }
    }else {
        color_scan_world_->resize(scan_down_world_->size());
        for (int i =0; i < scan_down_world_->size(); i++) {
            color_scan_world_->at(i).getVector3fMap() = scan_down_world_->at(i).getVector3fMap();
            color_scan_world_->at(i).intensity = scan_down_world_->at(i).intensity;
        }
    }
}
void LaserMapping::PostUpdate() {
    auto state_point_ = ieskf_lio->State();
    auto scan_undistort_ = ieskf_lio->ScanUndistort();

    // save to trajectory
    Pose3 body_pose = Pose3(state_point_->rot, state_point_->pos);
    trajectory_.emplace_back(state_point_->timestamp + global_offset_time, body_pose.Isometry3d());

    // add scan frame to global optimize
    static scan_t scan_id = 1;
    ScanFrame::Ptr scan = std::make_shared<ScanFrame>(scan_id);
    std::stringstream stamp_string;
    stamp_string << std::setw(17) << std::setfill('0') << std::fixed << std::setprecision(8) << measures_.end_time_;
    scan->cloud_fname = output_dir + "/scans/" + stamp_string.str() + ".pcd";
    scan->world_from_body = Pose3(state_point_->rot, state_point_->pos);
    scan->timestamp = measures_.end_time_;
    mapper->AddScan(scan);
    scan_id++;

    // save the pcd
    if (param->image_save_en_ && !measures_.img_.empty()) {
        // save image
        static auto once = fs::create_directories(output_dir + "/images");
        cv::imwrite(output_dir + "/images/" + stamp_string.str() + ".jpg", measures_.img_);
    }
    if (param->pcd_save_en_) {
        static auto once = fs::create_directories(output_dir + "/scans");
        if (scan_undistort_->size() > 0)
            pcl::io::savePCDFileBinary(scan->cloud_fname, *scan_undistort_);
    }
    if (param->pcd_save_en_) {
        *pcl_wait_save_ += *ieskf_lio->ScanDownWorld();
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (param->pcd_save_interval_ > 0 && scan_wait_num >= param->pcd_save_interval_) {
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
            pcl_wait_save_->clear();
            scan_wait_num = 0;
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
bool LaserMapping::SyncPackages() {
    if (points_buffer_.empty())
        return false;
    if (param->imu_enable_ && imu_buffer_.empty())
        return false;
    if (param->camera_enable_ && image_buffer_.empty())
        return false;

    const double state_timestamp = ieskf_lio->State()->timestamp;

    // set the measure end timestamp
    if (std::isnan(state_timestamp)) {
        // for first time
        if (param->camera_enable_) {
            measures_.end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        } else
            measures_.end_time_ = points_buffer_.front().timestamp + param->scan_interval_;
    } else if (measures_.end_time_ == state_timestamp) {
        // after the update, incre the end time
        if (param->camera_enable_) {
            measures_.end_time_ = image_buffer_.front().timestamp_;
            measures_.img_ = image_buffer_.front().image_data_;
            image_buffer_.pop_front();
        } else
            measures_.end_time_ = measures_.end_time_ + param->scan_interval_;
    } else {
        // the measure is not synced, no need to set the lidar end time
        measures_.end_time_ = measures_.end_time_;
    }


    if (imu_buffer_.back().timestamp < measures_.end_time_)
        return false;
    if (points_buffer_.back().timestamp < measures_.end_time_) return false;

    // push the imu data
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < measures_.end_time_) {
        measures_.imu_.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    // push the lidar points
    measures_.lidar_->clear();
    while (!points_buffer_.empty() && points_buffer_.front().timestamp < measures_.end_time_) {
        measures_.lidar_->emplace_back(points_buffer_.front());
        points_buffer_.pop_front();
    }

    if (measures_.lidar_->empty() || measures_.imu_.empty()) {
        std::cout << "Empty lidar or imu data, skip this measure" << std::endl;
        return false;
    }
    return true;
}

}  // namespace faster_lio

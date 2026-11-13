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

#include <gflags/gflags.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <mutex>
#include <thread>

#include <boost/filesystem.hpp>
#include "laser_mapping.h"

namespace fs = boost::filesystem;

namespace {
// One message of the bag, read and decoded by the reader thread: the lidar ones already turned into their scan.
struct BagItem {
    enum Kind { kLidar, kImu, kImage, kCompressedImage, kEnd } kind = kEnd;
    kuma_lio::PointCloud::Ptr scan;  // kLidar: null when the message is older than the offset time
    sensor_msgs::Imu::ConstPtr imu;
    sensor_msgs::Image::ConstPtr image;
    sensor_msgs::CompressedImage::ConstPtr compressed;
};

// FIFO between the reader and the processing thread; the reader blocks once `capacity` items wait.
class BagQueue {
   public:
    explicit BagQueue(size_t capacity) : capacity_(capacity) {}
    void Push(BagItem item) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [this] { return items_.size() < capacity_; });
        items_.push_back(std::move(item));
        not_empty_.notify_one();
    }
    BagItem Pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this] { return !items_.empty(); });
        BagItem item = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return item;
    }

   private:
    const size_t capacity_;
    std::mutex mtx_;
    std::condition_variable not_full_, not_empty_;
    std::deque<BagItem> items_;
};
}  // namespace

DEFINE_string(config_file, "./config/avia.yaml", "path to config file");
DEFINE_string(bag_file, "", "path to the ros bag");
DEFINE_string(output_dir, "", "save the result to the dir");
DEFINE_double(start, 0.0, "start time in seconds from beginning of bag");
DEFINE_double(duration, -1.0, "duration in seconds, -1 means till end");
DEFINE_bool(dynamic_init, false, "try the dynamic (moving-start) initialization before the stationary one");
DEFINE_string(camera_fname, "", "camera yaml like config/i2nav_robot_camera.yaml: extrin_R, extrin_t (T_imu_cam) and camera (cameras.txt lines); empty = no camera");
DEFINE_string(gnss_obs_fname, "", "");
DEFINE_string(pgo_config, "", "config/pgo.yaml: runs the loop closure coupled to the fastlio front end (Voxel-SLAM); empty = no loop closure");
DEFINE_string(nav_fname, "", "");
std::atomic<bool> if_exit{false};  // set by the signal handler, read by the reader thread
void SigHandle(int sig) {
    if_exit = true;
    LOG(INFO) << "Catch Kill Signal";
}
int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const std::string bag_file = FLAGS_bag_file;
    const std::string config_file = FLAGS_config_file;
    const std::string output_dir = FLAGS_output_dir;
    const std::string camera_fname = FLAGS_camera_fname;
    const std::string gnss_obs_fname = FLAGS_gnss_obs_fname;
    const std::string nav_fname = FLAGS_nav_fname;
    fs::create_directories(output_dir);
    // glog
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);

    // ros
    ros::init(argc, argv, "faster_lio");
    ros::NodeHandle nh;

    // laser mapping
    auto laser_mapping = std::make_shared<kuma_lio::LaserMapping>();
    if (!laser_mapping->Init(config_file, camera_fname, FLAGS_pgo_config)) {
        LOG(ERROR) << "laser mapping init failed.";
        return -1;
    }
    laser_mapping->output_dir = output_dir;
    laser_mapping->dynamic_init_ = FLAGS_dynamic_init;
    laser_mapping->InitialPublishers(nh);

    // handle ctrl-c
    signal(SIGINT, SigHandle);

    // read the bag
    LOG(INFO) << "Opening rosbag, be patient";
    rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
    if (!bag.isOpen()) {
        LOG(ERROR) << "Failed to open bag";
        return -1;
    }
    rosbag::View full_view(bag);
    ros::Time bag_start = full_view.getBeginTime();
    ros::Time bag_end = full_view.getEndTime();
    ros::Time start_time = bag_start + ros::Duration(FLAGS_start);

    ros::Time end_time;
    if (FLAGS_duration < 0) {
        end_time = bag_end;
    } else {
        end_time = start_time + ros::Duration(FLAGS_duration);
        if (end_time > bag_end) {
            end_time = bag_end;
        }
    }
    LOG(INFO) << "Go!";
    // only the topics the loop below handles: the rest of the bag (event cameras, other images, INS status, ...) is
    // neither read nor tried against every message type
    std::vector<std::string> topics = {laser_mapping->param->lidar_topic_, laser_mapping->param->imu_topic_};
    if (!laser_mapping->param->img_topic_.empty()) topics.push_back(laser_mapping->param->img_topic_);
    rosbag::View view(bag, rosbag::TopicQuery(topics), start_time, end_time);
    for (const rosbag::MessageInstance &m : view) {
        auto imu_msg = m.instantiate<sensor_msgs::Imu>();
        if (m.getTopic() == laser_mapping->param->imu_topic_ && imu_msg) {
            double start_offset_time = imu_msg->header.stamp.toSec();
            CHECK(std::isnan(kuma_lio::global_offset_time));
            kuma_lio::global_offset_time = start_offset_time;
            LOG(INFO) << "Offset time: " << std::fixed << start_offset_time;
            break;
        }
    }

    if (!nav_fname.empty() && !gnss_obs_fname.empty()) {
        fs::path sol_fname = fs::path(output_dir) / "rtkpos.txt";
        laser_mapping->ReadObsNav(gnss_obs_fname, nav_fname, sol_fname.string());
    }
    const auto run_front_end = [&laser_mapping]() {
        if (laser_mapping->param->front_end == "fastlio") {
            laser_mapping->RunFastLio();
        } else {
            laser_mapping->RunRESPLE();
        }
    };
    // The reader thread reads and decodes the bag and builds the lidar scans (ScanFromPCL / ScanFromLivox depend on
    // the message and the parameters only); this thread takes the items in bag order and does everything that
    // touches the state, so the result is the one of a single loop whatever the scheduling.
    BagQueue queue(64);
    std::thread reader([&]() {
        const auto& prm = laser_mapping->param;
        for (const rosbag::MessageInstance &m : view) {
            if (if_exit) break;
            BagItem item;
            if (m.getTopic() == prm->lidar_topic_) {
                if (auto livox_msg = m.instantiate<livox_ros_driver::CustomMsg>()) {
                    item.kind = BagItem::kLidar;
                    item.scan = laser_mapping->ScanFromLivox(livox_msg);
                } else if (auto point_cloud_msg = m.instantiate<sensor_msgs::PointCloud2>()) {
                    item.kind = BagItem::kLidar;
                    item.scan = laser_mapping->ScanFromPCL(point_cloud_msg);
                }
            } else if (m.getTopic() == prm->imu_topic_) {
                if ((item.imu = m.instantiate<sensor_msgs::Imu>())) item.kind = BagItem::kImu;
            } else if (m.getTopic() == prm->img_topic_) {
                if ((item.compressed = m.instantiate<sensor_msgs::CompressedImage>())) {
                    item.kind = BagItem::kCompressedImage;
                } else if ((item.image = m.instantiate<sensor_msgs::Image>())) {
                    item.kind = BagItem::kImage;
                }
            }
            if (item.kind != BagItem::kEnd) queue.Push(std::move(item));
        }
        queue.Push(BagItem());  // kEnd
    });
    for (BagItem item = queue.Pop(); item.kind != BagItem::kEnd; item = queue.Pop()) {
        switch (item.kind) {
            case BagItem::kLidar:
                laser_mapping->PushScan(item.scan);
                run_front_end();
                break;
            case BagItem::kImu:
                laser_mapping->IMUCallBack(item.imu);
                break;
            case BagItem::kCompressedImage:
                laser_mapping->CompressedImageCallback(item.compressed);
                break;
            case BagItem::kImage:
                laser_mapping->ImageCallBack(item.image);
                break;
            default:
                break;
        }
    }
    reader.join();

    LOG(INFO) << "Finish bag iteration";
    laser_mapping->Finish();
    laser_mapping->Savetrajectory();

    return 0;
}

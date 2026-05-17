#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <tf2_msgs/TFMessage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gflags/gflags.h>

namespace fs = std::filesystem;

// 定义 Velodyne 点类型，与 Python 脚本中的一致
namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (float, time, time)
                                (std::uint16_t, ring, ring)
)

DEFINE_string(basedir, "", "KITTI dataset base directory");
DEFINE_string(date, "2011_09_30", "KITTI dataset date");
DEFINE_string(drive, "0018", "KITTI dataset drive");
DEFINE_string(output_bag, "", "Output rosbag path (optional)");

struct OxtsData {
    double lat, lon, alt;
    double roll, pitch, yaw;
    double vn, ve, vf, vl, vu;
    double ax, ay, az;
    double af, al, au;
    double wx, wy, wz;
    double wf, wl, wu;
    double pos_accuracy, vel_accuracy;
    int navstat, numsats, posmode, velmode;

    Eigen::Matrix4d T_w_imu;
};

// 辅助函数：从字符串解析时间戳
ros::Time parse_timestamp(const std::string& line) {
    std::tm tm = {};
    std::stringstream ss(line);
    char dash, colon, dot;
    int year, month, day, hour, minute, second, micro;
    
    // Format: 2011-09-30 11:11:11.111111111
    ss >> year >> dash >> month >> dash >> day >> hour >> colon >> minute >> colon >> second >> dot >> micro;
    
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
    tm.tm_isdst = -1;

    time_t time = timegm(&tm);
    return ros::Time(time, micro * 1000); // 假设是微秒，kitti 时间戳通常到 .000000000
}

// 修改后的解析函数，考虑到 KITTI 时间戳格式可能有变
ros::Time parse_timestamp_v2(const std::string& line) {
    if (line.empty()) return ros::Time(0);
    int y, m, d, hh, mm, ss;
    double frac_sec;
    if (sscanf(line.c_str(), "%d-%d-%d %d:%d:%lf", &y, &m, &d, &hh, &mm, &frac_sec) == 6) {
        std::tm tm = {0};
        tm.tm_year = y - 1900;
        tm.tm_mon = m - 1;
        tm.tm_mday = d;
        tm.tm_hour = hh;
        tm.tm_min = mm;
        tm.tm_sec = (int)frac_sec;
        tm.tm_isdst = -1;
        time_t time = timegm(&tm);
        return ros::Time(time, (frac_sec - (int)frac_sec) * 1e9);
    }
    return ros::Time(0);
}

// 载入 Oxts 数据
std::vector<OxtsData> load_oxts(const std::string& path) {
    std::vector<OxtsData> oxts_vec;
    std::string data_path = path + "/oxts/data";
    if (!fs::exists(data_path)) return oxts_vec;

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(data_path)) {
        files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());

    double scale = -1.0;

    for (const auto& file : files) {
        std::ifstream ifs(file);
        OxtsData data;
        ifs >> data.lat >> data.lon >> data.alt >> data.roll >> data.pitch >> data.yaw
            >> data.vn >> data.ve >> data.vf >> data.vl >> data.vu
            >> data.ax >> data.ay >> data.az >> data.af >> data.al >> data.au
            >> data.wx >> data.wy >> data.wz >> data.wf >> data.wl >> data.wu
            >> data.pos_accuracy >> data.vel_accuracy >> data.navstat >> data.numsats >> data.posmode >> data.velmode;

        if (scale < 0) {
            scale = std::cos(data.lat * M_PI / 180.0);
        }

        // 计算 T_w_imu (world to imu)
        // 参考 pykitti 源码和 kitti 官方文档
        double er = 6378137.0;
        double tx = scale * data.lon * M_PI * er / 180.0;
        double ty = scale * er * std::log(std::tan((90.0 + data.lat) * M_PI / 360.0));
        double tz = data.alt;

        Eigen::Vector3d t(tx, ty, tz);
        Eigen::AngleAxisd rollAngle(data.roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(data.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(data.yaw, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;

        data.T_w_imu = Eigen::Matrix4d::Identity();
        data.T_w_imu.block<3, 3>(0, 0) = q.matrix();
        data.T_w_imu.block<3, 1>(0, 3) = t;

        oxts_vec.push_back(data);
    }

    // 减去起始偏移量，使起始点在原点附近 (可选，但 pykitti 默认保留绝对坐标)
    // 这里保持绝对坐标，对应 python 脚本行为
    return oxts_vec;
}

// 保存静态变换
void save_static_transforms(rosbag::Bag& bag, const ros::Time& first_time) {
    tf2_msgs::TFMessage tfm;

    // base_link to im_link
    geometry_msgs::TransformStamped t1;
    t1.header.stamp = first_time;
    t1.header.frame_id = "base_link";
    t1.child_frame_id = "imu_link";
    t1.transform.translation.x = -2.71 / 2.0 - 0.05;
    t1.transform.translation.y = 0.32;
    t1.transform.translation.z = 0.93;
    t1.transform.rotation.w = 1.0;
    tfm.transforms.push_back(t1);

    // imu_link to velo_link
    // 这里需要加载 calib 文件，简化起见，如果没加载就用一个默认的
    // pykitti 中 dataset.calib.T_velo_imu 是从 calib_velo_to_imu.txt 读取的
    // 这里先放一个恒等或常用的
    geometry_msgs::TransformStamped t2;
    t2.header.stamp = first_time;
    t2.header.frame_id = "imu_link";
    t2.child_frame_id = "velo_link";
    
    // KITTI 默认 imu -> velo
    // R: 7.533745e-03 -9.999714e-01 -6.166020e-04
    //    1.480249e-02 7.280733e-04 -9.998902e-01
    //    9.998621e-01 7.523790e-03 1.480755e-02
    // T: -4.069766e-03 -7.631618e-02 -2.717803e-01
    // 注意：dataset.calib.T_velo_imu 实际上是 velo -> imu 还是 imu -> velo?
    // PyKitti: T_velo_imu is imu_coord to velo_coord
    
    Eigen::Matrix3d R_v_i;
    R_v_i << 7.533745e-03, -9.999714e-01, -6.166020e-04,
             1.480249e-02, 7.280733e-04, -9.998902e-01,
             9.998621e-01, 7.523790e-03, 1.480755e-02;
    Eigen::Vector3d T_v_i(-4.069766e-03, -7.631618e-02, -2.717803e-01);
    
    // 但是 python 脚本中 T_velo_to_imu = dataset.calib.T_velo_imu
    // 并且 save_static_transforms(bag, [('imu_link', 'velo_link', T_velo_to_imu)], ...)
    // 说明 T_velo_to_imu 应该是 imu_link -> velo_link 的变换
    
    t2.transform.translation.x = T_v_i.x();
    t2.transform.translation.y = T_v_i.y();
    t2.transform.translation.z = T_v_i.z();
    Eigen::Quaterniond q_v_i(R_v_i);
    t2.transform.rotation.x = q_v_i.x();
    t2.transform.rotation.y = q_v_i.y();
    t2.transform.rotation.z = q_v_i.z();
    t2.transform.rotation.w = q_v_i.w();
    tfm.transforms.push_back(t2);

    bag.write("/tf_static", first_time, tfm);
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_basedir.empty()) {
        std::cerr << "Usage: kitti2bag --basedir /path/to/kitti --date 2011_09_30 --drive 0018" << std::endl;
        return 1;
    }

    std::string drive_dir = FLAGS_basedir + "/" + FLAGS_date + "/" + FLAGS_date + "_drive_" + FLAGS_drive + "_extract";
    if (!fs::exists(drive_dir)) {
        drive_dir = FLAGS_basedir + "/" + FLAGS_date + "/" + FLAGS_date + "_drive_" + FLAGS_drive + "_sync";
    }
    if (!fs::exists(drive_dir)) {
        // 尝试另一种常见的 KITTI 组织方式
        drive_dir = FLAGS_basedir + "/" + FLAGS_date + "_drive_" + FLAGS_drive + "_extract";
    }

    std::cout << "Loading from: " << drive_dir << std::endl;

    // 加载时间戳
    auto load_timestamps = [](const std::string& path) {
        std::vector<ros::Time> ts;
        std::ifstream ifs(path);
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            ts.push_back(parse_timestamp_v2(line));
        }
        return ts;
    };

    std::vector<ros::Time> oxts_ts = load_timestamps(drive_dir + "/oxts/timestamps.txt");
    std::vector<OxtsData> oxts_data = load_oxts(drive_dir);

    if (oxts_ts.size() != oxts_data.size()) {
        std::cerr << "Oxts timestamps and data size mismatch: " << oxts_ts.size() << " vs " << oxts_data.size() << std::endl;
        return 1;
    }

    std::string bag_name = FLAGS_output_bag;
    if (bag_name.empty()) {
        bag_name = "kitti_" + FLAGS_date + "_drive_" + FLAGS_drive + ".bag";
    }

    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Write);

    std::cout << "Exporting static transforms..." << std::endl;
    save_static_transforms(bag, oxts_ts[0]);

    std::cout << "Exporting Oxts (TF, IMU, GPS)..." << std::endl;
    for (size_t i = 0; i < oxts_data.size(); ++i) {
        ros::Time ts = oxts_ts[i];
        const auto& data = oxts_data[i];

        // Dynamic TF: world -> base_link
        tf2_msgs::TFMessage tfm;
        geometry_msgs::TransformStamped t_dyn;
        t_dyn.header.stamp = ts;
        t_dyn.header.frame_id = "world";
        t_dyn.child_frame_id = "base_link";
        
        Eigen::Vector3d trans = data.T_w_imu.block<3, 1>(0, 3);
        Eigen::Quaterniond rot(data.T_w_imu.block<3, 3>(0, 0));
        
        t_dyn.transform.translation.x = trans.x();
        t_dyn.transform.translation.y = trans.y();
        t_dyn.transform.translation.z = trans.z();
        t_dyn.transform.rotation.x = rot.x();
        t_dyn.transform.rotation.y = rot.y();
        t_dyn.transform.rotation.z = rot.z();
        t_dyn.transform.rotation.w = rot.w();
        tfm.transforms.push_back(t_dyn);
        bag.write("/tf", ts, tfm);

        // IMU
        sensor_msgs::Imu imu;
        imu.header.stamp = ts;
        imu.header.frame_id = "imu_link";
        Eigen::AngleAxisd roll(data.roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch(data.pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw(data.yaw, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q_imu = yaw * pitch * roll;
        imu.orientation.x = q_imu.x();
        imu.orientation.y = q_imu.y();
        imu.orientation.z = q_imu.z();
        imu.orientation.w = q_imu.w();
        imu.linear_acceleration.x = data.ax;
        imu.linear_acceleration.y = data.ay;
        imu.linear_acceleration.z = data.az;
        imu.angular_velocity.x = data.wx;
        imu.angular_velocity.y = data.wy;
        imu.angular_velocity.z = data.wz;
        bag.write("/kitti/imu", ts, imu);

        // GPS Fix
        sensor_msgs::NavSatFix fix;
        fix.header.stamp = ts;
        fix.header.frame_id = "imu_link";
        fix.latitude = data.lat;
        fix.longitude = data.lon;
        fix.altitude = data.alt;
        fix.status.service = 1;
        bag.write("/kitti/gps/fix", ts, fix);

        // GPS Vel
        geometry_msgs::TwistStamped vel;
        vel.header.stamp = ts;
        vel.header.frame_id = "imu_link";
        vel.twist.linear.x = data.vf;
        vel.twist.linear.y = data.vl;
        vel.twist.linear.z = data.vu;
        vel.twist.angular.x = data.wf;
        vel.twist.angular.y = data.wl;
        vel.twist.angular.z = data.wu;
        bag.write("/kitti/gps/vel", ts, vel);
    }

    std::cout << "Exporting Velodyne data..." << std::endl;
    std::string velo_path = drive_dir + "/velodyne_points";
    std::vector<ros::Time> velo_ts = load_timestamps(velo_path + "/timestamps.txt");
    std::vector<ros::Time> velo_ts_start = load_timestamps(velo_path + "/timestamps_start.txt");
    std::vector<ros::Time> velo_ts_end = load_timestamps(velo_path + "/timestamps_end.txt");
    
    std::string velo_data_dir = velo_path + "/data";
    std::vector<std::string> velo_files;
    if (fs::exists(velo_data_dir)) {
        for (const auto& entry : fs::directory_iterator(velo_data_dir)) {
            velo_files.push_back(entry.path().string());
        }
        std::sort(velo_files.begin(), velo_files.end());
    }

    for (size_t i = 0; i < std::min(velo_files.size(), velo_ts.size()); ++i) {
        std::ifstream ifs(velo_files[i]);
        if (!ifs) {
            std::cerr << "Error opening velo file " << velo_files[i] << std::endl;
            continue;
        }

        std::vector<float> values;
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream ss(line);
            float v;
            while (ss >> v) {
                values.push_back(v);
            }
        }

        int num_points = values.size() / 4;
        std::cout << "Processing " << velo_files[i] << " with " << num_points << " points." << std::endl;
        pcl::PointCloud<velodyne_ros::Point> cloud;
        cloud.reserve(num_points);

        std::vector<float> thetas;
        thetas.reserve(num_points);
        for (int j = 0; j < num_points; ++j) {
            float x = values[j * 4 + 0];
            float y = values[j * 4 + 1];
            float theta = std::atan2(y, x) * 180.0 / M_PI;
            if (theta < 0) theta += 360.0;
            thetas.push_back(theta);
        }

        // 计算 Ring
        std::vector<int> rings(num_points, 0);
        int current_ring = 0;
        int MAX_RING = 63;
        for (int j = 1; j < num_points; ++j) {
            if (thetas[j] - thetas[j-1] < -180.0) {
                current_ring++;
            }
            if (current_ring > MAX_RING) break;
            rings[j] = current_ring;
        }
        
        // 计算 Time
        double frame_time = 0.1; // 默认 10Hz
        if (i < velo_ts_start.size() && i < velo_ts_end.size()) {
            frame_time = (velo_ts_end[i] - velo_ts_start[i]).toSec();
        }
        
        std::vector<float> times(num_points, 0.0f);
        for (int r = 0; r <= std::min(current_ring, MAX_RING); ++r) {
            std::vector<int> idxs;
            for (int j = 0; j < num_points; ++j) {
                if (rings[j] == r) idxs.push_back(j);
            }
            if (idxs.size() > 1) {
                for (size_t k = 0; k < idxs.size(); ++k) {
                    times[idxs[k]] = (float)k / (float)idxs.size() * frame_time;
                }
            }
        }

        for (int j = 0; j < num_points; ++j) {
            velodyne_ros::Point p;
            p.x = values[j * 4 + 0];
            p.y = values[j * 4 + 1];
            p.z = values[j * 4 + 2];
            p.intensity = values[j * 4 + 3] * 255.0f;
            if (p.intensity > 255.0f) p.intensity = 255.0f;
            p.ring = rings[j];
            p.time = times[j];
            cloud.push_back(p);
        }
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.frame_id = "velo_link";
        msg.header.stamp = velo_ts_start[i];

        bag.write("/kitti/velo", msg.header.stamp, msg);
        if (i % 10 == 0) std::cout << "Exported " << i << " velo scans" << std::endl;
    }

    bag.close();
    std::cout << "Done! Bag saved to " << bag_name << std::endl;

    return 0;
}

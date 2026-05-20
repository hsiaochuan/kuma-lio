#include <pcl/common/transforms.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "stamp_pose.h"
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using PointType = pcl::PointXYZI;
bool LoadPoints(const std::string& points_fname, pcl::PointCloud<PointType>& points) {
    std::string ext = fs::path(points_fname).extension().string();
    if (ext == ".pcd") {
        if (pcl::io::loadPCDFile(points_fname, points) < 0) {
            std::cerr << "Failed to load PCD file: " << points_fname << std::endl;
            return false;
        }
    } else if (ext == ".ply") {
        if (pcl::io::loadPLYFile(points_fname, points) < 0) {
            std::cerr << "Failed to load PLY file: " << points_fname << std::endl;
            return false;
        }
    } else if (ext == ".txt") {
        std::ifstream infile(points_fname);
        if (!infile.is_open()) {
            std::cerr << "Failed to open points file: " << points_fname << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            PointType point;
            if (!(iss >> point.x >> point.y >> point.z >> point.intensity)) {
                std::cerr << "Invalid line in points file: " << line << std::endl;
                continue;
            }
            points.push_back(point);
        }
    } else {
        std::cerr << "Unsupported points file format: " << ext << std::endl;
        return false;
    }
    return true;
}
int main(int argc, char** argv) {
    std::string trajectory_fname;
    std::string pcd_dir;
    std::string output_pcd;
    int start_idx = 0;
    int end_idx = -1;
    double voxel_size = 0.05;
    po::options_description desc("Jet smoothing options");
    // clang-format off
    desc.add_options()
        ("help,h", "Show help message")
        ("trajectory,t", po::value<std::string>(&trajectory_fname)->required(), "Trajectory file (TUM format)")
        ("pcd_dir,p", po::value<std::string>(&pcd_dir)->required(), "Directory containing PCD files")
        ("output,o", po::value<std::string>(&output_pcd)->required(), "Output merged PCD file")
        ("start_idx,s", po::value<int>(&start_idx)->default_value(0), "Start index for merging")
        ("end_idx,e", po::value<int>(&end_idx)->default_value(-1), "End index for merging (exclusive)")
        ("voxel_size,e", po::value<double>(&voxel_size)->default_value(-1), "Voxel size for downsampling (negative to disable)");
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::vector<std::string> points_fnames;
    boost::filesystem::path pcd_path(pcd_dir);
    for (boost::filesystem::directory_iterator it(pcd_path); it != boost::filesystem::directory_iterator(); ++it) {
        points_fnames.push_back(it->path().string());
    }
    std::sort(points_fnames.begin(), points_fnames.end());
    if (points_fnames.empty()) {
        std::cerr << "No PCD files found in " << pcd_path << std::endl;
        return 1;
    }

    if (end_idx == -1) end_idx = points_fnames.size();
    if (start_idx < 0 || end_idx > points_fnames.size() || start_idx >= end_idx) {
        std::cerr << "Invalid start/end indices: start=" << start_idx << " end=" << end_idx
                  << " total=" << points_fnames.size() << std::endl;
        return 1;
    }

    // load stamped poses
    faster_lio::Trajectory stamped_poses = faster_lio::TrajectoryGenerator::load_from_tumtxt(trajectory_fname);
    std::cout << "Loaded trajectory with " << stamped_poses.size() << " poses from " << trajectory_fname << std::endl;

    // if the poses and pcd is not match, try to extract the stamp
    bool try_use_pcd_stamp = false;
    faster_lio::TrajectoryInterpolator interpolator(stamped_poses);
    if (stamped_poses.size() != points_fnames.size()) {
        std::cout
            << "The number of poses and PCD files do not match. Trying to extract the stamp from the PCD file names."
            << std::endl;
        try {
            std::string pcd_fname = points_fnames[0];
            boost::filesystem::path path(pcd_fname);
            std::string pcd_stem = path.stem().string();
        } catch (std::exception& e) {
            std::cerr << "Failed to extract the stamp from the PCD file names: " << e.what() << std::endl;
            return 1;
        }
        std::cout << "Successfully extracted the stamp from the PCD file names." << std::endl;
        try_use_pcd_stamp = true;
    }

    // merge
    pcl::PointCloud<PointType>::Ptr merged_cloud(new pcl::PointCloud<PointType>);
    for (int i = start_idx; i < end_idx; ++i) {
        std::string points_fname = points_fnames[i];
        Eigen::Isometry3d pose;
        if (try_use_pcd_stamp) {
            double timestamp;
            try {
                boost::filesystem::path path(points_fname);
                std::string pcd_stem = path.stem().string();
                timestamp = std::stod(pcd_stem);
            } catch (std::exception& e) {
                std::cerr << "Failed to extract the stamp from the PCD file name: " << points_fname << std::endl;
                throw;
            }
            pose = interpolator.query(timestamp);
        } else {
            pose = stamped_poses[i].pose;
        }

        pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>);
        if (!LoadPoints(points_fname, *points)) {
            std::cerr << "Failed to load points from " << points_fname << std::endl;
            continue;
        }
        pcl::io::loadPCDFile(points_fname, *points);
        pcl::transformPointCloud(*points, *points, pose.matrix());
        *merged_cloud += *points;
    }

    if (voxel_size > 0) {
        pcl::UniformSampling<PointType> uniform_sampling;
        uniform_sampling.setInputCloud(merged_cloud);
        uniform_sampling.setRadiusSearch(voxel_size);
        pcl::PointCloud<PointType>::Ptr downsampled_cloud(new pcl::PointCloud<PointType>);
        uniform_sampling.filter(*downsampled_cloud);
    }
    std::cout << "Merged " << merged_cloud->points.size() << " points from " << start_idx << " to " << end_idx
              << std::endl;
    std::cout << "Saving " << merged_cloud->size() << " points to " << output_pcd << std::endl;
    pcl::io::savePCDFileBinary(output_pcd, *merged_cloud);
    return 0;
}
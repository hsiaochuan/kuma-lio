#include <pcl/io/pcd_io.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "global_optimizor.h"
#include "stamp_pose.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace faster_lio;
int main(int argc, char** argv) {
    std::string scans_dir;
    std::string stamp_poses_fname;
    std::string output_dir;
    std::string config_fname;
    int start_idx = 0;
    int end_idx = -1;
    po::options_description desc("Allowed options");
    // clang-format off
  desc.add_options()
      ("help,h", "Print help message")
      ("scans_dir", po::value<std::string>(&scans_dir)->required(), "Input file")
      ("stamp_poses_fname", po::value<std::string>(&stamp_poses_fname), "Input file")
      ("config_fname", po::value<std::string>(&config_fname), "Input file")
      ("output_dir", po::value<std::string>(&output_dir), "Input file")
      ("start_idx", po::value<int>(&start_idx), "Input file")
      ("end_idx", po::value<int>(&end_idx), "Input file")
      ;
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    GlobalOptimizor::Options options;
    options.LoadFromYaml(config_fname);
    GlobalOptimizor mapper;
    fs::create_directories(output_dir);
    mapper.options_ = options;
    mapper.output_dir = output_dir;

    // scans
    std::vector<std::string> pc_fnames;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(scans_dir)) {
        if (entry.path().extension().string() == ".pcd") pc_fnames.push_back(entry.path().string());
    }
    std::sort(pc_fnames.begin(), pc_fnames.end());
    std::vector<ScanFrame::Ptr> scans(pc_fnames.size());
    for (int i = 0; i < pc_fnames.size(); ++i) {
        scans[i] = std::make_shared<ScanFrame>(i);
        scans[i]->cloud_fname = pc_fnames[i];
        scans[i]->timestamp = scans[i]->TryGetTimeFromName();
        scans[i]->GetScan();
    }
    if (end_idx < 0) end_idx = pc_fnames.size();

    // stamp poses
    Trajectory lidar_stamp_poses = TrajectoryGenerator::load_from_tumtxt(stamp_poses_fname);
    if (lidar_stamp_poses.size() != pc_fnames.size()) {
        TrajectoryInterpolator interpolator(lidar_stamp_poses);
        std::cout
            << "The number of poses and PCD files do not match. Trying to extract the stamp from the PCD file names."
            << std::endl;
        for (int i = 0; i < pc_fnames.size(); ++i) {
            Eigen::Isometry3d world_from_body = interpolator.query(scans[i]->GetTimestamp());
            scans[i]->world_from_body = Pose3(world_from_body);
        }
    } else {
        std::cout << "The number of poses and PCD files match." << std::endl;
        for (int i = 0; i < pc_fnames.size(); ++i) {
            scans[i]->world_from_body = Pose3(lidar_stamp_poses[i].pose);
        }
    }

    std::cout << "Loaded " << scans.size() << " scans." << std::endl;
    for (int i = 0; i < scans.size(); ++i) {
        mapper.AddScan(scans[i]);
    }
    mapper.ScanFilter();
    boost::filesystem::create_directories(output_dir + "/global/");
    TrajectoryGenerator::save_to_tumtxt(mapper.ExportStampedPoses(), output_dir + "/global/init.txt");
    mapper.ExportMap(output_dir + "/global/init.pcd");
    auto loops = mapper.DetectLoopClosure();
    mapper.SaveLoopToPcd(output_dir + "/global/loops.pcd");
    if (!loops.empty()) {
        mapper.PoseGraphOptimize();
        TrajectoryGenerator::save_to_tumtxt(mapper.ExportStampedPoses(), output_dir + "/global/pgo.txt");
        mapper.ExportMap(output_dir + "/global/pgo.pcd");
    }
    if (options.ba_enable) {
        for (int i = 0; i < options.ba_iters; ++i) {
            mapper.BundleAdjustment();
            mapper.ExportMap(output_dir + "/global/ba_" + std::to_string(i) + ".pcd");
            TrajectoryGenerator::save_to_tumtxt(mapper.ExportStampedPoses(), output_dir + "/global/ba_" + std::to_string(i) + ".txt");
        }
    }

    mapper.ExportMap(output_dir + "/final.pcd");
    TrajectoryGenerator::save_to_tumtxt(mapper.ExportStampedPoses(), output_dir + "/final.txt");
    return 0;
}
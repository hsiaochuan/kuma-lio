#include <boost/program_options.hpp>

#include "reconstruction.h"
#include "stamp_pose.h"
namespace po = boost::program_options;
int main(int argc, char** argv) {
    std::string lidar_trajectory_fname;
    std::string cam_trajectory_fname;
    std::string config_fname;
    std::string databse_fname;
    po::options_description desc("Jet smoothing options");
    desc.add_options()
        ("help,h", "Show help message")
        ("lidar_trajectory,l", po::value<std::string>(&lidar_trajectory_fname)->required(), "LiDAR trajectory file (TUM format)")
        ("cam_trajectory,c", po::value<std::string>(&cam_trajectory_fname)->required(), "Camera trajectory file (TUM format)")
        ("config,f", po::value<std::string>(&config_fname), "Configuration file (YAML format)")
        ("database,d", po::value<std::string>(&databse_fname)->required(), "Colmap database file")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    auto lidar_stamped_poses = faster_lio::TrajectoryGenerator::load_from_tumtxt(lidar_trajectory_fname);
    auto cam_stamped_poses = faster_lio::TrajectoryGenerator::load_from_tumtxt(cam_trajectory_fname);
    faster_lio::TrajectoryInterpolator cam_interpolator(cam_stamped_poses);

    Reconstruction recon;
    recon.LoadFromDatabase(databse_fname);

    return 0;
}
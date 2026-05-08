#pragma once
#include <Eigen/Eigen>
#include <string>
#include <vector>
namespace faster_lio {
struct StampedPose {
    double time;
    Eigen::Isometry3d pose;  ///< T_world_body
    StampedPose();
    StampedPose(double t, const Eigen::Isometry3d &p);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using Trajectory = std::vector<StampedPose, Eigen::aligned_allocator<StampedPose> >;

// ============================================================================
// Trajectory generator
// ============================================================================

class TrajectoryGenerator {
   public:
    /**
     * Linear motion with constant velocity
     * @param start_pos  start position
     * @param velocity   velocity vector [m/s]
     * @param duration   total duration [s]
     * @param dt         sampling time step [s]
     */
    static Trajectory linear(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &velocity, double duration,
                             double dt = 0.01);

    /**
     * Circular motion with constant speed (horizontal plane)
     * @param center      circle center (x, y)
     * @param radius      radius [m]
     * @param linear_vel  linear velocity [m/s]
     * @param height      motion height z [m]
     * @param duration    total duration [s]
     */
    static Trajectory circular(const Eigen::Vector2d &center, double radius, double linear_vel, double height,
                               double duration, double dt = 0.01);

    static Trajectory load_from_tumtxt(const std::string &filename);
    static void save_to_tumtxt(const Trajectory &traj, const std::string &filename);
    static void save_to_pcd(const Trajectory &traj, const std::string &filename, double axis_length = 0.1,
                            int points_per_axis = 20);
};

/** SE(3) pose interpolation: linear translation + SLERP rotation */
Eigen::Isometry3d interpolate_pose(const Eigen::Isometry3d &T0, const Eigen::Isometry3d &T1, double t);
class TrajectoryInterpolator {
   public:
    explicit TrajectoryInterpolator(const Trajectory &traj);

    double t_start() const;
    double t_end() const;

    Eigen::Isometry3d query(double t) const;

    const Trajectory &trajectory() const;

   private:
    Trajectory traj_;
};
}  // namespace faster_lio

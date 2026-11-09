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

#pragma once

#include <Eigen/Eigen>
#include <optional>
#include <string>
#include <vector>
namespace kuma_lio {
struct StampedPose {
    double time;
    Eigen::Isometry3d pose;  // T_world_body
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

    // Pose at `t`, interpolated between the two poses around it (clamped to the ends outside the
    // trajectory); nullopt when the pose stamp nearest to `t` is more than `thr` [s] away.
    std::optional<Eigen::Isometry3d> query(double t, double thr) const;

    const Trajectory &trajectory() const;

   private:
    Trajectory traj_;
};
}  // namespace kuma_lio

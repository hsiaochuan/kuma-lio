#pragma once
#include <Eigen/Eigen>
namespace faster_lio {
struct StampedPose {
    double time;
    Eigen::Isometry3d pose;  ///< T_world_body
    StampedPose() : time(0), pose(Eigen::Isometry3d::Identity()) {}
    StampedPose(double t, const Eigen::Isometry3d &p) : time(t), pose(p) {}
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
                             double dt = 0.01) {
        Trajectory traj;
        Eigen::Vector3d pos = start_pos;
        for (double t = 0; t <= duration + 1e-9; t += dt) {
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            T.translation() = pos;
            traj.push_back({t, T});
            pos += velocity * dt;
        }
        return traj;
    }

    /**
     * Circular motion with constant speed (horizontal plane)
     * @param center      circle center (x, y)
     * @param radius      radius [m]
     * @param linear_vel  linear velocity [m/s]
     * @param height      motion height z [m]
     * @param duration    total duration [s]
     */
    static Trajectory circular(const Eigen::Vector2d &center, double radius, double linear_vel, double height,
                               double duration, double dt = 0.01) {
        Trajectory traj;
        double w = linear_vel / radius;  // angular velocity
        double theta = 0.0;
        for (double t = 0; t <= duration + 1e-9; t += dt) {
            Eigen::Vector3d pos(center.x() + radius * std::cos(theta), center.y() + radius * std::sin(theta), height);
            double yaw = theta + M_PI / 2.0;  // tangent direction
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            T.translation() = pos;
            T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            traj.push_back({t, T});
            theta += w * dt;
        }
        return traj;
    }

    static Trajectory load_from_tumtxt(const std::string &filename) {
        Trajectory traj;
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            throw std::runtime_error("failed to open trajectory file: " + filename);
        }
        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;  // skip empty lines and comments
            std::istringstream iss(line);
            double t, x, y, z, qx, qy, qz, qw;
            if (!(iss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) {
                throw std::runtime_error("invalid line format in trajectory file: " + line);
            }
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            T.translation() = Eigen::Vector3d(x, y, z);
            T.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            traj.push_back({t, T});
        }
        return traj;
    };
}

/** SE(3) pose interpolation: linear translation + SLERP rotation */
static Eigen::Isometry3d
interpolate_pose(const Eigen::Isometry3d &T0, const Eigen::Isometry3d &T1, double t) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = T0.translation() + t * (T1.translation() - T0.translation());
    T.linear() = Eigen::Quaterniond(T0.linear()).slerp(t, Eigen::Quaterniond(T1.linear())).toRotationMatrix();
    return T;
}
class TrajectoryInterpolator {
   public:
    explicit TrajectoryInterpolator(const Trajectory &traj) : traj_(traj) {
        if (traj_.empty()) throw std::invalid_argument("trajectory is empty");
    }

    double t_start() const { return traj_.front().time; }
    double t_end() const { return traj_.back().time; }

    Eigen::Isometry3d query(double t) const {
        if (t <= traj_.front().time) return traj_.front().pose;
        if (t >= traj_.back().time) return traj_.back().pose;

        // Binary search for adjacent frames
        std::size_t lo = 0, hi = traj_.size() - 1;
        while (hi - lo > 1) {
            std::size_t mid = (lo + hi) / 2;
            (traj_[mid].time <= t ? lo : hi) = mid;
        }
        double span = traj_[hi].time - traj_[lo].time;
        double frac = (span > 1e-12) ? (t - traj_[lo].time) / span : 0.0;
        return interpolate_pose(traj_[lo].pose, traj_[hi].pose, frac);
    }

    const Trajectory &trajectory() const { return traj_; }

   private:
    Trajectory traj_;
};
}  // namespace faster_lio

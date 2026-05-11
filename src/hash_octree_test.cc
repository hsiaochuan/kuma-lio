#include "hash_octree.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>
using namespace faster_lio;
namespace {

auto MakeCov(double sigma) {
    return (sigma * sigma) * Eigen::Matrix3d::Identity();
}

std::vector<PointWithCov> MakePlanePoints(double x_min, double x_max,
                                          double y_min, double y_max,
                                          double step, double z,
                                          double sigma) {
    std::vector<PointWithCov> points;
    for (double x = x_min; x <= x_max + 1e-9; x += step) {
        for (double y = y_min; y <= y_max + 1e-9; y += step) {
            PointWithCov pv;
            pv.point = Eigen::Vector3d(x, y, z);
            pv.cov = MakeCov(sigma);
            points.push_back(pv);
        }
    }
    return points;
}

void ExpectPlaneLikeZ(const PlaneMatch &match, double abs_dot_min, double abs_dist_max) {
    const double abs_dot = std::fabs(match.normal.dot(Eigen::Vector3d::UnitZ()));
    EXPECT_GE(abs_dot, abs_dot_min);
    EXPECT_LE(std::fabs(match.dis_to_plane), abs_dist_max);
}

}  // namespace

TEST(HashOctree, AddPointsAndMatchPlane) {
    HashOctreeConfig config;
    config.voxel_size = 1.0;
    config.max_layer = 1;
    config.layer_init_num = {5, 5};
    config.max_points_num = 200;
    config.planer_threshold = 1e-4;
    config.sigma_num = 3.0;

    HashOctree map(config);
    const auto map_points = MakePlanePoints(0.1, 0.9, 0.1, 0.9, 0.2, 0.0, 1e-3);
    map.AddPoints(map_points);

    const auto query_points = MakePlanePoints(0.2, 0.8, 0.2, 0.8, 0.3, 0.0, 1e-3);
    const auto matches = map.Match(query_points);

    EXPECT_EQ(matches.size(), query_points.size());
    for (const auto &match : matches) {
        ExpectPlaneLikeZ(match, 0.9, 5e-3);
    }
}

TEST(HashOctree, NoMatchWhenFarFromPlane) {
    HashOctreeConfig config;
    config.voxel_size = 1.0;
    config.max_layer = 1;
    config.layer_init_num = {5, 5};
    config.max_points_num = 200;
    config.planer_threshold = 1e-4;
    config.sigma_num = 0.5;

    HashOctree map(config);
    const auto map_points = MakePlanePoints(-0.4, 0.4, -0.4, 0.4, 0.2, 0.0, 1e-4);
    map.AddPoints(map_points);

    std::vector<PointWithCov> query_points;
    PointWithCov pv;
    pv.point = Eigen::Vector3d(0.0, 0.0, 0.3);
    pv.cov = MakeCov(1e-4);
    query_points.push_back(pv);

    const auto matches = map.Match(query_points);
    EXPECT_TRUE(matches.empty());
}

TEST(HashOctree, NeighborVoxelFallback) {
    HashOctreeConfig config;
    config.voxel_size = 1.0;
    config.max_layer = 1;
    config.layer_init_num = {5, 5};
    config.max_points_num = 200;
    config.planer_threshold = 1e-4;
    config.sigma_num = 3.0;

    HashOctree map(config);
    std::vector<PointWithCov> seed_points;
    PointWithCov seed;
    seed.point = Eigen::Vector3d(0.1, 0.1, 0.5);
    seed.cov = MakeCov(1e-3);
    seed_points.push_back(seed);
    map.AddPoints(seed_points);

    const auto plane_points = MakePlanePoints(1.1, 1.7, 0.1, 0.7, 0.2, 0.5, 1e-3);
    map.AddPoints(plane_points);

    std::vector<PointWithCov> query_points;
    PointWithCov pv;
    pv.point = Eigen::Vector3d(0.9, 0.4, 0.5);
    pv.cov = MakeCov(1e-3);
    query_points.push_back(pv);

    const auto matches = map.Match(query_points);
    ASSERT_EQ(matches.size(), 1u);
    ExpectPlaneLikeZ(matches.front(), 0.9, 5e-3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

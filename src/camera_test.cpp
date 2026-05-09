#include "cameras/pinhole_camera.h"
#include "cameras/pinhole_fisheye_camera.h"
#include "cameras/spherial_camera.h"
#include "cameras/pinhole_radial.h"
#include "gtest/gtest.h"
#include <cmath>

using namespace faster_lio;

namespace {
constexpr double kTol = 1e-6;
constexpr double kTolFisheye = 1e-5;

void ExpectVec2Near(const Vec2 &a, const Vec2 &b, double tol) {
    EXPECT_NEAR(a.x(), b.x(), tol);
    EXPECT_NEAR(a.y(), b.y(), tol);
}
}  // namespace

TEST(PinholeFisheyeCamera, fisheye_test) {
    PinholeFisheyeCamera cam(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, -0.005, 0.0005, -0.0001);

    const Vec2 p(0.1, -0.05);
    const Vec2 p_dist = cam.add_disto(p);
    const Vec2 p_undist = cam.remove_disto(p_dist);
    ExpectVec2Near(p_undist, p, kTolFisheye);

    const Vec2 pixel_dist = cam.cam2ima(p_dist);
    const Vec2 pixel_ud = cam.get_ud_pixel(pixel_dist);
    ExpectVec2Near(pixel_ud, cam.cam2ima(p), kTolFisheye);

    const Vec3 X(1.2, -0.4, 2.0);
    const Vec2 proj = cam.project(X);
    const Vec2 expected = cam.cam2ima(cam.add_disto(X.hnormalized()));
    ExpectVec2Near(proj, expected, kTolFisheye);
}

TEST(PinholeRadialCamera, pinhole_radial_test) {
    PinholeRadialCamera cam(640, 480, 520.0, 510.0, 320.0, 240.0, 0.01, -0.005, 0.0008, 0.0003, -0.0002);

    const Vec2 p(0.12, 0.08);
    const Vec2 p_dist = cam.add_disto(p);
    const Vec2 p_undist = cam.remove_disto(p_dist);
    ExpectVec2Near(p_undist, p, kTol);

    const Vec2 pixel_dist = cam.cam2ima(p_dist);
    const Vec2 pixel_ud = cam.get_ud_pixel(pixel_dist);
    ExpectVec2Near(pixel_ud, cam.cam2ima(p), kTol);

    const Vec3 X(0.8, 0.3, 1.6);
    const Vec2 proj = cam.project(X);
    const Vec2 expected = cam.cam2ima(cam.add_disto(X.hnormalized()));
    ExpectVec2Near(proj, expected, kTol);
}

TEST(PinholeCamera, pinhole_test) {
    PinholeCamera cam(640, 480, 500.0, 520.0, 320.0, 240.0);

    const Vec2 p(0.1, -0.2);
    ExpectVec2Near(cam.remove_disto(cam.add_disto(p)), p, kTol);

    const Vec2 pixel = cam.cam2ima(p);
    const Vec2 cam_p = cam.ima2cam(pixel);
    ExpectVec2Near(cam_p, p, kTol);

    const Vec3 X(0.5, -0.25, 2.0);
    const Vec2 proj = cam.project(X);
    const Vec2 expected = cam.cam2ima(X.hnormalized());
    ExpectVec2Near(proj, expected, kTol);
}

TEST(SphericalCamera, spherial_test) {
    SphericalCamera cam(800, 400);

    const Vec3 X(1.0, 0.0, 1.0);
    const double lon = std::atan2(X.x(), X.z());
    const double lat = std::atan2(-X.y(), std::hypot(X.x(), X.z()));
    const Vec2 expected = cam.cam2ima(Vec2(lon / (2.0 * M_PI), -lat / (2.0 * M_PI)));
    const Vec2 proj = cam.project(X);
    ExpectVec2Near(proj, expected, kTol);

    const Vec2 uv(123.0, 210.0);
    const Vec2 cam_p = cam.ima2cam(cam.cam2ima(cam.ima2cam(uv)));
    ExpectVec2Near(cam_p, cam.ima2cam(uv), kTol);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "cameras/pinhole_camera.h"
#include "cameras/pinhole_fisheye_camera.h"
#include "cameras/spherial_camera.h"
#include "cameras/pinhole_radial.h"
#include "gtest/gtest.h"
using namespace faster_lio;
TEST(PinholeFisheyeCamera, fisheye_test) {
    const PinholeFisheyeCamera cam(1000, 1000,
                                   1000, 1000, 500, 500,
                                   -0.054, 0.014, 0.006, 0.011); // K1, K2, K3, K4
}

TEST(PinholeRadialCamera, pinhole_radial_test) {
    const PinholeRadialCamera cam(
        1000, 1000,
        1000, 1000, 500, 500,
        -0.054, 0.014, 0.006, 0.001, -0.001);
}

TEST(PinholeCamera, pinhole_test) {
    const PinholeCamera cam(
        1000, 1000,
        1000, 1000, 500, 500);
}

TEST(SphericalCamera, spherial_test) {
    const SphericalCamera cam(2000, 1000);
}

int main() {
    RUN_ALL_TESTS();
}

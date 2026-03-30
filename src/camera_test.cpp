#include "cameras/pinhole_camera.h"
#include "cameras/pinhole_radial_camera.h"
#include "cameras/pinhole_brown_camera.h"
#include "cameras/pinhole_fisheye_camera.h"
#include "cameras/spherial_camera.h"
#include "gtest/gtest.h"
using namespace faster_lio;
TEST(Cameras_Fisheye, disto_undisto_Fisheye) {

    const Pinhole_Intrinsic_Fisheye cam(1000, 1000, 1000, 500, 500,
                                        -0.054, 0.014, 0.006, 0.011); // K1, K2, K3, K4

}
TEST(Cameras_Brown, disto_undisto_T2) {

    const Pinhole_Intrinsic_Brown_T2 cam(1000, 1000, 1000, 500, 500,
      // K1, K2, K3, T1, T2
      -0.054, 0.014, 0.006, 0.001, -0.001);
}
TEST(Cameras_Radial, disto_undisto_K3) {

    const Pinhole_Intrinsic_Radial_K3 cam(1000, 1000, 1000, 500, 500,
      // K1, K2, K3
      -0.245539, 0.255195, 0.163773);
}
TEST(Cameras_Spherical, disto_undisto) {

    const Intrinsic_Spherical cam(2000, 1000);
}

int main() {
    RUN_ALL_TESTS();
}
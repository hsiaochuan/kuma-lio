//
// Created by hsiaochuan on 2026/09/05.
//

#ifndef CAMERA_IO_H
#define CAMERA_IO_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pinhole_camera.h"
#include "pinhole_fisheye_camera.h"
#include "pinhole_radial.h"

namespace faster_lio {

/**
 * @brief Reads the first successfully-parsed camera from a COLMAP-style
 * cameras.txt stream (format: "CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]", one
 * camera per line, '#'-prefixed / empty lines are comments and are skipped).
 *
 * Supported COLMAP models -> faster_lio types (params in COLMAP's documented
 * order, see colmap/src/colmap/sensor/models.h):
 *   SIMPLE_PINHOLE, PINHOLE       -> PinholeCamera
 *   SIMPLE_RADIAL, RADIAL, OPENCV -> PinholeRadialCamera
 *   OPENCV_FISHEYE                -> PinholeFisheyeCamera
 * Any other model is logged as a warning and skipped. Returns nullptr if no
 * line parses into a supported camera.
 */
inline CamModel::Ptr ReadCameraFromCamerasText(std::istream& stream) {
    std::string line;
    while (std::getline(stream, line)) {
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') {
            continue;
        }

        std::istringstream line_stream(line);
        uint32_t cam_id = 0;
        std::string model_name;
        uint32_t width = 0;
        uint32_t height = 0;
        if (!(line_stream >> cam_id >> model_name >> width >> height)) {
            continue;
        }

        std::vector<double> params;
        double param = 0.0;
        while (line_stream >> param) {
            params.push_back(param);
        }

        CamModel::Ptr cam = nullptr;
        if (model_name == "SIMPLE_PINHOLE") {
            // f, cx, cy
            if (params.size() >= 3) {
                const double f = params[0], cx = params[1], cy = params[2];
                cam = std::make_shared<PinholeCamera>(width, height, f, f, cx, cy);
            }
        } else if (model_name == "PINHOLE") {
            // fx, fy, cx, cy
            if (params.size() >= 4) {
                const double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                cam = std::make_shared<PinholeCamera>(width, height, fx, fy, cx, cy);
            }
        } else if (model_name == "SIMPLE_RADIAL") {
            // f, cx, cy, k
            if (params.size() >= 4) {
                const double f = params[0], cx = params[1], cy = params[2], k = params[3];
                cam = std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k, 0.0, 0.0, 0.0, 0.0);
            }
        } else if (model_name == "RADIAL") {
            // f, cx, cy, k1, k2
            if (params.size() >= 5) {
                const double f = params[0], cx = params[1], cy = params[2];
                const double k1 = params[3], k2 = params[4];
                cam = std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k1, k2, 0.0, 0.0, 0.0);
            }
        } else if (model_name == "OPENCV") {
            // fx, fy, cx, cy, k1, k2, p1, p2
            if (params.size() >= 8) {
                const double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                const double k1 = params[4], k2 = params[5], p1 = params[6], p2 = params[7];
                cam = std::make_shared<PinholeRadialCamera>(width, height, fx, fy, cx, cy, k1, k2, 0.0, p1, p2);
            }
        } else if (model_name == "OPENCV_FISHEYE") {
            // fx, fy, cx, cy, k1, k2, k3, k4
            if (params.size() >= 8) {
                const double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                const double k1 = params[4], k2 = params[5], k3 = params[6], k4 = params[7];
                cam = std::make_shared<PinholeFisheyeCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, k4);
            }
        } else {
            LOG(WARNING) << "Unsupported camera model in cameras.txt: " << model_name;
            continue;
        }

        if (cam) {
            return cam;
        }
        LOG(WARNING) << "Camera " << cam_id << " (" << model_name << ") has too few parameters, skipping";
    }
    return nullptr;
}

/**
 * @brief Same as ReadCameraFromCamerasText(std::istream&), but reads directly
 * from a cameras.txt file path.
 */
inline CamModel::Ptr ReadCameraFromCamerasText(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open cameras.txt: " << path;
        return nullptr;
    }
    return ReadCameraFromCamerasText(file);
}

}  // namespace faster_lio

#endif  // CAMERA_IO_H

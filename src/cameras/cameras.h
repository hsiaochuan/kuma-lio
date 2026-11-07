//
// Every camera model of COLMAP (src/colmap/sensor/models), one class per model deriving directly from CamModel,
// and the COLMAP cameras.txt file that the program takes as its camera parameter file: read and write.
//

#ifndef CAMERAS_CAMERAS_H
#define CAMERAS_CAMERAS_H

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "camera_base.h"
#include "division.h"
#include "eucm.h"
#include "fisheye.h"
#include "fisheye_radial.h"
#include "fov.h"
#include "opencv.h"
#include "pinhole.h"
#include "radial.h"
#include "spherical.h"
#include "thin_prism.h"

namespace faster_lio {

namespace camera_internal {
/// The camera if `name` is Model's COLMAP name; *matched tells whether it was, so that a wrong parameter count
/// (nullptr with *matched) can be told apart from another model.
template <typename Model>
CamModel::Ptr MakeIfNamed(const std::string& name, unsigned int w, unsigned int h, const std::vector<double>& params,
                          bool* matched) {
    if (*matched || name != Model::kModelName) return nullptr;
    *matched = true;
    if (params.size() != Model::kNumParams) {
        LOG(WARNING) << name << " takes " << Model::kNumParams << " parameters (" << Model::kParamsInfo << "), got "
                     << params.size();
        return nullptr;
    }
    return std::make_shared<Model>(w, h, params);
}
}  // namespace camera_internal

/**
 * @brief The camera of a COLMAP model name (cameras.txt), with its parameters in COLMAP's order.
 * nullptr if the name is unknown or the parameter count is wrong.
 */
inline CamModel::Ptr CreateCamera(const std::string& model_name, unsigned int width, unsigned int height,
                                  const std::vector<double>& params) {
    using camera_internal::MakeIfNamed;
    bool matched = false;
    CamModel::Ptr cam;
    // one line per model, in COLMAP's CameraModelId order
    if (!cam) cam = MakeIfNamed<SimplePinholeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<PinholeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<SimpleRadialCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<RadialCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<OpenCVCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<OpenCVFisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<FullOpenCVCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<FOVCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<SimpleRadialFisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<RadialFisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<ThinPrismFisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<RadTanThinPrismFisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<SimpleDivisionCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<DivisionCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<SimpleFisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<FisheyeCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<EUCMCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<EquirectangularCamera>(model_name, width, height, params, &matched);
    if (!cam) cam = MakeIfNamed<SkewedPinholeCamera>(model_name, width, height, params, &matched);
    if (!matched) LOG(WARNING) << "unknown camera model: " << model_name;
    return cam;
}

/**
 * @brief Every camera of a COLMAP cameras.txt stream ("CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]" per line;
 * '#'-prefixed and empty lines are skipped), by CAMERA_ID.  Every COLMAP model is supported, see CreateCamera;
 * a line that gives no camera, or repeats a CAMERA_ID, is warned about and skipped.
 */
inline std::map<uint32_t, CamModel::Ptr> ReadCamerasFromCamerasText(std::istream& stream) {
    std::map<uint32_t, CamModel::Ptr> cameras;
    std::string line;
    while (std::getline(stream, line)) {
        const size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;

        std::istringstream line_stream(line);
        uint32_t cam_id = 0;
        std::string model_name;
        uint32_t width = 0, height = 0;
        if (!(line_stream >> cam_id >> model_name >> width >> height)) continue;
        std::vector<double> params;
        double param = 0.0;
        while (line_stream >> param) params.push_back(param);

        CamModel::Ptr cam = CreateCamera(model_name, width, height, params);
        if (!cam) {
            LOG(WARNING) << "camera " << cam_id << " (" << model_name << ") skipped";
        } else if (!cameras.emplace(cam_id, cam).second) {
            LOG(WARNING) << "camera " << cam_id << " appears twice, the second one skipped";
        }
    }
    return cameras;
}

/// Same as ReadCamerasFromCamerasText(std::istream&), from a cameras.txt file; empty if it cannot be opened.
inline std::map<uint32_t, CamModel::Ptr> ReadCamerasFromCamerasText(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open cameras.txt: " << path;
        return {};
    }
    return ReadCamerasFromCamerasText(file);
}

/// The camera with the smallest CAMERA_ID of a cameras.txt stream, nullptr if there is none.
inline CamModel::Ptr ReadCameraFromCamerasText(std::istream& stream) {
    const std::map<uint32_t, CamModel::Ptr> cameras = ReadCamerasFromCamerasText(stream);
    return cameras.empty() ? nullptr : cameras.begin()->second;
}

/// The camera with the smallest CAMERA_ID of a cameras.txt file, nullptr if there is none.
inline CamModel::Ptr ReadCameraFromCamerasText(const std::string& path) {
    const std::map<uint32_t, CamModel::Ptr> cameras = ReadCamerasFromCamerasText(path);
    return cameras.empty() ? nullptr : cameras.begin()->second;
}

/**
 * @brief Writes `cameras` (by CAMERA_ID) as a COLMAP cameras.txt: the file ReadCamerasFromCamerasText reads back,
 * and that colmap (e.g. point_triangulator) takes.  False if it cannot be written.
 */
inline bool WriteCamerasToCamerasText(const std::string& path, const std::map<uint32_t, CamModel::Ptr>& cameras) {
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open cameras.txt for writing: " << path;
        return false;
    }
    file << std::setprecision(17);
    file << "# Camera list with one line of data per camera:\n"
         << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
         << "# Number of cameras: " << cameras.size() << "\n";
    for (const auto& camera_it : cameras) {
        const CamModel& cam = *camera_it.second;
        file << camera_it.first << ' ' << cam.model_name() << ' ' << cam.w() << ' ' << cam.h();
        for (double p : cam.params()) file << ' ' << p;
        file << '\n';
    }
    return file.good();
}

/// Writes one camera, CAMERA_ID `camera_id`, as a COLMAP cameras.txt.
inline bool WriteCameraToCamerasText(const std::string& path, const CamModel::Ptr& cam, uint32_t camera_id = 1) {
    return WriteCamerasToCamerasText(path, {{camera_id, cam}});
}

}  // namespace faster_lio

#endif  // CAMERAS_CAMERAS_H

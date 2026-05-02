#pragma once
#include <sqlite3.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>
#include "cameras/cameras.h"
#include "pose3.h"
#include "types.h"
using namespace faster_lio;
namespace fs = boost::filesystem;

inline int SQLite3CallHelper(int result_code, const std::string& filename, int line) {
    switch (result_code) {
        case SQLITE_OK:
        case SQLITE_ROW:
        case SQLITE_DONE:
            return result_code;
        default:
            return result_code;
    }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                                                        \
    {                                                                                                \
        char* err_msg = nullptr;                                                                     \
        const int result_code = sqlite3_exec(database, sql, callback, nullptr, &err_msg);            \
        if (result_code != SQLITE_OK) {                                                              \
            LOG(ERROR) << "SQLite error [" << __FILE__ << ", line " << __LINE__ << "]: " << err_msg; \
            sqlite3_free(err_msg);                                                                   \
        }                                                                                            \
    }
struct Database {
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FeatureKeypointsBlob;
    typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FeatureDescriptorsBlob;
    typedef Eigen::Matrix<point2D_t, Eigen::Dynamic, 2, Eigen::RowMajor> FeatureMatchesBlob;

    sqlite3* database_ = nullptr;
    std::vector<sqlite3_stmt*> sql_stmts_;
    sqlite3_stmt* sql_stmt_read_images_ = nullptr;
    sqlite3_stmt* sql_stmt_read_cameras_ = nullptr;
    sqlite3_stmt* sql_stmt_read_matches_all_ = nullptr;
    sqlite3_stmt* sql_stmt_read_keypoints_ = nullptr;
    sqlite3_stmt* sql_stmt_read_two_view_geometries_ = nullptr;
    Database(const std::string& db_path) {
        Close();
        SQLITE3_CALL(sqlite3_open_v2(db_path.c_str(), &database_,
                                     SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nullptr));

        std::string sql;
        sql = "SELECT * FROM images;";
        SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_images_, 0));
        sql_stmts_.push_back(sql_stmt_read_images_);

        sql = "SELECT * FROM cameras;";
        SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_cameras_, 0));
        sql_stmts_.push_back(sql_stmt_read_cameras_);

        sql = "SELECT * FROM matches WHERE rows > 0;";
        SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_matches_all_, 0));
        sql_stmts_.push_back(sql_stmt_read_matches_all_);

        sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;";
        SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_keypoints_, 0));
        sql_stmts_.push_back(sql_stmt_read_keypoints_);

        sql = "SELECT * FROM two_view_geometries WHERE rows > 0;";
        SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_two_view_geometries_, 0));
        sql_stmts_.push_back(sql_stmt_read_two_view_geometries_);
    }
    size_t CountRows(const std::string& table) {
        const std::string sql = "SELECT COUNT(*) FROM " + table + ";";

        sqlite3_stmt* sql_stmt;
        SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

        size_t count = 0;
        const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
        if (rc == SQLITE_ROW) {
            count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
        }

        SQLITE3_CALL(sqlite3_finalize(sql_stmt));

        return count;
    }
    Image ReadImageRow(sqlite3_stmt* sql_stmt) {
        Image image;

        image.image_id_ = static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0));
        image.name_ = std::string(reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1)));
        image.camera_id_ = static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2));

        return image;
    }
    std::vector<Image> ReadAllImages() {
        std::vector<Image> images;
        images.reserve(CountRows("images"));

        while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
            images.push_back(ReadImageRow(sql_stmt_read_images_));
        }

        SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

        return images;
    }

    std::vector<CamModel::Ptr> ReadAllCameras() {
        std::vector<std::shared_ptr<CamModel>> cameras;
        cameras.reserve(CountRows("cameras"));

        while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_)) == SQLITE_ROW) {
            auto camera = ReadCameraRow(sql_stmt_read_cameras_);
            if (camera) {
                cameras.push_back(camera);
            }
        }

        SQLITE3_CALL(sqlite3_reset(sql_stmt_read_cameras_));

        return cameras;
    }

    template <typename MatrixType>
    MatrixType ReadStaticMatrixBlob(sqlite3_stmt* sql_stmt, const int rc, const int col) {
        CHECK_GE(col, 0);

        MatrixType matrix;

        if (rc == SQLITE_ROW) {
            const size_t num_bytes = static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
            if (num_bytes > 0) {
                CHECK_EQ(num_bytes, matrix.size() * sizeof(typename MatrixType::Scalar));
                memcpy(reinterpret_cast<char*>(matrix.data()), sqlite3_column_blob(sql_stmt, col), num_bytes);
            } else {
                matrix = MatrixType::Zero();
            }
        } else {
            matrix = MatrixType::Zero();
        }

        return matrix;
    }

    template <typename MatrixType>
    MatrixType ReadDynamicMatrixBlob(sqlite3_stmt* sql_stmt, const int rc, const int col) {
        CHECK_GE(col, 0);

        MatrixType matrix;

        if (rc == SQLITE_ROW) {
            const size_t rows = static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
            const size_t cols = static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));

            CHECK_GE(rows, 0);
            CHECK_GE(cols, 0);
            matrix = MatrixType(rows, cols);

            const size_t num_bytes = static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));
            CHECK_EQ(matrix.size() * sizeof(typename MatrixType::Scalar), num_bytes);

            memcpy(reinterpret_cast<char*>(matrix.data()), sqlite3_column_blob(sql_stmt, col + 2), num_bytes);
        } else {
            const typename MatrixType::Index rows =
                (MatrixType::RowsAtCompileTime == Eigen::Dynamic) ? 0 : MatrixType::RowsAtCompileTime;
            const typename MatrixType::Index cols =
                (MatrixType::ColsAtCompileTime == Eigen::Dynamic) ? 0 : MatrixType::ColsAtCompileTime;
            matrix = MatrixType(rows, cols);
        }

        return matrix;
    }

    FeatureMatches FeatureMatchesFromBlob(const FeatureMatchesBlob& blob) {
        CHECK_EQ(blob.cols(), 2);
        FeatureMatches matches(static_cast<size_t>(blob.rows()));
        for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
            matches[i].point2D_idx1 = blob(i, 0);
            matches[i].point2D_idx2 = blob(i, 1);
        }
        return matches;
    }

    std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches() {
        std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;

        int rc;
        while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) == SQLITE_ROW) {
            const image_pair_t pair_id = static_cast<image_pair_t>(sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
            const FeatureMatchesBlob blob =
                ReadDynamicMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_matches_all_, rc, 1);
            all_matches.emplace_back(pair_id, FeatureMatchesFromBlob(blob));
        }

        SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

        return all_matches;
    }
    FeatureKeypoints FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob) {
        FeatureKeypoints keypoints(static_cast<size_t>(blob.rows()));
        if (blob.cols() == 2) {
            for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
                keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1));
            }
        } else if (blob.cols() == 4) {
            for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
                keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1), blob(i, 2), blob(i, 3));
            }
        } else if (blob.cols() == 6) {
            for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
                keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1), blob(i, 2), blob(i, 3), blob(i, 4), blob(i, 5));
            }
        } else {
            std::cout << "Keypoint format not supported";
        }
        return keypoints;
    }
    FeatureKeypointsBlob ReadKeypointsBlob(const image_t image_id) {
        SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

        const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
        FeatureKeypointsBlob blob = ReadDynamicMatrixBlob<FeatureKeypointsBlob>(sql_stmt_read_keypoints_, rc, 0);

        SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));
        return blob;
    }
    FeatureKeypoints ReadKeypoints(const image_t image_id) {
        return FeatureKeypointsFromBlob(ReadKeypointsBlob(image_id));
    }
    std::vector<std::pair<image_pair_t, TwoViewGeometry>> ReadTwoViewGeometries() {
        std::vector<std::pair<image_pair_t, TwoViewGeometry>> all_two_view_geometries;

        int rc;
        while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_two_view_geometries_))) == SQLITE_ROW) {
            const image_pair_t pair_id =
                static_cast<image_pair_t>(sqlite3_column_int64(sql_stmt_read_two_view_geometries_, 0));

            TwoViewGeometry two_view_geometry;

            const FeatureMatchesBlob blob =
                ReadDynamicMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_two_view_geometries_, rc, 1);
            two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);

            two_view_geometry.config = static_cast<int>(sqlite3_column_int64(sql_stmt_read_two_view_geometries_, 4));

            two_view_geometry.F = ReadStaticMatrixBlob<Eigen::Matrix3d>(sql_stmt_read_two_view_geometries_, rc, 5);
            two_view_geometry.E = ReadStaticMatrixBlob<Eigen::Matrix3d>(sql_stmt_read_two_view_geometries_, rc, 6);
            two_view_geometry.H = ReadStaticMatrixBlob<Eigen::Matrix3d>(sql_stmt_read_two_view_geometries_, rc, 7);
            const Eigen::Vector4d quat_wxyz =
                ReadStaticMatrixBlob<Eigen::Vector4d>(sql_stmt_read_two_view_geometries_, rc, 8);
            two_view_geometry.cam2_from_cam1.SetQuat(
                Eigen::Quaterniond(quat_wxyz(0), quat_wxyz(1), quat_wxyz(2), quat_wxyz(3)));
            two_view_geometry.cam2_from_cam1.SetTrans(
                ReadStaticMatrixBlob<Eigen::Vector3d>(sql_stmt_read_two_view_geometries_, rc, 9));

            two_view_geometry.F.transposeInPlace();
            two_view_geometry.E.transposeInPlace();
            two_view_geometry.H.transposeInPlace();

            all_two_view_geometries.emplace_back(pair_id, std::move(two_view_geometry));
        }

        SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometries_));

        return all_two_view_geometries;
    }
    enum CameraModelId {
        kInvalid = -1,           // = -1
        kSimplePinhole,          // = 0
        kPinhole,                // = 1
        kSimpleRadial,           // = 2
        kRadial,                 // = 3
        kOpenCV,                 // = 4
        kOpenCVFisheye,          // = 5
        kFullOpenCV,             // = 6
        kFOV,                    // = 7
        kSimpleRadialFisheye,    // = 8
        kRadialFisheye,          // = 9
        kThinPrismFisheye,       // = 10
        kRadTanThinPrismFisheye  // = 11
    };
    std::shared_ptr<CamModel> ReadCameraRow(sqlite3_stmt* sql_stmt) {
        camera_t camera_id = static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 0));
        CameraModelId model_id = static_cast<CameraModelId>(sqlite3_column_int64(sql_stmt, 1));
        size_t width = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 2));
        size_t height = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 3));

        const size_t num_params_bytes = static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
        const size_t num_params = num_params_bytes / sizeof(double);
        std::vector<double> params(num_params, 0.);
        memcpy(params.data(), sqlite3_column_blob(sql_stmt, 4), num_params_bytes);

        // Construct camera model based on COLMAP model ID
        std::shared_ptr<CamModel> camera;
        switch (model_id) {
            case kSimplePinhole: {
                // Params: [f, cx, cy]
                if (params.size() >= 3) {
                    double f = params[0];
                    double cx = params[1];
                    double cy = params[2];
                    camera = std::make_shared<PinholeCamera>(width, height, f, f, cx, cy);
                }
                break;
            }
            case kPinhole: {
                // Params: [fx, fy, cx, cy]
                if (params.size() >= 4) {
                    double fx = params[0];
                    double fy = params[1];
                    double cx = params[2];
                    double cy = params[3];
                    camera = std::make_shared<PinholeCamera>(width, height, fx, fy, cx, cy);
                }
                break;
            }
            case kOpenCV: {
                // Params: [fx, fy, cx, cy, k1, k2, p1, p2]
                // Map to PinholeRadialCamera: [fx, fy, cx, cy, k1, k2, k3, p1, p2]
                // Note: kOpenCV has k1, k2, p1, p2; PinholeRadial needs k1, k2, k3, p1, p2
                if (params.size() >= 8) {
                    double fx = params[0];
                    double fy = params[1];
                    double cx = params[2];
                    double cy = params[3];
                    double k1 = params[4];
                    double k2 = params[5];
                    double p1 = params[6];
                    double p2 = params[7];
                    double k3 = 0.0;  // OpenCV does not have k3 by default
                    camera = std::make_shared<PinholeRadialCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2);
                }
                break;
            }
            case kOpenCVFisheye: {
                // Params: [fx, fy, cx, cy, k1, k2, k3, k4]
                // Map directly to PinholeFisheyeCamera
                if (params.size() >= 8) {
                    double fx = params[0];
                    double fy = params[1];
                    double cx = params[2];
                    double cy = params[3];
                    double k1 = params[4];
                    double k2 = params[5];
                    double k3 = params[6];
                    double k4 = params[7];
                    camera = std::make_shared<PinholeFisheyeCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, k4);
                }
                break;
            }
            case kRadial: {
                // Params: [f, cx, cy, k1, k2]
                // Map to PinholeRadialCamera with single focal length
                if (params.size() >= 5) {
                    double f = params[0];
                    double cx = params[1];
                    double cy = params[2];
                    double k1 = params[3];
                    double k2 = params[4];
                    double k3 = 0.0;
                    double p1 = 0.0;
                    double p2 = 0.0;
                    camera = std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k1, k2, k3, p1, p2);
                }
                break;
            }
            case kFullOpenCV: {
                // Params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
                // Map first 8 params to PinholeRadialCamera (k3 is the 9th param)
                if (params.size() >= 12) {
                    double fx = params[0];
                    double fy = params[1];
                    double cx = params[2];
                    double cy = params[3];
                    double k1 = params[4];
                    double k2 = params[5];
                    double p1 = params[6];
                    double p2 = params[7];
                    double k3 = params[8];  // FullOpenCV has k3
                    camera = std::make_shared<PinholeRadialCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2);
                }
                break;
            }
            default: {
                // For other unsupported models, create a simple pinhole camera as fallback
                LOG(WARNING) << "Camera ID " << camera_id
                             << ": Unsupported camera model ID: " << static_cast<int>(model_id)
                             << ", creating default pinhole camera.";
                if (params.size() >= 4) {
                    double fx = params[0];
                    double fy = params[1];
                    double cx = params[2];
                    double cy = params[3];
                    camera = std::make_shared<PinholeCamera>(width, height, fx, fy, cx, cy);
                } else if (params.size() >= 3) {
                    double f = params[0];
                    double cx = params[1];
                    double cy = params[2];
                    camera = std::make_shared<PinholeCamera>(width, height, f, f, cx, cy);
                }
                break;
            }
        }

        return camera;
    }
    void Close() {
        for (auto& stmt : sql_stmts_) {
            SQLITE3_CALL(sqlite3_finalize(stmt));
        }
        SQLITE3_CALL(sqlite3_close(database_));
    }
};
struct Reconstruction {
    void LoadFromDatabase(const std::string& db_path) {
        Database db(db_path);

        // Read cameras
        auto all_cameras = db.ReadAllCameras();
        for (size_t i = 0; i < all_cameras.size(); ++i) {
            if (all_cameras[i]) {
                cameras_[static_cast<camera_t>(i)] = all_cameras[i];
            }
        }
        std::cout << "Read " << all_cameras.size() << " cameras from database." << std::endl;
        for (const auto& camera : cameras_) {
            std::string model_name = CameraModelToString(camera.second->getType());
            std::cout << "Camera " << camera.first << ": " << model_name << std::endl;
            std::cout << "  width: " << camera.second->w() << std::endl;
            std::cout << "  height: " << camera.second->h() << std::endl;
        }
        // Read images
        auto all_images = db.ReadAllImages();
        std::cout << "Read " << all_images.size() << " images from database." << std::endl;

        int all_keypoints_count = 0;
        for (const auto& image : all_images) {
            images_[image.image_id_] = std::make_shared<struct Image>(image);
            Image::Ptr img = images_[image.image_id_];
            FeatureKeypoints keypoints = db.ReadKeypoints(image.image_id_);
            img->points2D_.resize(keypoints.size());
            img->camera_id_ = 0;
            for (size_t i = 0; i < keypoints.size(); ++i) {
                img->points2D_[i].xy.x() = keypoints[i].x;
                img->points2D_[i].xy.y() = keypoints[i].y;
            }
            all_keypoints_count += keypoints.size();
        }
        int mean_keypoints_per_image = all_keypoints_count / all_images.size();
        std::cout << "Average number of keypoints per image: " << mean_keypoints_per_image << std::endl;

        db.ReadAllMatches();
        auto two_view_geometries = db.ReadTwoViewGeometries();
        for (const auto& pair_id_and_geometry : two_view_geometries) {
            two_view_geometries_[pair_id_and_geometry.first] = pair_id_and_geometry.second;
        }
        std::cout << "Read " << two_view_geometries.size() << " two-view geometries from database." << std::endl;
        db.Close();
    }

    static bool IsNotWhiteSpace(const int character) {
        return character != ' ' && character != '\n' && character != '\r' && character != '\t';
    }
    static void StringLeftTrim(std::string* str) {
        str->erase(str->begin(), std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
    }

    static void StringRightTrim(std::string* str) {
        str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(), str->end());
    }

    static void StringTrim(std::string* str) {
        StringLeftTrim(str);
        StringRightTrim(str);
    }
    void ReadImagesText(std::istream& stream) {
        CHECK(stream.good());

        std::string line;
        std::string item;

        std::vector<Eigen::Vector2d> points2D;
        std::vector<point3D_t> point3D_ids;

        while (std::getline(stream, line)) {
            StringTrim(&line);

            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::stringstream line_stream1(line);

            // ID
            std::getline(line_stream1, item, ' ');
            const image_t image_id = std::stoul(item);

            std::shared_ptr<struct Image> image = std::make_shared<struct Image>();
            image->image_id_ = image_id;

            Pose3 cam_from_world;

            std::getline(line_stream1, item, ' ');
            cam_from_world.q_.w() = std::stold(item);

            std::getline(line_stream1, item, ' ');
            cam_from_world.q_.x() = std::stold(item);

            std::getline(line_stream1, item, ' ');
            cam_from_world.q_.y() = std::stold(item);

            std::getline(line_stream1, item, ' ');
            cam_from_world.q_.z() = std::stold(item);

            std::getline(line_stream1, item, ' ');
            cam_from_world.t_.x() = std::stold(item);

            std::getline(line_stream1, item, ' ');
            cam_from_world.t_.y() = std::stold(item);

            std::getline(line_stream1, item, ' ');
            cam_from_world.t_.z() = std::stold(item);

            image->cam_from_world_ = cam_from_world;

            // CAMERA_ID
            std::getline(line_stream1, item, ' ');
            image->camera_id_ = std::stoul(item);

            // NAME
            std::getline(line_stream1, item, ' ');
            image->name_ = item;

            // POINTS2D
            if (!std::getline(stream, line)) {
                break;
            }

            StringTrim(&line);
            std::stringstream line_stream2(line);

            points2D.clear();
            point3D_ids.clear();

            if (!line.empty()) {
                while (!line_stream2.eof()) {
                    Eigen::Vector2d point;

                    std::getline(line_stream2, item, ' ');
                    point.x() = std::stold(item);

                    std::getline(line_stream2, item, ' ');
                    point.y() = std::stold(item);

                    points2D.push_back(point);

                    std::getline(line_stream2, item, ' ');
                    if (item == "-1") {
                        point3D_ids.push_back(kInvalidPoint3DId);
                    } else {
                        point3D_ids.push_back(std::stoll(item));
                    }
                }
            }

            image->points2D_.resize(points2D.size());
            for (int i = 0; i < points2D.size(); ++i) {
                image->points2D_[i].xy = points2D[i];
            }

            for (point2D_t point2D_idx = 0; point2D_idx < image->points2D_.size(); ++point2D_idx) {
                if (point3D_ids[point2D_idx] != kInvalidPoint3DId) {
                    image->points2D_.at(point2D_idx).point3D_id = point3D_ids[point2D_idx];
                }
            }

            images_[image_id] = image;
        }  // for image
    }

    void ReadPoints3DText(std::istream& stream) {
        CHECK(stream.good());

        std::string line;
        std::string item;

        while (std::getline(stream, line)) {
            StringTrim(&line);

            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::stringstream line_stream(line);

            // ID
            std::getline(line_stream, item, ' ');
            const point3D_t point3D_id = std::stoll(item);

            struct Point3D point3D;

            // XYZ
            std::getline(line_stream, item, ' ');
            point3D.xyz(0) = std::stold(item);

            std::getline(line_stream, item, ' ');
            point3D.xyz(1) = std::stold(item);

            std::getline(line_stream, item, ' ');
            point3D.xyz(2) = std::stold(item);

            // Color
            std::getline(line_stream, item, ' ');
            // point3D.color(0) = static_cast<uint8_t>(std::stoi(item));

            std::getline(line_stream, item, ' ');
            // point3D.color(1) = static_cast<uint8_t>(std::stoi(item));

            std::getline(line_stream, item, ' ');
            // point3D.color(2) = static_cast<uint8_t>(std::stoi(item));

            // ERROR
            std::getline(line_stream, item, ' ');
            // point3D.error = std::stold(item);

            // TRACK
            while (!line_stream.eof()) {
                Observation track_el;

                std::getline(line_stream, item, ' ');
                StringTrim(&item);
                if (item.empty()) {
                    break;
                }
                track_el.image_id = std::stoul(item);

                std::getline(line_stream, item, ' ');
                track_el.point2D_idx = std::stoul(item);

                point3D.track.push_back(track_el);
            }

            point3D.track.shrink_to_fit();

            points3D_[point3D_id] = point3D;
        }
    }
    void LoadFromCOLMAPResult(const std::string& colmap_result_path) {
        if (!fs::is_regular_file(colmap_result_path + "/cameras.txt") ||
            !fs::is_regular_file(colmap_result_path + "/images.txt") ||
            !fs::is_regular_file(colmap_result_path + "/points3D.txt")) {
            std::cout << "cameras.txt or images.txt or points3D.txt not found in " << colmap_result_path << std::endl;
            return;
        }

        std::ifstream fin(colmap_result_path + "/cameras.txt");
        if (fin) {
            std::string line;
            while (std::getline(fin, line)) {
                // trim
                if (line.empty() || line[0] == '#') continue;
                std::stringstream ss(line);
                unsigned long cam_id_ul = 0;
                std::string model;
                size_t width = 0, height = 0;
                if (!(ss >> cam_id_ul)) continue;
                if (!(ss >> model)) continue;
                if (!(ss >> width)) continue;
                if (!(ss >> height)) continue;

                std::vector<double> params;
                double p;
                while (ss >> p) params.push_back(p);

                camera_t cam_id = static_cast<camera_t>(cam_id_ul);

                // construct camera according to model name (COLMAP naming)
                CamModel::Ptr cam_ptr = nullptr;
                if (model == "SIMPLE_PINHOLE") {
                    if (params.size() >= 3) {
                        double f = params[0];
                        double cx = params[1];
                        double cy = params[2];
                        cam_ptr = std::make_shared<PinholeCamera>(width, height, f, f, cx, cy);
                    }
                } else if (model == "PINHOLE") {
                    if (params.size() >= 4) {
                        double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                        cam_ptr = std::make_shared<PinholeCamera>(width, height, fx, fy, cx, cy);
                    }
                } else if (model == "SIMPLE_RADIAL") {
                    if (params.size() >= 4) {
                        double f = params[0], cx = params[1], cy = params[2], k = params[3];
                        cam_ptr =
                            std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k, 0.0, 0.0, 0.0, 0.0);
                    }
                } else if (model == "RADIAL") {
                    if (params.size() >= 5) {
                        double f = params[0], cx = params[1], cy = params[2], k1 = params[3], k2 = params[4];
                        cam_ptr =
                            std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k1, k2, 0.0, 0.0, 0.0);
                    }
                } else if (model == "OPENCV") {
                    if (params.size() >= 8) {
                        double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                        double k1 = params[4], k2 = params[5], p1 = params[6], p2 = params[7];
                        double k3 = 0.0;
                        cam_ptr =
                            std::make_shared<PinholeRadialCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2);
                    }
                } else if (model == "OPENCV_FISHEYE") {
                    if (params.size() >= 8) {
                        double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                        double k1 = params[4], k2 = params[5], k3 = params[6], k4 = params[7];
                        cam_ptr = std::make_shared<PinholeFisheyeCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, k4);
                    }
                } else if (model == "FULL_OPENCV") {
                    if (params.size() >= 9) {
                        double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                        double k1 = params[4], k2 = params[5], p1 = params[6], p2 = params[7], k3 = params[8];
                        cam_ptr =
                            std::make_shared<PinholeRadialCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2);
                    }
                } else {
                    // fallback: try to use first 4 or 3 params
                    if (params.size() >= 4) {
                        double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                        cam_ptr = std::make_shared<PinholeCamera>(width, height, fx, fy, cx, cy);
                    } else if (params.size() >= 3) {
                        double f = params[0], cx = params[1], cy = params[2];
                        cam_ptr = std::make_shared<PinholeCamera>(width, height, f, f, cx, cy);
                    }
                }

                if (cam_ptr) cameras_[cam_id] = cam_ptr;
            }
        }

        std::ifstream images_fin(colmap_result_path + "/images.txt");
        ReadImagesText(images_fin);
        std::ifstream points3D_fin(colmap_result_path + "/points3D.txt");
        ReadPoints3DText(points3D_fin);

        double track_len_sum = 0;
        for (const auto [id, point3d] : points3D_) {
            track_len_sum += point3d.track.size();
        }
        std::cout << "Read " << cameras_.size() << " cameras, " << images_.size() << " images, " << points3D_.size()
                  << " points3D." << std::endl;
        std::cout << "Average track length: " << track_len_sum / points3D_.size() << std::endl;
    }

    int FilterOutlier(const double max_reproj_error) {
       int filter_count = 0;
        for (auto& [id, point3D] : points3D_) {
            for (int j = 0; j < point3D.track.size(); ++j) {
                Image::Ptr img = images_[point3D.track[j].image_id];

                CamModel::Ptr cameara = cameras_[img->CameraId()];
                Point2D& point2D = img->points2D_[point3D.track[j].point2D_idx];
                Eigen::Vector3d point_cam = img->CameraFromWorld() * point3D.xyz;
                double err = (cameara->project(point_cam) - point2D.xy).norm();
                if (err > max_reproj_error) {
                    point3D.track.erase(point3D.track.begin() + j);
                    --j;
                }
            }  // for obs
        }  // for points

        for (auto it = points3D_.begin(); it != points3D_.end();) {
            if (it->second.track.size() < 2) {
                it = points3D_.erase(it);
                ++filter_count;
            } else {
                ++it;
            }
        }
        return filter_count;
    }

    double MeanTrackLength() {
        double track_len_sum = 0;
        for (const auto [id, point3d] : points3D_) {
            track_len_sum += point3d.track.size();
        }
        return track_len_sum / points3D_.size();
    }
    double CalcMeanError() {
        double error_count = 0;
        double error_sum = 0;
        for (auto& [id, point3D] : points3D_) {
            for (int j = 0; j < point3D.track.size(); ++j) {
                Image::Ptr img = image(point3D.track[j].image_id);
                CamModel::Ptr cameara = camera(img->CameraId());
                Point2D& point2D = img->points2D_[point3D.track[j].point2D_idx];
                if (!point3D.xyz.hasNaN()) {
                    Eigen::Vector3d point_cam = img->CameraFromWorld() * point3D.xyz;
                    double err = (cameara->project(point_cam) - point2D.xy).norm();
                    error_sum += err;
                    ++error_count;
                }
            }  // for obs
        }  // for points
        return error_sum / error_count;
    }

    bool TriangulateNViewAlgebraic(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points,
                                   const std::vector<Pose3>& poses, Eigen::Vector4d& X) {
        assert(poses.size() == points.cols());

        Eigen::Matrix4d AtA = Eigen::Matrix4d::Zero();
        for (Eigen::Matrix<double, 3, Eigen::Dynamic>::Index i = 0; i < points.cols(); ++i) {
            const Eigen::Vector3d point_norm = points.col(i).normalized();
            const Eigen::Matrix<double, 3, 4> cost =
                poses[i].Mat34() - point_norm * point_norm.transpose() * poses[i].Mat34();
            AtA += cost.transpose() * cost;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(AtA);
        X = eigen_solver.eigenvectors().col(0);
        return eigen_solver.info() == Eigen::Success;
    }

    bool TriangulatePoint(const point3D_t& point_3d_id) {
        Point3D point3D = points3D_[point_3d_id];
        // prepare the data
        std::vector<Pose3> poses;
        Eigen::Matrix<double, 3, Eigen::Dynamic> points_2d;
        Eigen::Vector4d X;

        poses.resize(point3D.track.size());
        points_2d.resize(3, point3D.track.size());
        for (int i = 0; i < point3D.track.size(); ++i) {
            Observation track_el = point3D.track[i];
            Image::Ptr img = image(track_el.image_id);
            if (img->cam_from_world_.IsValid()) {
                std::cout << "Image " << track_el.image_id << " has no camera pose." << std::endl;
                return false;
            }
            poses[i] = img->cam_from_world_;
            CamModel::Ptr camera = cameras_[img->camera_id_];
            Eigen::Vector2d img_p = img->points2D_[track_el.point2D_idx].xy;
            points_2d.col(i) = camera->ima2cam(camera->get_ud_pixel(img_p)).homogeneous();
        }
        if (TriangulateNViewAlgebraic(points_2d, poses, X)) {
            points3D_[point_3d_id].xyz = X.head<3>();
            return true;
        }
        return false;
    }
    std::unordered_map<camera_t, CamModel::Ptr> cameras_;
    std::unordered_map<camera_t, Image::Ptr> images_;
    std::unordered_map<point3D_t, Point3D> points3D_;
    std::unordered_map<image_pair_t, TwoViewGeometry> two_view_geometries_;

    CamModel::Ptr& camera(camera_t camera_id) { return cameras_[camera_id]; }
    Image::Ptr& image(image_t image_id) { return images_[image_id]; }
    Point3D& point3D(point3D_t point3D_id) { return points3D_[point3D_id]; }
    TwoViewGeometry& two_view_geometry(image_pair_t image_pair_id) { return two_view_geometries_[image_pair_id]; }
};

namespace std {
template <>
struct hash<Eigen::Vector2i> {
    size_t operator()(const Eigen::Vector2i &s) const {
        using std::hash;
        using std::size_t;
        return ((hash<int64_t>()(s.x()) ^ (hash<int64_t>()(s.y()) << 1)) >> 1);
    }
};
}
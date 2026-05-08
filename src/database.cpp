#include "database.h"

#include <cstring>
#include <iostream>

Database::Database(const std::string& db_path) {
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

size_t Database::CountRows(const std::string& table) {
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

Image::Ptr Database::ReadImageRow(sqlite3_stmt* sql_stmt) {
    Image::Ptr image = std::make_shared<Image>();

    image->image_id_ = static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0));
    image->name_ = std::string(reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1)));
    image->camera_id_ = static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2));

    return image;
}

std::vector<Image::Ptr> Database::ReadAllImages() {
    std::vector<Image::Ptr> images;
    images.reserve(CountRows("images"));

    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
        images.push_back(ReadImageRow(sql_stmt_read_images_));
    }

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

    return images;
}

std::vector<CamModel::Ptr> Database::ReadAllCameras() {
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

FeatureMatches Database::FeatureMatchesFromBlob(const FeatureMatchesBlob& blob) {
    CHECK_EQ(blob.cols(), 2);
    FeatureMatches matches(static_cast<size_t>(blob.rows()));
    for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
        matches[i].point2D_idx1 = blob(i, 0);
        matches[i].point2D_idx2 = blob(i, 1);
    }
    return matches;
}

std::vector<std::pair<image_pair_t, FeatureMatches>> Database::ReadAllMatches() {
    std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) == SQLITE_ROW) {
        const image_pair_t pair_id = static_cast<image_pair_t>(sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
        const FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_matches_all_, rc, 1);
        all_matches.emplace_back(pair_id, FeatureMatchesFromBlob(blob));
    }

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

    return all_matches;
}

FeatureKeypoints Database::FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob) {
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

Database::FeatureKeypointsBlob Database::ReadKeypointsBlob(const image_t image_id) {
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
    FeatureKeypointsBlob blob = ReadDynamicMatrixBlob<FeatureKeypointsBlob>(sql_stmt_read_keypoints_, rc, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));
    return blob;
}

FeatureKeypoints Database::ReadKeypoints(const image_t image_id) {
    return FeatureKeypointsFromBlob(ReadKeypointsBlob(image_id));
}

std::vector<std::pair<image_pair_t, TwoViewGeometry>> Database::ReadTwoViewGeometries() {
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

std::shared_ptr<CamModel> Database::ReadCameraRow(sqlite3_stmt* sql_stmt) {
    camera_t camera_id = static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 0));
    CameraModelId model_id = static_cast<CameraModelId>(sqlite3_column_int64(sql_stmt, 1));
    size_t width = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 2));
    size_t height = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 3));

    const size_t num_params_bytes = static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
    const size_t num_params = num_params_bytes / sizeof(double);
    std::vector<double> params(num_params, 0.0);
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
            LOG(WARNING) << "Camera ID " << camera_id << ": Unsupported camera model ID: "
                         << static_cast<int>(model_id) << ", creating default pinhole camera.";
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

void Database::Close() {
    for (auto& stmt : sql_stmts_) {
        SQLITE3_CALL(sqlite3_finalize(stmt));
    }
    SQLITE3_CALL(sqlite3_close(database_));
}


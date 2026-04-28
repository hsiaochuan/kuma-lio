#pragma once
#include <algorithm>
#include <boost/filesystem.hpp>
#include <vector>
#include "cameras/cameras.h"
#include "pose3.h"
#include "types.h"
#include <sqlite3.h>
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
        const std::string sql = "SELECT COUNT(*) FROM" + table + ";";

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
        auto all_images = db.ReadAllImages();
        std::cout << "Read " << all_images.size() << " images from database." << std::endl;
        db.ReadAllMatches();
        auto two_view_geometries = db.ReadTwoViewGeometries();
        std::cout << "Read " << two_view_geometries.size() << " two-view geometries from database." << std::endl;
        db.Close();
    }
    std::unordered_map<camera_t, std::shared_ptr<CameraBase>> cameras_;
    std::unordered_map<camera_t, std::shared_ptr<Image>> images_;
};
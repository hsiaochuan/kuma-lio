#pragma once
#include <sqlite3.h>
#include <boost/filesystem.hpp>
#include <cstring>
#include <memory>
#include <vector>
#include "cameras/cameras.h"
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
    typedef Eigen::Matrix<point2d_t, Eigen::Dynamic, 2, Eigen::RowMajor> FeatureMatchesBlob;

    sqlite3* database_ = nullptr;
    std::vector<sqlite3_stmt*> sql_stmts_;
    sqlite3_stmt* sql_stmt_read_images_ = nullptr;
    sqlite3_stmt* sql_stmt_read_cameras_ = nullptr;
    sqlite3_stmt* sql_stmt_read_matches_all_ = nullptr;
    sqlite3_stmt* sql_stmt_read_keypoints_ = nullptr;
    sqlite3_stmt* sql_stmt_read_two_view_geometries_ = nullptr;
    Database(const std::string& db_path);
    size_t CountRows(const std::string& table);
    Image::Ptr ReadImageRow(sqlite3_stmt* sql_stmt);
    std::vector<Image::Ptr> ReadAllImages();
    std::vector<CamModel::Ptr> ReadAllCameras();

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

    FeatureMatches FeatureMatchesFromBlob(const FeatureMatchesBlob& blob);
    std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches();
    FeatureKeypoints FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob);
    FeatureKeypointsBlob ReadKeypointsBlob(const image_t image_id);
    FeatureKeypoints ReadKeypoints(const image_t image_id);
    std::vector<std::pair<image_pair_t, TwoViewGeometry>> ReadTwoViewGeometries();
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
    std::shared_ptr<CamModel> ReadCameraRow(sqlite3_stmt* sql_stmt);
    void Close();
};



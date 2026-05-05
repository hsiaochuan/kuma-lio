#pragma once
#include <sqlite3.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>
#include "cameras/cameras.h"
#include "database.h"
#include "pose3.h"
#include "types.h"
using namespace faster_lio;
namespace fs = boost::filesystem;

struct Sfm_Data {
    void LoadFromDatabase(const std::string& db_path) {
        Database db(db_path);

        // Read cameras
        auto all_cameras = db.ReadAllCameras();
        camera_t camera_id = 1;
        for (const auto & cam : all_cameras) {
            cameras_[camera_id] = cam;
            camera_id++;

        }
        std::cout << "Read " << all_cameras.size() << " cameras from database." << std::endl;
        for (const auto& camera : cameras_) {
            std::string model_name = CameraModelToString(camera.second->get_type());
            std::vector<double> params = camera.second->get_params();
            std::stringstream ss;
            ss << model_name << " ";
            for (double param : params) {
                ss << param << " ";
            }
            std::cout << "Camera " << camera.first << ": " << ss.str() << std::endl;
        }
        // Read images
        auto all_images = db.ReadAllImages();
        for (const auto& image : all_images) {
            CHECK(cameras_.count(image->camera_id_) > 0)
                << "Camera ID " << image->camera_id_ << " not found for image ID " << image->image_id_;
        }
        std::cout << "Read " << all_images.size() << " images from database." << std::endl;
        int all_keypoints_count = 0;
        for (const auto& im : all_images) {
            FeatureKeypoints keypoints = db.ReadKeypoints(im->image_id_);
            im->points_.resize(keypoints.size());
            for (size_t i = 0; i < keypoints.size(); ++i) {
                im->points_[i].x() = keypoints[i].x;
                im->points_[i].y() = keypoints[i].y;
            }
            all_keypoints_count += keypoints.size();
            images_[im->image_id_] = im;
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

        std::vector<Eigen::Vector2d> points;
        std::vector<landmark_t> landmark_ids;

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

            points.clear();
            landmark_ids.clear();

            if (!line.empty()) {
                while (!line_stream2.eof()) {
                    Eigen::Vector2d point;

                    std::getline(line_stream2, item, ' ');
                    point.x() = std::stold(item);

                    std::getline(line_stream2, item, ' ');
                    point.y() = std::stold(item);

                    points.push_back(point);

                    std::getline(line_stream2, item, ' ');
                    if (item == "-1") {
                        landmark_ids.push_back(kInvalidPoint3DId);
                    } else {
                        landmark_ids.push_back(std::stoll(item));
                    }
                }
            }

            image->points_ = points;
            image->landmark_ids_ = landmark_ids;

            images_[image_id] = image;
        }  // for image
    }

    void ReadCamerasText(std::istream& stream) {
        CHECK(stream.good());

        std::string line;
        while (std::getline(stream, line)) {
            StringTrim(&line);
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::stringstream ss(line);
            unsigned long cam_id_ul = 0;
            std::string model;
            size_t width = 0, height = 0;
            if (!(ss >> cam_id_ul)) {
                continue;
            }
            if (!(ss >> model)) {
                continue;
            }
            if (!(ss >> width)) {
                continue;
            }
            if (!(ss >> height)) {
                continue;
            }

            std::vector<double> params;
            double p = 0.0;
            while (ss >> p) {
                params.push_back(p);
            }

            camera_t cam_id = static_cast<camera_t>(cam_id_ul);

            // Construct camera according to model name (COLMAP naming)
            CamModel::Ptr cam_ptr = nullptr;
            if (model == "SIMPLE_PINHOLE") {
                // f, cx, cy
                if (params.size() >= 3) {
                    double f = params[0], cx = params[1], cy = params[2];
                    cam_ptr = std::make_shared<PinholeCamera>(width, height, f, f, cx, cy);
                }
            } else if (model == "PINHOLE") {
                // fx, fy, cx, cy
                if (params.size() >= 4) {
                    double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                    cam_ptr = std::make_shared<PinholeCamera>(width, height, fx, fy, cx, cy);
                }
            } else if (model == "SIMPLE_RADIAL") {
                // f, cx, cy, k
                if (params.size() >= 4) {
                    double f = params[0], cx = params[1], cy = params[2], k = params[3];
                    cam_ptr = std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k, 0.0, 0.0, 0.0, 0.0);
                }
            } else if (model == "RADIAL") {
                // f, cx, cy, k1, k2
                if (params.size() >= 5) {
                    double f = params[0], cx = params[1], cy = params[2], k1 = params[3], k2 = params[4];
                    cam_ptr = std::make_shared<PinholeRadialCamera>(width, height, f, f, cx, cy, k1, k2, 0.0, 0.0, 0.0);
                }
            } else if (model == "OPENCV") {
                // fx, fy, cx, cy, k1, k2, p1, p2
                if (params.size() >= 8) {
                    double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                    double k1 = params[4], k2 = params[5], p1 = params[6], p2 = params[7];
                    double k3 = 0.0;
                    cam_ptr = std::make_shared<PinholeRadialCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2);
                }
            } else if (model == "OPENCV_FISHEYE") {
                // fx, fy, cx, cy, k1, k2, k3 ,k4
                if (params.size() >= 8) {
                    double fx = params[0], fy = params[1], cx = params[2], cy = params[3];
                    double k1 = params[4], k2 = params[5], k3 = params[6], k4 = params[7];
                    cam_ptr = std::make_shared<PinholeFisheyeCamera>(width, height, fx, fy, cx, cy, k1, k2, k3, k4);
                }
            } else {
                LOG(WARNING) << "Unsupported camera model: " << model;
                continue;
            }

            if (cam_ptr) {
                cameras_[cam_id] = cam_ptr;
            }
        }
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
            const landmark_t point3D_id = std::stoll(item);

            struct Landmark point3D;

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
                track_el.point2d_id = std::stoul(item);

                point3D.track.push_back(track_el);
            }

            point3D.track.shrink_to_fit();

            landmarks_[point3D_id] = point3D;
        }
    }
    void WriteCamerasText(std::ostream& stream) {
        CHECK(stream.good());

        stream.precision(17);
        stream << "# Camera list with one line of data per camera:" << std::endl;
        stream << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;

        for (const auto& [camera_id, camera] : cameras_) {
            if (!camera) {
                continue;
            }

            std::string model;
            std::vector<double> params;
            const CAMERA_MODEL type = camera->get_type();
            if (type == PINHOLE) {
                const std::vector<double> raw = camera->get_params();
                const double fx = raw[0];
                const double fy = raw[1];
                const double cx = raw[2];
                const double cy = raw[3];
                model = "PINHOLE";
                params = {fx, fy, cx, cy};
            } else if (type == PINHOLE_RADIAL) {
                const std::vector<double> raw = camera->get_params();
                const double fx = raw[0];
                const double fy = raw[1];
                const double cx = raw[2];
                const double cy = raw[3];
                const double k1 = raw[4];
                const double k2 = raw[5];
                const double k3 = raw[6];
                const double p1 = raw[7];
                const double p2 = raw[8];
                model = "OPENCV";
                params = {fx, fy, cx, cy, k1, k2, p1, p2};
            } else if (type == PINHOLE_FISHEYE) {
                model = "OPENCV_FISHEYE";
                params = camera->get_params();
            } else {
                LOG(WARNING) << "Unsupported camera type for writing: " << static_cast<int>(type);
                continue;
            }

            stream << camera_id << " " << model << " " << camera->w() << " " << camera->h();
            for (const double value : params) {
                stream << " " << value;
            }
            stream << std::endl;
        }
    }
    void WriteImagesText(std::ostream& stream) {
        CHECK(stream.good());

        // Ensure that we don't loose any precision by storing in text.
        stream.precision(17);

        stream << "# Image list with two lines of data per image:" << std::endl;
        stream << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, "
                  "NAME"
               << std::endl;
        stream << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;

        std::ostringstream line;
        line.precision(17);

        for (const auto& [image_id, image] : images_) {
            line.str("");
            line.clear();

            line << image_id << " ";

            const Pose3& cam_from_world = image->CameraFromWorld();
            line << cam_from_world.q_.w() << " ";
            line << cam_from_world.q_.x() << " ";
            line << cam_from_world.q_.y() << " ";
            line << cam_from_world.q_.z() << " ";
            line << cam_from_world.t_.x() << " ";
            line << cam_from_world.t_.y() << " ";
            line << cam_from_world.t_.z() << " ";

            line << image->CameraId() << " ";

            line << image->Name();

            stream << line.str() << std::endl;

            line.str("");
            line.clear();

            for (point2d_t i = 0; i < image->points_.size(); ++i) {
                line << image->points_[i].x() << " ";
                line << image->points_[i].y() << " ";
                if (image->landmark_ids_[i] != kInvalidPoint3DId) {
                    line << image->landmark_ids_[i] << " ";
                } else {
                    line << -1 << " ";
                }
            }
            if (image->points_.size() > 0) {
                line.seekp(-1, std::ios_base::end);
            }
            stream << line.str() << std::endl;
        }
    }
    void WritePoints3DText(std::ostream& stream) {
        CHECK(stream.good());

        // Ensure that we don't loose any precision by storing in text.
        stream.precision(17);

        stream << "# 3D point list with one line of data per point:" << std::endl;
        stream << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                  "TRACK[] as (IMAGE_ID, POINT2D_IDX)"
               << std::endl;

        for (const auto& [point_id, point3D] : landmarks_) {
            stream << point_id << " ";
            stream << point3D.xyz(0) << " ";
            stream << point3D.xyz(1) << " ";
            stream << point3D.xyz(2) << " ";
            stream << static_cast<int>(255) << " ";
            stream << static_cast<int>(255) << " ";
            stream << static_cast<int>(255) << " ";
            stream << 0.0 << " ";

            std::ostringstream line;
            line.precision(17);

            for (const auto& track_el : point3D.track) {
                line << track_el.image_id << " ";
                line << track_el.point2d_id << " ";
            }

            std::string line_string = line.str();
            line_string = line_string.substr(0, line_string.size() - 1);

            stream << line_string << std::endl;
        }
    }
    void WriteCOLMAPText(const std::string& colmap_result_path) {
        std::ofstream cam_ofs(colmap_result_path + "/cameras.txt");
        WriteCamerasText(cam_ofs);
        std::ofstream img_ofs(colmap_result_path + "/images.txt");
        WriteImagesText(img_ofs);
        std::ofstream landmark_ofs(colmap_result_path + "/points3D.txt");
        WritePoints3DText(landmark_ofs);
    }
    void LoadFromCOLMAPResult(const std::string& colmap_result_path) {
        if (!fs::is_regular_file(colmap_result_path + "/cameras.txt") ||
            !fs::is_regular_file(colmap_result_path + "/images.txt") ||
            !fs::is_regular_file(colmap_result_path + "/points3D.txt")) {
            std::cout << "cameras.txt or images.txt or points3D.txt not found in " << colmap_result_path << std::endl;
            return;
        }

        std::ifstream cameras_fin(colmap_result_path + "/cameras.txt");
        ReadCamerasText(cameras_fin);
        std::ifstream images_fin(colmap_result_path + "/images.txt");
        ReadImagesText(images_fin);
        std::ifstream points3D_fin(colmap_result_path + "/points3D.txt");
        ReadPoints3DText(points3D_fin);

        double track_len_sum = 0;
        for (const auto [id, point3d] : landmarks_) {
            track_len_sum += point3d.track.size();
        }
        std::cout << "Read " << cameras_.size() << " cameras, " << images_.size() << " images, " << landmarks_.size()
                  << " points3D." << std::endl;
        std::cout << "Average track length: " << track_len_sum / landmarks_.size() << std::endl;
    }

    int FilterOutlier(const double max_reproj_error) {
        int filter_count = 0;
        for (auto& [id, point3D] : landmarks_) {
            for (int j = 0; j < point3D.track.size(); ++j) {
                Image::Ptr img = images_[point3D.track[j].image_id];

                CamModel::Ptr cameara = cameras_[img->CameraId()];
                Eigen::Vector2d xy = img->points_[point3D.track[j].point2d_id];
                Eigen::Vector3d point_cam = img->CameraFromWorld() * point3D.xyz;
                double err = (cameara->project(point_cam) - xy).norm();
                if (err > max_reproj_error) {
                    point3D.track.erase(point3D.track.begin() + j);
                    --j;
                }
            }  // for obs
        }  // for points

        for (auto it = landmarks_.begin(); it != landmarks_.end();) {
            if (it->second.track.size() < 2) {
                it = landmarks_.erase(it);
                ++filter_count;
            } else {
                ++it;
            }
        }
        return filter_count;
    }

    double MeanTrackLength() {
        double track_len_sum = 0;
        for (const auto [id, point3d] : landmarks_) {
            track_len_sum += point3d.track.size();
        }
        return track_len_sum / landmarks_.size();
    }
    double CalcMeanError() {
        double error_count = 0;
        double error_sum = 0;
        for (auto& [id, point3D] : landmarks_) {
            for (int j = 0; j < point3D.track.size(); ++j) {
                Image::Ptr img = GetImage(point3D.track[j].image_id);
                CamModel::Ptr cameara = GetCamera(img->CameraId());
                Eigen::Vector2d xy = img->points_[point3D.track[j].point2d_id];
                if (!point3D.xyz.hasNaN()) {
                    Eigen::Vector3d point_cam = img->CameraFromWorld() * point3D.xyz;
                    double err = (cameara->project(point_cam) - xy).norm();
                    error_sum += err;
                    ++error_count;
                }
            }  // for obs
        }  // for points
        return error_sum / error_count;
    }

    std::unordered_map<camera_t, CamModel::Ptr> cameras_;
    std::unordered_map<camera_t, Image::Ptr> images_;
    std::unordered_map<landmark_t, Landmark> landmarks_;
    std::unordered_map<image_pair_t, TwoViewGeometry> two_view_geometries_;

    CamModel::Ptr& GetCamera(camera_t camera_id) {
        CHECK(cameras_.find(camera_id) != cameras_.end()) << "Camera ID " << camera_id << " not found.";
        return cameras_[camera_id];
    }
    Image::Ptr& GetImage(image_t image_id) {
        CHECK(images_.find(image_id) != images_.end()) << "Image ID " << image_id << " not found.";
        return images_[image_id];
    }
    Landmark& GetLandMark(landmark_t point3D_id) {
        CHECK(landmarks_.find(point3D_id) != landmarks_.end()) << "Point3D ID " << point3D_id << " not found.";
        return landmarks_[point3D_id];
    }
    TwoViewGeometry& GetTwoViewGeometry(image_pair_t image_pair_id) {
        CHECK(two_view_geometries_.find(image_pair_id) != two_view_geometries_.end())
            << "Image pair ID " << image_pair_id << " not found.";
        return two_view_geometries_[image_pair_id];
    }
};
enum class ETriangulationMethod : unsigned char {
    DIRECT_LINEAR_TRANSFORM,  // DLT
};
void TriangulateDLT(const Mat34& P1, const Vec3& x1, const Mat34& P2, const Vec3& x2, Vec4* X_homogeneous) {
    // Solve:
    // [cross(x0,P0) X = 0]
    // [cross(x1,P1) X = 0]
    Mat4 design;
    design.row(0) = x1[0] * P1.row(2) - x1[2] * P1.row(0);
    design.row(1) = x1[1] * P1.row(2) - x1[2] * P1.row(1);
    design.row(2) = x2[0] * P2.row(2) - x2[2] * P2.row(0);
    design.row(3) = x2[1] * P2.row(2) - x2[2] * P2.row(1);

    const Eigen::JacobiSVD<Mat4> svd(design, Eigen::ComputeFullV);
    (*X_homogeneous) = svd.matrixV().col(3);
}
void TriangulateDLT(const Mat34& P1, const Vec3& x1, const Mat34& P2, const Vec3& x2, Vec3* X_euclidean) {
    Vec4 X_homogeneous;
    TriangulateDLT(P1, x1, P2, x2, &X_homogeneous);
    (*X_euclidean) = X_homogeneous.hnormalized();
}
bool TriangulateDLT(const Mat3& R0, const Vec3& t0, const Vec3& x0, const Mat3& R1, const Vec3& t1, const Vec3& x1,
                    Vec3* X) {
    Mat34 P0, P1;
    P0.block<3, 3>(0, 0) = R0;
    P1.block<3, 3>(0, 0) = R1;
    P0.block<3, 1>(0, 3) = t0;
    P1.block<3, 1>(0, 3) = t1;
    TriangulateDLT(P0, x0, P1, x1, X);
    return x0.dot(R0 * (*X + R0.transpose() * t0)) > 0.0 && x1.dot(R1 * (*X + R1.transpose() * t1)) > 0.0;
}
bool Triangulate2View(const Mat3& R0, const Vec3& t0, const Vec3& bearing0, const Mat3& R1, const Vec3& t1,
                      const Vec3& bearing1, Vec3& X, ETriangulationMethod etri_method) {
    switch (etri_method) {
        case ETriangulationMethod::DIRECT_LINEAR_TRANSFORM:
            return TriangulateDLT(R0, t0, bearing0, R1, t1, bearing1, &X);
            break;
        default:
            return false;
    }
    return false;
}
double Nullspace(const Eigen::Ref<const Mat>& A, Eigen::Ref<Vec> nullspace) {
    if (A.rows() >= A.cols()) {
        Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
        nullspace = svd.matrixV().col(A.cols() - 1);
        return svd.singularValues()(A.cols() - 1);
    }
    // Extend A with rows of zeros to make it square. It's a hack, but it is
    // necessary until Eigen supports SVD with more columns than rows.
    Mat A_extended(A.cols(), A.cols());
    A_extended.block(A.rows(), 0, A.cols() - A.rows(), A.cols()).setZero();
    A_extended.block(0, 0, A.rows(), A.cols()) = A;
    return Nullspace(A_extended, nullspace);
}
void TriangulateNView(const Mat3X& x, const std::vector<Mat34>& poses, Vec4* X) {
    assert(X != nullptr);
    const Mat2X::Index nviews = x.cols();
    assert(static_cast<size_t>(nviews) == poses.size());

    Mat A = Mat::Zero(3 * nviews, 4 + nviews);
    for (Mat::Index i = 0; i < nviews; ++i) {
        A.block<3, 4>(3 * i, 0) = -poses[i];
        A.block<3, 1>(3 * i, 4 + i) = x.col(i);
    }
    Vec X_and_alphas(4 + nviews);
    Nullspace(A, X_and_alphas);
    *X = X_and_alphas.head(4);
}

bool TriangulateNViewAlgebraic(const Mat3X& points, const std::vector<Mat34>& poses, Vec4* X) {
    assert(poses.size() == points.cols());

    Mat4 AtA = Mat4::Zero();
    for (Mat3X::Index i = 0; i < points.cols(); ++i) {
        const Vec3 point_norm = points.col(i).normalized();
        const Mat34 cost = poses[i] - point_norm * point_norm.transpose() * poses[i];
        AtA += cost.transpose() * cost;
    }

    Eigen::SelfAdjointEigenSolver<Mat4> eigen_solver(AtA);
    *X = eigen_solver.eigenvectors().col(0);
    return eigen_solver.info() == Eigen::Success;
}
bool TrackTriangulate(Sfm_Data& recon, std::vector<Observation>& obs, Vec3& X) {
    if (obs.size() >= 2) {
        std::vector<Vec3> bearing;
        std::vector<Pose3> poses_;
        bearing.reserve(obs.size());
        poses_.reserve(obs.size());
        for (const auto& observation : obs) {
            Image::Ptr im = recon.GetImage(observation.image_id);
            CamModel::Ptr cam = recon.GetCamera(im->CameraId());
            bearing.emplace_back(cam->bearing(im->points_[observation.point2d_id]));
            poses_.emplace_back(im->CameraFromWorld());
        }
        if (bearing.size() > 2) {
            const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, bearing.size());
            Vec4 Xhomogeneous;
            std::vector<Mat34> pose_mats(poses_.size());
            for (int i = 0; i < poses_.size(); ++i) pose_mats[i] = poses_[i].Mat34();
            if (TriangulateNViewAlgebraic(bearing_matrix, pose_mats, &Xhomogeneous)) {
                X = Xhomogeneous.hnormalized();
                return true;
            }
        } else {
            return Triangulate2View(poses_.front().Mat3d(), poses_.front().Trans(), bearing.front(),
                                    poses_.back().Mat3d(), poses_.back().Trans(), bearing.back(), X,
                                    ETriangulationMethod::DIRECT_LINEAR_TRANSFORM);
        }
    }
    return false;
}
inline bool CheiralityTest(const Vec3& bearing, const Pose3& pose, const Vec3& X) {
    return bearing.dot(pose * X) > 0.0;
}
struct ResidualAndCheiralityPredicate {
    const double squared_pixel_threshold_;

    ResidualAndCheiralityPredicate(const double squared_pixel_threshold)
        : squared_pixel_threshold_(squared_pixel_threshold) {}

    bool predicate(const CamModel::Ptr& cam, const Pose3& pose, const Vec2& x, const Vec3& X) {
        const Vec2 residual = cam->project(pose * X) - x;
        return CheiralityTest(cam->bearing(x), pose, X) && residual.squaredNorm() < squared_pixel_threshold_;
    }
};
bool track_check_predicate(
    const std::vector<Observation>& obs, Sfm_Data& sfm_data, const Vec3& X,
    std::function<bool(const CamModel::Ptr&, const Pose3&, const Vec2&, const Vec3&)> predicate) {
    bool visibility = false;  // assume that no observation has been looked yet
    for (const auto& obs_it : obs) {
        const Image::Ptr view = sfm_data.GetImage(obs_it.image_id);
        visibility = true;  // at least an observation is evaluated
        CamModel::Ptr cam = sfm_data.GetCamera(view->CameraId());
        const Pose3 pose = view->CameraFromWorld();
        if (!predicate(cam, pose, view->points_[obs_it.point2d_id], X)) return false;
    }
    return visibility;
}
template <class RandomGeneratorT, typename SamplingType>
inline void UniformSample(const uint32_t num_samples, const uint32_t total_samples, RandomGeneratorT& random_generator,
                          std::vector<SamplingType>* samples) {
    static_assert(std::is_integral<SamplingType>::value, "SamplingType must be an integral type");

    std::uniform_int_distribution<SamplingType> distribution(0, total_samples - 1);
    samples->resize(0);
    while (samples->size() < num_samples) {
        const auto sample = distribution(random_generator);
        bool bFound = false;
        for (size_t j = 0; j < samples->size() && !bFound; ++j) {
            bFound = (*samples)[j] == sample;
        }
        if (!bFound) {
            samples->push_back(sample);
        }
    }
}
std::vector<Observation> ObservationsSampler(const std::vector<Observation>& obs,
                                             const std::vector<std::uint32_t>& samples) {
    std::vector<Observation> sampled_obs;
    for (const auto& idx : samples) {
        sampled_obs.push_back(obs[idx]);
    }
    return sampled_obs;
}
bool RobustTriangulate(Sfm_Data& sfm_data, std::vector<Observation>& obs, Landmark& landmark) {
    int min_required_inliers_ = 3;
    int min_sample_index_ = 3;
    double max_reprojection_error_ = 4.0;
    if (obs.size() < min_required_inliers_ || obs.size() < min_sample_index_) {
        return false;
    }

    const double dSquared_pixel_threshold = max_reprojection_error_ * max_reprojection_error_;

    // Predicate to validate a sample (cheirality and residual error)
    ResidualAndCheiralityPredicate predicate(dSquared_pixel_threshold);
    auto predicate_binding = std::bind(&ResidualAndCheiralityPredicate::predicate, predicate, std::placeholders::_1,
                                       std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    // Handle the case where all observations must be used
    if (min_required_inliers_ == min_sample_index_ && obs.size() == min_required_inliers_) {
        // Generate the 3D point hypothesis by triangulating all the observations
        Vec3 X;
        if (TrackTriangulate(sfm_data, obs, X) && track_check_predicate(obs, sfm_data, X, predicate_binding)) {
            landmark.xyz = X;
            landmark.track = obs;
            return true;
        }
        return false;
    }

    // else we perform a robust estimation since
    //  there is more observations than the minimal number of required sample.

    const int iters = obs.size() * 2;  // TODO: automatic computation of the number of iterations?

    // - Ransac variables
    Vec3 best_model = Vec3::Zero();
    std::vector<Observation> best_inlier_set;
    double best_error = std::numeric_limits<double>::max();

    //--
    // Random number generation
    std::mt19937 random_generator(std::mt19937::default_seed);

    // - Ransac loop
    for (int i = 0; i < iters; ++i) {
        std::vector<uint32_t> samples;
        UniformSample(min_sample_index_, obs.size(), random_generator, &samples);

        Vec3 X;
        // Hypothesis generation
        auto minimal_sample = ObservationsSampler(obs, samples);

        if (!TrackTriangulate(sfm_data, minimal_sample, X)) continue;

        // Test validity of the hypothesis
        if (!track_check_predicate(minimal_sample, sfm_data, X, predicate_binding)) continue;

        std::vector<Observation> inlier_set;
        double current_error = 0.0;

        // inlier/outlier classification according pixel residual errors.
        for (const auto& obs_it : obs) {
            Image::Ptr view = sfm_data.GetImage(obs_it.image_id);
            CamModel::Ptr cam = sfm_data.GetCamera(view->CameraId());
            const Pose3 pose = view->CameraFromWorld();
            if (!CheiralityTest(cam->bearing(view->points_[obs_it.point2d_id]), pose, X)) continue;
            const double residual_sq = (cam->project(pose * X) - view->points_[obs_it.point2d_id]).squaredNorm();
            if (residual_sq < dSquared_pixel_threshold) {
                inlier_set.push_back(obs_it);
                current_error += residual_sq;
            } else {
                current_error += dSquared_pixel_threshold;
            }
        }
        // Does the hypothesis:
        // - is the best one we have seen so far.
        // - has sufficient inliers.
        if (current_error < best_error && inlier_set.size() >= min_required_inliers_) {
            best_model = X;
            best_inlier_set = inlier_set;
            best_error = current_error;
        }
    }
    if (!best_inlier_set.empty() && best_inlier_set.size() >= min_required_inliers_) {
        // Update information (3D landmark position & valid observations)
        landmark.xyz = best_model;
        landmark.track = best_inlier_set;
    }
    return !best_inlier_set.empty();
}
void RobustTriangulate(Sfm_Data& recon) {
    for (auto& [lm_id, landmark] : recon.landmarks_) {
        Landmark lm;
        if (RobustTriangulate(recon, landmark.track, lm)) {
            landmark = lm;
        }
    }
}
namespace std {
template <>
struct hash<Eigen::Vector2i> {
    size_t operator()(const Eigen::Vector2i& s) const {
        using std::hash;
        using std::size_t;
        return ((hash<int64_t>()(s.x()) ^ (hash<int64_t>()(s.y()) << 1)) >> 1);
    }
};
}  // namespace std
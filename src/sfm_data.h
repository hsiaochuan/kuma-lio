#pragma once

#include <boost/filesystem.hpp>
#include <cmath>
#include <memory>
#include <vector>
#include "cameras/cameras.h"
#include "types.h"
using namespace faster_lio;
namespace fs = boost::filesystem;

struct sfm_data {
    void LoadFromDatabase(const std::string& db_path);

    static bool IsNotWhiteSpace(const int character);
    static void StringLeftTrim(std::string* str);
    static void StringRightTrim(std::string* str);
    static void StringTrim(std::string* str);

    void ReadImagesText(std::istream& stream);
    void ReadCamerasText(std::istream& stream);
    void ReadPoints3DText(std::istream& stream);

    void WriteCamerasText(std::ostream& stream);
    void WriteImagesText(std::ostream& stream);
    void WritePoints3DText(std::ostream& stream);
    void WriteCOLMAPText(const std::string& colmap_result_path);
    void LoadFromCOLMAPResult(const std::string& colmap_result_path);

    int FilterOutlier(const double max_reproj_error);
    double MeanTrackLength();
    double CalcMeanError();

    std::unordered_map<camera_t, CamModel::Ptr> cameras_;
    std::unordered_map<camera_t, Image::Ptr> images_;
    std::unordered_map<landmark_t, Landmark> landmarks_;
    std::unordered_map<image_pair_t, TwoViewGeometry> two_view_geometries_;

    CamModel::Ptr& GetCamera(camera_t camera_id);
    Image::Ptr& GetImage(image_t image_id);
    Landmark& GetLandMark(landmark_t point3D_id);
    TwoViewGeometry& GetTwoViewGeometry(image_pair_t image_pair_id);
};
// enum class ETriangulationMethod : unsigned char {
//     DIRECT_LINEAR_TRANSFORM,  // DLT
// };
// void TriangulateDLT(const Mat34& P1, const Vec3& x1, const Mat34& P2, const Vec3& x2, Vec4* X_homogeneous) {
//     // Solve:
//     // [cross(x0,P0) X = 0]
//     // [cross(x1,P1) X = 0]
//     Mat4 design;
//     design.row(0) = x1[0] * P1.row(2) - x1[2] * P1.row(0);
//     design.row(1) = x1[1] * P1.row(2) - x1[2] * P1.row(1);
//     design.row(2) = x2[0] * P2.row(2) - x2[2] * P2.row(0);
//     design.row(3) = x2[1] * P2.row(2) - x2[2] * P2.row(1);
//
//     const Eigen::JacobiSVD<Mat4> svd(design, Eigen::ComputeFullV);
//     (*X_homogeneous) = svd.matrixV().col(3);
// }
// void TriangulateDLT(const Mat34& P1, const Vec3& x1, const Mat34& P2, const Vec3& x2, Vec3* X_euclidean) {
//     Vec4 X_homogeneous;
//     TriangulateDLT(P1, x1, P2, x2, &X_homogeneous);
//     (*X_euclidean) = X_homogeneous.hnormalized();
// }
// bool TriangulateDLT(const Mat3& R0, const Vec3& t0, const Vec3& x0, const Mat3& R1, const Vec3& t1, const Vec3& x1,
//                     Vec3* X) {
//     Mat34 P0, P1;
//     P0.block<3, 3>(0, 0) = R0;
//     P1.block<3, 3>(0, 0) = R1;
//     P0.block<3, 1>(0, 3) = t0;
//     P1.block<3, 1>(0, 3) = t1;
//     TriangulateDLT(P0, x0, P1, x1, X);
//     return x0.dot(R0 * (*X + R0.transpose() * t0)) > 0.0 && x1.dot(R1 * (*X + R1.transpose() * t1)) > 0.0;
// }
// bool Triangulate2View(const Mat3& R0, const Vec3& t0, const Vec3& bearing0, const Mat3& R1, const Vec3& t1,
//                       const Vec3& bearing1, Vec3& X, ETriangulationMethod etri_method) {
//     switch (etri_method) {
//         case ETriangulationMethod::DIRECT_LINEAR_TRANSFORM:
//             return TriangulateDLT(R0, t0, bearing0, R1, t1, bearing1, &X);
//             break;
//         default:
//             return false;
//     }
//     return false;
// }
// double Nullspace(const Eigen::Ref<const Mat>& A, Eigen::Ref<Vec> nullspace) {
//     if (A.rows() >= A.cols()) {
//         Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
//         nullspace = svd.matrixV().col(A.cols() - 1);
//         return svd.singularValues()(A.cols() - 1);
//     }
//     // Extend A with rows of zeros to make it square. It's a hack, but it is
//     // necessary until Eigen supports SVD with more columns than rows.
//     Mat A_extended(A.cols(), A.cols());
//     A_extended.block(A.rows(), 0, A.cols() - A.rows(), A.cols()).setZero();
//     A_extended.block(0, 0, A.rows(), A.cols()) = A;
//     return Nullspace(A_extended, nullspace);
// }
// void TriangulateNView(const Mat3X& x, const std::vector<Mat34>& poses, Vec4* X) {
//     assert(X != nullptr);
//     const Mat2X::Index nviews = x.cols();
//     assert(static_cast<size_t>(nviews) == poses.size());
//
//     Mat A = Mat::Zero(3 * nviews, 4 + nviews);
//     for (Mat::Index i = 0; i < nviews; ++i) {
//         A.block<3, 4>(3 * i, 0) = -poses[i];
//         A.block<3, 1>(3 * i, 4 + i) = x.col(i);
//     }
//     Vec X_and_alphas(4 + nviews);
//     Nullspace(A, X_and_alphas);
//     *X = X_and_alphas.head(4);
// }
//
// bool TriangulateNViewAlgebraic(const Mat3X& points, const std::vector<Mat34>& poses, Vec4* X) {
//     assert(poses.size() == points.cols());
//
//     Mat4 AtA = Mat4::Zero();
//     for (Mat3X::Index i = 0; i < points.cols(); ++i) {
//         const Vec3 point_norm = points.col(i).normalized();
//         const Mat34 cost = poses[i] - point_norm * point_norm.transpose() * poses[i];
//         AtA += cost.transpose() * cost;
//     }
//
//     Eigen::SelfAdjointEigenSolver<Mat4> eigen_solver(AtA);
//     *X = eigen_solver.eigenvectors().col(0);
//     return eigen_solver.info() == Eigen::Success;
// }
// bool TrackTriangulate(Sfm_Data& recon, std::vector<Observation>& obs, Vec3& X) {
//     if (obs.size() >= 2) {
//         std::vector<Vec3> bearing;
//         std::vector<Pose3> poses_;
//         bearing.reserve(obs.size());
//         poses_.reserve(obs.size());
//         for (const auto& observation : obs) {
//             Image::Ptr im = recon.GetImage(observation.image_id);
//             CamModel::Ptr cam = recon.GetCamera(im->CameraId());
//             bearing.emplace_back(cam->bearing(im->points_[observation.point2d_id]));
//             poses_.emplace_back(im->CameraFromWorld());
//         }
//         if (bearing.size() > 2) {
//             const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, bearing.size());
//             Vec4 Xhomogeneous;
//             std::vector<Mat34> pose_mats(poses_.size());
//             for (int i = 0; i < poses_.size(); ++i) pose_mats[i] = poses_[i].Mat34();
//             if (TriangulateNViewAlgebraic(bearing_matrix, pose_mats, &Xhomogeneous)) {
//                 X = Xhomogeneous.hnormalized();
//                 return true;
//             }
//         } else {
//             return Triangulate2View(poses_.front().Mat3d(), poses_.front().Trans(), bearing.front(),
//                                     poses_.back().Mat3d(), poses_.back().Trans(), bearing.back(), X,
//                                     ETriangulationMethod::DIRECT_LINEAR_TRANSFORM);
//         }
//     }
//     return false;
// }
// inline bool CheiralityTest(const Vec3& bearing, const Pose3& pose, const Vec3& X) {
//     return bearing.dot(pose * X) > 0.0;
// }
// struct ResidualAndCheiralityPredicate {
//     const double squared_pixel_threshold_;
//
//     ResidualAndCheiralityPredicate(const double squared_pixel_threshold)
//         : squared_pixel_threshold_(squared_pixel_threshold) {}
//
//     bool predicate(const CamModel::Ptr& cam, const Pose3& pose, const Vec2& x, const Vec3& X) {
//         const Vec2 residual = cam->project(pose * X) - x;
//         return CheiralityTest(cam->bearing(x), pose, X) && residual.squaredNorm() < squared_pixel_threshold_;
//     }
// };
// bool track_check_predicate(
//     const std::vector<Observation>& obs, Sfm_Data& sfm_data, const Vec3& X,
//     std::function<bool(const CamModel::Ptr&, const Pose3&, const Vec2&, const Vec3&)> predicate) {
//     bool visibility = false;  // assume that no observation has been looked yet
//     for (const auto& obs_it : obs) {
//         const Image::Ptr view = sfm_data.GetImage(obs_it.image_id);
//         visibility = true;  // at least an observation is evaluated
//         CamModel::Ptr cam = sfm_data.GetCamera(view->CameraId());
//         const Pose3 pose = view->CameraFromWorld();
//         if (!predicate(cam, pose, view->points_[obs_it.point2d_id], X)) return false;
//     }
//     return visibility;
// }
// template <class RandomGeneratorT, typename SamplingType>
// inline void UniformSample(const uint32_t num_samples, const uint32_t total_samples, RandomGeneratorT& random_generator,
//                           std::vector<SamplingType>* samples) {
//     static_assert(std::is_integral<SamplingType>::value, "SamplingType must be an integral type");
//
//     std::uniform_int_distribution<SamplingType> distribution(0, total_samples - 1);
//     samples->resize(0);
//     while (samples->size() < num_samples) {
//         const auto sample = distribution(random_generator);
//         bool bFound = false;
//         for (size_t j = 0; j < samples->size() && !bFound; ++j) {
//             bFound = (*samples)[j] == sample;
//         }
//         if (!bFound) {
//             samples->push_back(sample);
//         }
//     }
// }
// std::vector<Observation> ObservationsSampler(const std::vector<Observation>& obs,
//                                              const std::vector<std::uint32_t>& samples) {
//     std::vector<Observation> sampled_obs;
//     for (const auto& idx : samples) {
//         sampled_obs.push_back(obs[idx]);
//     }
//     return sampled_obs;
// }
// bool RobustTriangulate(Sfm_Data& sfm_data, std::vector<Observation>& obs, Landmark& landmark) {
//     int min_required_inliers_ = 3;
//     int min_sample_index_ = 3;
//     double max_reprojection_error_ = 4.0;
//     if (obs.size() < min_required_inliers_ || obs.size() < min_sample_index_) {
//         return false;
//     }
//
//     const double dSquared_pixel_threshold = max_reprojection_error_ * max_reprojection_error_;
//
//     // Predicate to validate a sample (cheirality and residual error)
//     ResidualAndCheiralityPredicate predicate(dSquared_pixel_threshold);
//     auto predicate_binding = std::bind(&ResidualAndCheiralityPredicate::predicate, predicate, std::placeholders::_1,
//                                        std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
//
//     // Handle the case where all observations must be used
//     if (min_required_inliers_ == min_sample_index_ && obs.size() == min_required_inliers_) {
//         // Generate the 3D point hypothesis by triangulating all the observations
//         Vec3 X;
//         if (TrackTriangulate(sfm_data, obs, X) && track_check_predicate(obs, sfm_data, X, predicate_binding)) {
//             landmark.xyz = X;
//             landmark.track = obs;
//             return true;
//         }
//         return false;
//     }
//
//     // else we perform a robust estimation since
//     //  there is more observations than the minimal number of required sample.
//
//     const int iters = obs.size() * 2;  // TODO: automatic computation of the number of iterations?
//
//     // - Ransac variables
//     Vec3 best_model = Vec3::Zero();
//     std::vector<Observation> best_inlier_set;
//     double best_error = std::numeric_limits<double>::max();
//
//     //--
//     // Random number generation
//     std::mt19937 random_generator(std::mt19937::default_seed);
//
//     // - Ransac loop
//     for (int i = 0; i < iters; ++i) {
//         std::vector<uint32_t> samples;
//         UniformSample(min_sample_index_, obs.size(), random_generator, &samples);
//
//         Vec3 X;
//         // Hypothesis generation
//         auto minimal_sample = ObservationsSampler(obs, samples);
//
//         if (!TrackTriangulate(sfm_data, minimal_sample, X)) continue;
//
//         // Test validity of the hypothesis
//         if (!track_check_predicate(minimal_sample, sfm_data, X, predicate_binding)) continue;
//
//         std::vector<Observation> inlier_set;
//         double current_error = 0.0;
//
//         // inlier/outlier classification according pixel residual errors.
//         for (const auto& obs_it : obs) {
//             Image::Ptr view = sfm_data.GetImage(obs_it.image_id);
//             CamModel::Ptr cam = sfm_data.GetCamera(view->CameraId());
//             const Pose3 pose = view->CameraFromWorld();
//             if (!CheiralityTest(cam->bearing(view->points_[obs_it.point2d_id]), pose, X)) continue;
//             const double residual_sq = (cam->project(pose * X) - view->points_[obs_it.point2d_id]).squaredNorm();
//             if (residual_sq < dSquared_pixel_threshold) {
//                 inlier_set.push_back(obs_it);
//                 current_error += residual_sq;
//             } else {
//                 current_error += dSquared_pixel_threshold;
//             }
//         }
//         // Does the hypothesis:
//         // - is the best one we have seen so far.
//         // - has sufficient inliers.
//         if (current_error < best_error && inlier_set.size() >= min_required_inliers_) {
//             best_model = X;
//             best_inlier_set = inlier_set;
//             best_error = current_error;
//         }
//     }
//     if (!best_inlier_set.empty() && best_inlier_set.size() >= min_required_inliers_) {
//         // Update information (3D landmark position & valid observations)
//         landmark.xyz = best_model;
//         landmark.track = best_inlier_set;
//     }
//     return !best_inlier_set.empty();
// }
// void RobustTriangulate(Sfm_Data& recon) {
//     for (auto& [lm_id, landmark] : recon.landmarks_) {
//         Landmark lm;
//         if (RobustTriangulate(recon, landmark.track, lm)) {
//             landmark = lm;
//         }
//     }
// }
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
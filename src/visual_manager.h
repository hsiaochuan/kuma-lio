//
// Created by hsiaochuan on 2026/05/11.
//

#ifndef FASTER_LIO_VISUAL_MANAGER_H
#define FASTER_LIO_VISUAL_MANAGER_H
#include "common_lib.h"
#include "laser_mapping_param.h"
#include <opencv2/opencv.hpp>
#include "state_point.h"
#include "ivox3d.h"
namespace faster_lio {

class VisualManager {
public:
    struct PatchObservation {
        using Ptr = std::shared_ptr<PatchObservation>;
        cv::Mat img;
        Vec2 px;
        Vec3 bearing;
        std::vector<float> patch;
        Pose3 camera_from_world;
        int level = 0;
        float score = 0.0;
        float mean = 0.0;
    };
    struct Options {
        Options(){
            half_p_size = patch_size / 2;
            total_patch_size = patch_size * patch_size;
            border = (half_p_size + 1) * (1 << patch_pyrimid_level);
        }
        int patch_pyrimid_level = 3;

        int max_iterations = 5;
        int grid_size = 10;

        int patch_size = 8;
        int half_p_size;
        int total_patch_size;
        int border;
        double visual_voxel_res = 0.2;

        double outlier_thr = 100;

        double img_point_cov = 1000;
    };
    struct VisualPoint {
        using Ptr = std::shared_ptr<VisualPoint>;
        Vec3 xyz;
        Vec3 normal;
        std::list<PatchObservation::Ptr> observation;
        PatchObservation::Ptr ref_obs = nullptr;
        bool is_converged = false;
    };
    struct SparseMap {
        std::vector<std::vector<float>> warp_patch;
        std::vector<VisualPoint::Ptr> visual_points;
        std::vector<int> search_levels;
    };
    enum GridType {
        UNKOWN,
        VISUAL_POINT,
        TYPE_POINTCLOUD,
    };
    std::shared_ptr<LaserMappingParam> param;
    Options options_;
    std::unordered_map<VOXEL_LOCATION, std::vector<VisualPoint::Ptr>> visual_points_map;
    std::shared_ptr<IVox> ivox_;

    StatePoint::Ptr state_point_;
    StatePoint state_predict_;
    SparseMap sparse_map;
    // tmp var
    int grid_n_h;
    int grid_n_w;
    int total_grids;
    std::vector<GridType> grid_states;
    Pose3 camera_from_world;

    // cache jaco
    Mat3 Jdp_dt;
    Mat3 Jdphi_dR;
    Mat3 Jdp_dR;
    void Run(cv::Mat& raw_im, std::vector<PointWithNormal>& scan);
    void Initialize();
    void RetrieveFromSparseMap(cv::Mat& img, std::vector<PointWithNormal>& scan);
    void UpdateEKF(cv::Mat& img);
    void UpdateEKFInLevel(cv::Mat& img, int level);
    void WarpAffine(const Mat2& A_cur_ref, const cv::Mat& img_ref, const Vec2& px, const int search_level, int pyr_level, float * patch);
    Mat2 AffineMatrix(CamModel::Ptr & cam_model, const Vec2 & px, const Vec3& xyz_ref, const Pose3& T_cur_ref, const int & level);
    Mat2 HomographyAffineMatrix(CamModel::Ptr & cam_model, const Vec2& px, const Vec3& xyz_ref, const Vec3& normal_ref, const Pose3& T_cur_ref, const int & level);
    void GenerateVisualPoints(cv::Mat& img, std::vector<PointWithNormal>& scan_points);
    void UpdateVisualPointObserv(cv::Mat& img);
    void UpdateVisualPointNormal();
    void UpdateReferencePatch();
    void PatchFromImage(cv::Mat & img, Vec2 & px, float* patch, int level);
    int GridIdx(const Vec2 & im_point) {
        return static_cast<int>(im_point[1] / options_.grid_size) * grid_n_w + static_cast<int>(im_point[0] / options_.grid_size);
    }
};

}  // namespace faster_lio

#endif  // FASTER_LIO_VISUAL_MANAGER_H

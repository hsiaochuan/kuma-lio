//
// Created by hsiaochuan on 2026/05/11.
//

#ifndef FASTER_LIO_VISUAL_MANAGER_H
#define FASTER_LIO_VISUAL_MANAGER_H
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "cameras/cameras.h"
#include "common_lib.h"
namespace faster_lio {

class LocalBA;

struct VisualParam {
    bool enable = false;
    int max_iterations = 5;          ///< IEKF iterations per pyramid level
    int patch_size = 8;              ///< [px], even
    int patch_pyramid_level = 4;     ///< levels, solved coarse to fine
    int grid_size = 32;              ///< [px] image cell; at most one visual point per cell
    double voxel_size = 0.5;         ///< [m] voxel of the visual point map
    double outlier_threshold = 500;  ///< mean squared intensity error per pixel above which a patch is dropped
    double img_point_cov = 1000;     ///< intensity variance of one patch pixel
    double pixel_std = 0.0;          ///< [px] position error of a visual point in the image (LiDAR map point, extrinsics); caps
                                     ///< what one patch can say about the pose, 0 = pixels independent and unbounded
    double scan_window = 0.3;        ///< [s] of recent LiDAR points that select voxels and seed new visual points
    bool exposure_estimate_en = true; ///< the inverse exposure is a filter state, else fixed at 1
    double inv_expo_cov = 0.01;      ///< [1/s] random-walk variance of the inverse exposure
    int min_points = 10;             ///< below this many tracked points the photometric update is skipped
    // ---- under evaluation ----
    bool converge_en = true;         ///< freeze a point on its reference patch once its normal stops changing
    double max_view_angle = 0.0;     ///< [deg] reject a patch seen (now or in its reference) more obliquely; 0 = off
    double huber_px = 0.0;           ///< Huber threshold of a patch's requested displacement, in units of pixel_std; 0 = off
};


class VisualManager {
   public:
    struct Observation {
        using Ptr = std::shared_ptr<Observation>;
        cv::Mat img;  ///< grayscale frame the patch was taken from, shared between observations
        Vec2 px;
        std::vector<float> patch;  ///< level-0 patch around px
        Pose3 camera_from_world;
        int level = 0;
        float score = 0.0f;
        float mean = 0.0f;
        double inv_expo = 1.0;  ///< relative inverse exposure of the frame
    };
    struct VisualPoint {
        using Ptr = std::shared_ptr<VisualPoint>;
        Vec3 xyz;
        Vec3 normal;
        std::list<Observation::Ptr> observation;  ///< oldest first
        Observation::Ptr ref_obs = nullptr;
        bool is_converged = false;
    };
    VisualManager(const VisualParam& prm, const CamModel::Ptr& cam);
    static cv::Mat ToGray(const cv::Mat& img);
    void Retrieve(const cv::Mat& gray, const Pose3& camera_from_world, const std::vector<PointWithNormal>& scan);
    int NumTracked() const { return static_cast<int>(tracked_.size()); }
    int PatchPixels() const { return patch_total_; }
    const Vec3& TrackedPoint(int i) const { return tracked_[i].vp->xyz; }
    bool PatchResidual(int i, const cv::Mat& gray, const Pose3& camera_from_world, int level, double inv_expo,
                       Eigen::Ref<VecX> res, Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 2>> G,
                       Eigen::Ref<VecX> d_expo, Eigen::Matrix<double, 2, 3>* J_proj) const;
    void UpdateMap(const cv::Mat& gray, const Pose3& camera_from_world, const std::vector<PointWithNormal>& scan,
                   const LocalBA& map);

    size_t MapVoxels() const { return map_.size(); }
    int LastAdded() const { return last_added_; }
    double InvExpo() const { return cur_inv_expo_; }
    void SetInvExpo(double inv_expo) { cur_inv_expo_ = inv_expo; }

   private:
    enum CellType { kUnknown, kMap, kPointCloud };
    struct Tracked {
        VisualPoint::Ptr vp;
        int search_level = 0;
        std::vector<float> warp_patch;
        double ref_inv_expo = 1.0;
        double gain = 1.0;
        bool obs_added = false;
    };

    int CellIdx(const Vec2& px) const {
        return static_cast<int>(px[1] / prm_.grid_size) * grid_w_ + static_cast<int>(px[0] / prm_.grid_size);
    }
    Eigen::Matrix<double, 2, 3> ProjectJacobian(const Vec3& pc) const;
    void PatchFromImage(const cv::Mat& gray, const Vec2& px, float* patch) const;
    Mat2 HomographyAffine(const Vec2& px_ref, const Vec3& xyz_ref, const Vec3& normal_ref, const Pose3& T_cur_ref) const;
    void WarpAffine(const Mat2& A_cur_ref, const cv::Mat& img_ref, const Vec2& px_ref, int search_level, int level,
                    float* patch) const;
    Observation::Ptr MakeObservation(const cv::Mat& gray, const Pose3& camera_from_world, const Vec2& px,
                                     int level) const;
    /// The observation that agrees best with all the others.
    Observation::Ptr SelectReference(const VisualPoint& vp) const;
    void GeneratePoints(const cv::Mat& gray, const Pose3& camera_from_world, const std::vector<PointWithNormal>& scan);
    void UpdateObservations(const cv::Mat& gray, const Pose3& camera_from_world);
    void UpdateReferences(const LocalBA& map);

    VisualParam prm_;
    CamModel::Ptr cam_;
    int half_patch_ = 4;
    int patch_total_ = 64;
    int border_ = 0;
    int grid_w_ = 0;
    int grid_h_ = 0;

    std::unordered_map<VOXEL_LOCATION, std::vector<VisualPoint::Ptr>> map_;
    std::vector<CellType> cell_type_;
    std::vector<Tracked> tracked_;
    double cur_inv_expo_ = 1.0;
    int last_added_ = 0;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_VISUAL_MANAGER_H

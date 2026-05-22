//
// Created by xiang on 2021/9/16.
//

#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <glog/logging.h>
#include <algorithm>
#include <execution>
#include <limits>
#include <list>
#include <thread>

#include "common_lib.h"
#include "eigen_type.h"

namespace faster_lio {
struct hash_vec {
    inline size_t operator()(const Vec3i& v) const {
        return size_t(((v[0]) * 73856093) ^ ((v[1]) * 471943) ^ ((v[2]) * 83492791)) % 10000000;
    }
};

class IVox {
   public:
    using PointType = Point;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    struct Vox {
        PointVector points;
    };
    struct DistPoint {
        double dist = 0;
        const PointVector* points = nullptr;
        std::size_t idx = 0;

        DistPoint() = default;
        DistPoint(double d, const PointVector* p, std::size_t i) : dist(d), points(p), idx(i) {}

        PointType Get() const { return (*points)[idx]; }

        inline bool operator<(const DistPoint& rhs) const { return dist < rhs.dist; }
    };

    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    struct Options {
        float resolution_ = 0.2;                        // ivox resolution
        float inv_resolution_ = 10.0;                   // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();
    }
    void AddPoints(const PointVector& points_to_add);
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0);
    void GenerateNearbyGrids();
    Vec3i Pos2Grid(const Vec3f& pt) const;

    Options options_;
    std::unordered_map<Vec3i, std::list<std::pair<Vec3i, Vox>>::iterator, hash_vec> grids_map_;  // voxel hash map
    std::list<std::pair<Vec3i, Vox>> grids_cache_;                                               // voxel cache
    std::vector<Vec3i> nearby_grids_;                                                            // nearbys
};

// squared distance of two pcl points
inline double distance2(const IVox::PointType& pt1, const IVox::PointType& pt2) {
    Eigen::Vector3f d = pt1.getVector3fMap() - pt2.getVector3fMap();
    return d.squaredNorm();
}

inline bool IVox::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num, double max_range) {
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num * nearby_grids_.size());

    auto key = Pos2Grid(pt.getVector3fMap());

    for (const Vec3i& delta : nearby_grids_) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            const auto& points = iter->second->second.points;
            std::size_t old_size = candidates.size();
            double max_range2 = max_range * max_range;
            for (std::size_t i = 0; i < points.size(); ++i) {
                double d = distance2(points[i], pt);
                if (d < max_range2) {
                    candidates.emplace_back(DistPoint(d, &points, i));
                }
            }
            if (old_size + static_cast<std::size_t>(max_num) < candidates.size()) {
                std::nth_element(candidates.begin() + old_size,
                                 candidates.begin() + old_size + static_cast<std::size_t>(max_num) - 1,
                                 candidates.end());
                candidates.resize(old_size + static_cast<std::size_t>(max_num));
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    if (candidates.size() <= max_num) {
    } else {
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);
    }
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

inline void IVox::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(Vec3i::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {Vec3i(0, 0, 0),  Vec3i(-1, 0, 0), Vec3i(1, 0, 0), Vec3i(0, 1, 0),
                         Vec3i(0, -1, 0), Vec3i(0, 0, -1), Vec3i(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_ = {Vec3i(0, 0, 0),   Vec3i(-1, 0, 0), Vec3i(1, 0, 0),  Vec3i(0, 1, 0),  Vec3i(0, -1, 0),
                         Vec3i(0, 0, -1),  Vec3i(0, 0, 1),  Vec3i(1, 1, 0),  Vec3i(-1, 1, 0), Vec3i(1, -1, 0),
                         Vec3i(-1, -1, 0), Vec3i(1, 0, 1),  Vec3i(-1, 0, 1), Vec3i(1, 0, -1), Vec3i(-1, 0, -1),
                         Vec3i(0, 1, 1),   Vec3i(0, -1, 1), Vec3i(0, 1, -1), Vec3i(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_ = {Vec3i(0, 0, 0),   Vec3i(-1, 0, 0),  Vec3i(1, 0, 0),  Vec3i(0, 1, 0),   Vec3i(0, -1, 0),
                         Vec3i(0, 0, -1),  Vec3i(0, 0, 1),   Vec3i(1, 1, 0),  Vec3i(-1, 1, 0),  Vec3i(1, -1, 0),
                         Vec3i(-1, -1, 0), Vec3i(1, 0, 1),   Vec3i(-1, 0, 1), Vec3i(1, 0, -1),  Vec3i(-1, 0, -1),
                         Vec3i(0, 1, 1),   Vec3i(0, -1, 1),  Vec3i(0, 1, -1), Vec3i(0, -1, -1), Vec3i(1, 1, 1),
                         Vec3i(-1, 1, 1),  Vec3i(1, -1, 1),  Vec3i(1, 1, -1), Vec3i(-1, -1, 1), Vec3i(-1, 1, -1),
                         Vec3i(1, -1, -1), Vec3i(-1, -1, -1)};
    } else {
        LOG(ERROR) << "Unknown nearby_type!";
    }
}

inline void IVox::AddPoints(const PointVector& points_to_add) {
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
        auto key = Pos2Grid(pt.getVector3fMap());

        auto iter = grids_map_.find(key);
        if (iter == grids_map_.end()) {
            grids_cache_.push_front({key, Vox{}});
            grids_map_.insert({key, grids_cache_.begin()});

            grids_cache_.front().second.points.emplace_back(pt);

            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
        } else {
            iter->second->second.points.emplace_back(pt);
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            grids_map_[key] = grids_cache_.begin();
        }
    });
}

inline Eigen::Matrix<int, 3, 1> IVox::Pos2Grid(const Vec3f& pt) const {
    return (pt * options_.inv_resolution_).array().round().cast<int>();
}

}  // namespace faster_lio

#endif

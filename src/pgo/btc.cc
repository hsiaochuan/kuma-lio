#include "pgo/btc.h"

#include <glog/logging.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace faster_lio {
namespace pgo {
namespace {

using Keypoints = std::vector<BtcKeypoint, Eigen::aligned_allocator<BtcKeypoint>>;

struct RefPlane {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 c = Vec3::Zero();
    Vec3 n = Vec3::Zero();
};
using RefPlanes = std::vector<RefPlane, Eigen::aligned_allocator<RefPlane>>;

/// A plane normal has no sign, so either orientation counts as the same.
bool SameNormal(const Vec3& a, const Vec3& b, double tol) { return (a - b).norm() < tol || (a + b).norm() < tol; }

int Find(std::vector<int>& parent, int i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
}

void MakeTree(const std::vector<Vec3>& pts, pcl::KdTreeFLANN<pcl::PointXYZ>* tree) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(pts.size());
    for (const Vec3& p : pts) cloud->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    tree->setInputCloud(cloud);
}

/// Paper IV-B2: plane cells that are coplanar (eq. 2) are merged into one plane (eq. 3), and the
/// `btc_ref_planes` merged planes with the most points become the reference planes.  A cell that
/// merged with nothing is not a candidate, as upstream.
RefPlanes ReferencePlanes(const PlaneCells& cells, const PgoParam& prm) {
    const int n = static_cast<int>(cells.size());
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    for (int i = 0; i < n; ++i) {
        const PlanePoint& a = cells[i].plane;
        for (int j = i + 1; j < n; ++j) {
            const PlanePoint& b = cells[j].plane;
            if (!SameNormal(a.n, b.n, prm.btc_merge_normal)) continue;
            const Vec3 d = b.p - a.p;
            if (std::abs(a.n.dot(d)) >= prm.btc_merge_dist || std::abs(b.n.dot(d)) >= prm.btc_merge_dist) continue;
            parent[Find(parent, i)] = Find(parent, j);
        }
    }

    struct Group {
        PointCluster pc;
        int cells = 0;
        int first = 0;  ///< lowest cell index, a deterministic tie-break
    };
    std::unordered_map<int, Group> groups;
    for (int i = 0; i < n; ++i) {
        auto ins = groups.emplace(Find(parent, i), Group());
        Group& g = ins.first->second;
        if (ins.second) g.first = i;
        g.pc += cells[i].pc;
        ++g.cells;
    }
    std::vector<const Group*> merged;
    for (const auto& kv : groups) {
        if (kv.second.cells >= 2) merged.push_back(&kv.second);
    }
    std::sort(merged.begin(), merged.end(), [](const Group* a, const Group* b) {
        return a->pc.N != b->pc.N ? a->pc.N > b->pc.N : a->first < b->first;
    });

    RefPlanes out;
    for (const Group* g : merged) {
        if (static_cast<int>(out.size()) >= prm.btc_ref_planes) break;
        Eigen::SelfAdjointEigenSolver<Mat3> es(g->pc.Cov());
        RefPlane r;
        r.c = g->pc.Mean();
        r.n = es.eigenvectors().col(0);
        // upstream flips the normal to +z; facing the sensor instead keeps the height band on the
        // side the scene was seen from whichever way the body z axis points
        if (r.n.dot(-r.c) < 0.0) r.n = -r.n;
        out.push_back(r);
    }
    return out;
}

/// Paper IV-B3/4: the height-encoded image of `cloud` on one reference plane, and its keypoints,
/// the strongest pixel of each 5x5 block that is bright enough and not on a line.
void ExtractKeypoints(const std::vector<Vec3>& kf_points, const RefPlane& ref, const PgoParam& prm, Keypoints* out_kps) {
    const int layers = static_cast<int>(std::lround((prm.btc_height_max - prm.btc_height_min) / prm.btc_layer_height));
    CHECK(layers > 0 && layers <= 64) << "btc height band must be 1-64 layers, got " << layers;
    const Vec3 ax = ref.n.unitOrthogonal();
    const Vec3 ay = ref.n.cross(ax);

    struct ProjPoint {
        double u, v;
        int layer;
    };
    std::vector<ProjPoint> proj_points;
    double umin = 1e18, vmin = 1e18, umax = -1e18, vmax = -1e18;
    for (const Vec3& p : kf_points) {
        const Vec3 d = p - ref.c;
        const double h = ref.n.dot(d);
        if (h < prm.btc_height_min || h >= prm.btc_height_max) continue;
        const int layer = std::min(layers - 1, static_cast<int>((h - prm.btc_height_min) / prm.btc_layer_height));
        const ProjPoint pj{ax.dot(d), ay.dot(d), layer};
        umin = std::min(umin, pj.u), umax = std::max(umax, pj.u);
        vmin = std::min(vmin, pj.v), vmax = std::max(vmax, pj.v);
        proj_points.push_back(pj);
    }
    if (proj_points.size() <= 5) return;

    constexpr int kBlock = 5;
    const double res = prm.btc_pixel;
    const int bw = static_cast<int>((umax - umin) / (kBlock * res)) + 1;
    const int bh = static_cast<int>((vmax - vmin) / (kBlock * res)) + 1;
    const int W = bw * kBlock, H = bh * kBlock;
    std::vector<uint64_t> bits(static_cast<size_t>(W) * H, 0);
    std::vector<int> cnt(bits.size(), 0);
    std::vector<double> su(bits.size(), 0.0), sv(bits.size(), 0.0);
    auto at = [H](int x, int y) { return static_cast<size_t>(x) * H + y; };
    for (const ProjPoint& pj : proj_points) {
        const int x = std::min(W - 1, static_cast<int>((pj.u - umin) / res));
        const int y = std::min(H - 1, static_cast<int>((pj.v - vmin) / res));
        const size_t k = at(x, y);
        bits[k] |= uint64_t(1) << pj.layer;
        ++cnt[k];
        su[k] += pj.u;
        sv[k] += pj.v;
    }
    auto intensity = [&](int x, int y) { return __builtin_popcountll(bits[at(x, y)]); };

    static const int kDir[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int bx = 0; bx < bw; ++bx) {
        for (int by = 0; by < bh; ++by) {
            int best = 0, px = -1, py = -1;
            for (int x = bx * kBlock; x < (bx + 1) * kBlock; ++x) {
                for (int y = by * kBlock; y < (by + 1) * kBlock; ++y) {
                    const int I = intensity(x, y);
                    if (I > best) best = I, px = x, py = y;
                }
            }
            if (best < prm.btc_min_bits) continue;
            if (px <= 0 || px >= W - 1 || py <= 0 || py >= H - 1) continue;
            if (prm.btc_line_filter) {
                // a maximum whose neighbours on either side along some direction are nearly as
                // bright is a ridge (a wall seen edge-on), not a point-like structure
                bool line = false;
                for (const auto& d : kDir) {
                    const int I1 = intensity(px + d[0], py + d[1]), I2 = intensity(px - d[0], py - d[1]);
                    const bool hi1 = I1 >= best - 3, hi2 = I2 >= best - 3;
                    const bool half1 = I1 >= 0.5 * best, half2 = I2 >= 0.5 * best;
                    if ((hi1 && (half2 || hi2)) || (hi2 && (half1 || hi1))) line = true;
                }
                if (line) continue;
            }
            const size_t k = at(px, py);
            BtcKeypoint kp;
            // the mean in-plane position of the pixel's points, not its centre: sub-pixel accuracy
            kp.p = ref.c + (su[k] / cnt[k]) * ax + (sv[k] / cnt[k]) * ay;
            kp.bits = bits[k];
            kp.count = best;
            out_kps->push_back(kp);
        }
    }
}

/// A keypoint survives when every other keypoint within btc_nms_radius is weaker, or equally
/// strong and later in the list.
Keypoints SuppressNonMaxima(const Keypoints& kps, const PgoParam& prm) {
    if (kps.empty()) return kps;
    std::vector<Vec3> pts;
    for (const BtcKeypoint& k : kps) pts.push_back(k.p);
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    MakeTree(pts, &tree);
    std::vector<int> idx;
    std::vector<float> d2;
    Keypoints out;
    for (size_t i = 0; i < kps.size(); ++i) {
        bool keep = true;
        tree.radiusSearch(pcl::PointXYZ(pts[i].x(), pts[i].y(), pts[i].z()), prm.btc_nms_radius, idx, d2);
        for (int j : idx) {
            if (j == static_cast<int>(i)) continue;
            if (kps[j].count > kps[i].count || (kps[j].count == kps[i].count && j < static_cast<int>(i))) {
                keep = false;
                break;
            }
        }
        if (keep) out.push_back(kps[i]);
    }
    return out;
}

/// Paper IV-C2/3: a triangle from every keypoint and two of its K nearest, sides within
/// [btc_side_min, btc_side_max], not degenerate, each vertex set once.
std::vector<BtcTriangle> MakeTriangles(const Keypoints& kps, const PgoParam& prm) {
    std::vector<BtcTriangle> out_triangles;
    const int Nkp = static_cast<int>(kps.size());
    if (Nkp < 3) return out_triangles;
    std::vector<Vec3> pts;
    for (const BtcKeypoint& k : kps) pts.push_back(k.p);
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    MakeTree(pts, &tree);
    std::unordered_set<uint64_t> seen;
    std::vector<int> nn_ids;
    std::vector<float> nn_d2;
    for (int i = 0; i < Nkp; ++i) {
        tree.nearestKSearch(pcl::PointXYZ(pts[i].x(), pts[i].y(), pts[i].z()), std::min(prm.btc_near_num, Nkp), nn_ids, nn_d2);
        std::vector<int> nb_ids;
        for (int j : nn_ids) {
            if (j != i)
                nb_ids.push_back(j);
        }
        for (size_t ni = 0; ni < nb_ids.size(); ++ni) {
            for (size_t nj = ni + 1; nj < nb_ids.size(); ++nj) {
                std::array<int, 3> vs{{i, nb_ids[ni], nb_ids[nj]}};
                // side k is the one opposite vertex k
                std::array<std::pair<double, int>, 3> sl_vs{{{(pts[vs[1]] - pts[vs[2]]).norm(), vs[0]},
                                                         {(pts[vs[0]] - pts[vs[2]]).norm(), vs[1]},
                                                         {(pts[vs[0]] - pts[vs[1]]).norm(), vs[2]}}};
                bool ok = true;
                for (const auto& sl_v : sl_vs)
                    ok &= sl_v.first >= prm.btc_side_min && sl_v.first <= prm.btc_side_max;
                if (!ok) continue;
                std::sort(sl_vs.begin(), sl_vs.end());
                if (std::abs(sl_vs[2].first - (sl_vs[0].first + sl_vs[1].first)) < 0.2)
                    continue;  // nearly collinear

                std::array<int, 3> sorted_vs = vs;
                std::sort(sorted_vs.begin(), sorted_vs.end());
                if (!seen.insert((uint64_t(sorted_vs[0]) * Nkp + sorted_vs[1]) * Nkp + sorted_vs[2]).second)
                    continue;

                BtcTriangle tri;
                tri.side_len = Eigen::Vector3f(sl_vs[0].first, sl_vs[1].first, sl_vs[2].first);
                tri.v = {{sl_vs[2].second, sl_vs[1].second, sl_vs[0].second}};
                out_triangles.push_back(tri);
            } // for nb
        } // for nb
    } // for key points
    return out_triangles;
}

/// Paper eq. 6, one vertex.
double BitSimilarity(const BtcKeypoint& a, const BtcKeypoint& b) {
    if (a.count + b.count == 0) return 0.0;
    return 2.0 * __builtin_popcountll(a.bits & b.bits) / (a.count + b.count);
}

/// Least-squares rigid transform with dst = T * src.
Eigen::Isometry3d Kabsch(const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
    Vec3 cs = Vec3::Zero(), cd = Vec3::Zero();
    for (size_t i = 0; i < src.size(); ++i) cs += src[i], cd += dst[i];
    cs /= src.size();
    cd /= dst.size();
    Mat3 Hm = Mat3::Zero();
    for (size_t i = 0; i < src.size(); ++i) Hm += (src[i] - cs) * (dst[i] - cd).transpose();
    Eigen::JacobiSVD<Mat3> svd(Hm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Mat3 U = svd.matrixU(), V = svd.matrixV();
    Mat3 D = Mat3::Identity();
    D(2, 2) = (V * U.transpose()).determinant() < 0 ? -1.0 : 1.0;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = V * D * U.transpose();
    T.translation() = cd - T.linear() * cs;
    return T;
}

}  // namespace

BtcFrame BuildBtcFrame(const std::vector<Vec3>& kf_points, const PlaneCells& cells, const PgoParam& prm) {
    Keypoints kps;
    for (const RefPlane& ref_plane : ReferencePlanes(cells, prm))
        ExtractKeypoints(kf_points, ref_plane, prm, &kps);
    kps = SuppressNonMaxima(kps, prm);
    std::stable_sort(kps.begin(), kps.end(), [](const BtcKeypoint& a, const BtcKeypoint& b) { return a.count > b.count; });
    if (static_cast<int>(kps.size()) > prm.btc_max_keypoints) kps.resize(prm.btc_max_keypoints);

    BtcFrame f;
    f.triangles = MakeTriangles(kps, prm);
    f.keypoints = std::move(kps);
    return f;
}

double PlaneOverlap(const std::vector<PlanePoint>& src, const std::vector<PlanePoint>& tar,
                    const Eigen::Isometry3d& T_tar_src, const PgoParam& prm) {
    if (src.empty() || tar.empty()) return 0.0;
    std::vector<Vec3> centres;
    for (const PlanePoint& p : tar) centres.push_back(p.p);
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    MakeTree(centres, &tree);
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    int hit = 0;
    for (const PlanePoint& sp : src) {
        const Vec3 p = T_tar_src * sp.p;
        const Vec3 n = T_tar_src.linear() * sp.n;
        if (tree.nearestKSearch(pcl::PointXYZ(p.x(), p.y(), p.z()), 1, idx, d2) <= 0) continue;
        const PlanePoint& tp = tar[idx[0]];
        if (SameNormal(n, tp.n, prm.btc_overlap_normal) && std::abs(tp.n.dot(p - tp.p)) < prm.btc_overlap_dist) ++hit;
    }
    return static_cast<double>(hit) / src.size();
}

namespace {
VOXEL_LOCATION SideKey(const Eigen::Vector3f& side, double res) {
    return VOXEL_LOCATION(std::llround(side.x() / res), std::llround(side.y() / res), std::llround(side.z() / res));
}
}  // namespace

void BtcDatabase::Add(BtcFrame btc_f) {
    const int frame_id = size();
    for (size_t ti = 0; ti < btc_f.triangles.size(); ++ti)
        table_[SideKey(btc_f.triangles[ti].side_len, prm_.btc_side_res)].push_back({frame_id, static_cast<int>(ti)});
    btc_frames_.push_back(std::move(btc_f));
}

std::vector<BtcCandidate, Eigen::aligned_allocator<BtcCandidate>> BtcDatabase::Query(
    const BtcFrame& q, const std::function<bool(int)>& eligible) const {
    // paper VI-A: a vote per old triangle with the same sides and the same vertex bit strings
    std::vector<char> ok(btc_frames_.size());
    for (size_t i = 0; i < btc_frames_.size(); ++i) ok[i] = eligible(static_cast<int>(i));
    std::unordered_map<int, std::vector<std::pair<int, int>>> pairs;  // frame -> (query tri, its tri)
    for (size_t qt = 0; qt < q.triangles.size(); ++qt) {
        const BtcTriangle& tq = q.triangles[qt];
        const VOXEL_LOCATION key = SideKey(tq.side_len, prm_.btc_side_res);
        const double tol = tq.side_len.norm() * prm_.btc_side_tol;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto it = table_.find(VOXEL_LOCATION(key.x + dx, key.y + dy, key.z + dz));
                    if (it == table_.end()) continue;
                    for (const Entry& e : it->second) {
                        if (!ok[e.frame]) continue;
                        const BtcFrame& f = btc_frames_[e.frame];
                        const BtcTriangle& tc = f.triangles[e.tri];
                        if ((tc.side_len - tq.side_len).norm() >= tol) continue;
                        double sim = 0.0;
                        for (int k = 0; k < 3; ++k) sim += BitSimilarity(q.keypoints[tq.v[k]], f.keypoints[tc.v[k]]);
                        if (sim / 3.0 <= prm_.btc_similarity) continue;
                        pairs[e.frame].emplace_back(static_cast<int>(qt), e.tri);
                    }
                }
            }
        }
    }

    std::vector<std::pair<int, int>> ranked;  // (votes, frame)
    for (const auto& kv : pairs) {
        if (static_cast<int>(kv.second.size()) >= prm_.btc_min_votes) ranked.emplace_back(kv.second.size(), kv.first);
    }
    std::sort(ranked.begin(), ranked.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    if (static_cast<int>(ranked.size()) > prm_.btc_candidates) ranked.resize(prm_.btc_candidates);

    // paper VI-B: RANSAC over the triangle pairs, each pair a hypothesis from its three vertices
    std::vector<BtcCandidate, Eigen::aligned_allocator<BtcCandidate>> out;
    for (const auto& r : ranked) {
        const BtcFrame& f = btc_frames_[r.second];
        const std::vector<std::pair<int, int>>& P = pairs.at(r.second);
        std::vector<std::array<Vec3, 3>> qv(P.size()), fv(P.size());  // corresponding vertices
        for (size_t i = 0; i < P.size(); ++i) {
            for (int k = 0; k < 3; ++k) {
                qv[i][k] = q.keypoints[q.triangles[P[i].first].v[k]].p;
                fv[i][k] = f.keypoints[f.triangles[P[i].second].v[k]].p;
            }
        }
        auto is_inlier = [&](const Eigen::Isometry3d& T, size_t i) {
            for (int k = 0; k < 3; ++k) {
                if ((T * qv[i][k] - fv[i][k]).norm() >= prm_.btc_ransac_dist) return false;
            }
            return true;
        };
        // at most ~50 hypotheses, evenly spread over the pairs, as upstream
        const size_t step = P.size() / 50 + 1;
        int best = 0;
        Eigen::Isometry3d best_T = Eigen::Isometry3d::Identity();
        for (size_t h = 0; h < P.size(); h += step) {
            const Eigen::Isometry3d T = Kabsch({qv[h].begin(), qv[h].end()}, {fv[h].begin(), fv[h].end()});
            int n = 0;
            for (size_t i = 0; i < P.size(); ++i) n += is_inlier(T, i);
            if (n > best) best = n, best_T = T;
        }
        if (best < prm_.btc_min_inliers) continue;
        std::vector<Vec3> s, d;
        for (size_t i = 0; i < P.size(); ++i) {
            if (!is_inlier(best_T, i)) continue;
            s.insert(s.end(), qv[i].begin(), qv[i].end());
            d.insert(d.end(), fv[i].begin(), fv[i].end());
        }

        BtcCandidate c;
        c.frame_id = r.second;
        c.votes = r.first;
        c.inliers = best;
        c.T = Kabsch(s, d);
        out.push_back(c);
    }
    return out;
}

}  // namespace pgo
}  // namespace faster_lio

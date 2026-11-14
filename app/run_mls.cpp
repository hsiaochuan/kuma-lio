// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>

#include <pcl/common/io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace po = boost::program_options;

namespace {

// The search, the outlier removal and the normals read x y z only (the default point representation of both
// PointXYZ and PointXYZI is the first three floats), so the cloud is PointXYZ (16 bytes, PointXYZI is 32) and the
// intensities are kept apart.
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Search = pcl::search::KdTree<pcl::PointXYZ>;
// PCA normal (x y z, NaN where the fit failed) and curvature of each point
using PcaNormals = std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>;

// boost::progress_display is not thread safe: the loops below advance it once per kProgressBlock points, under a
// critical section, from inside `omp for schedule(dynamic, kProgressBlock)` (whose chunks start at multiples of it).
constexpr long kProgressBlock = 1024;
inline void Tick(boost::progress_display& bar, long i) {
    if (i % kProgressBlock == kProgressBlock - 1) {
#pragma omp critical(progress)
        bar += kProgressBlock;
    }
}

struct RimlsParam {
    int k = 16;                 // neighbours of the per-point radius and of the PCA normal
    double filter_scale = 2.0;  // support radius of point j is filter_scale * h_j
    double sigma_n = 0.75;      // width of the normal refitting weight: 0.5 sharp ... 2 smooth
    int min_refitting_iters = 1;
    int max_refitting_iters = 3;  // 0 or 1 is plain IMLS
    double refitting_threshold = 1e-3;
    int max_projection_iters = 15;
    double projection_accuracy = 1e-4;  // stop when |f| < projection_accuracy * mean radius
};

// The robust implicit MLS surface of a point set with normals (MeshLab's GaelMls::RIMLS).
class Rimls {
   public:
    // `tree` searches `cloud`; `normals` holds the PCA normals.
    Rimls(const Cloud::ConstPtr& cloud, const Search::ConstPtr& tree, const PcaNormals& normals,
          const std::vector<double>& radii, const RimlsParam& prm)
        : cloud_(cloud), tree_(tree), normals_(normals), radii_(radii), prm_(prm) {
        double sum = 0;
        for (double h : radii_) sum += h;
        epsilon_ = prm_.projection_accuracy * sum / std::max<size_t>(radii_.size(), 1);
    }

    // Point i moved onto the surface, with the surface normal there; false (point unchanged, normal the PCA one,
    // NaN if none) if it has no normal or no neighbour.
    bool Project(int i, Eigen::Vector3d* out, Eigen::Vector3d* normal) const {
        const auto& q = cloud_->points[i];
        const Eigen::Vector3d x(q.x, q.y, q.z);
        *out = x;
        *normal = Normal(i);
        if (!HasNormal(i)) return false;
        const double search_radius = 2.0 * prm_.filter_scale * radii_[i];
        Eigen::Vector3d position = x, ref = Normal(i), grad;
        double potential = 0;
        int iters = 0;
        do {
            if (!PotentialAndGradient(position, search_radius, ref, &potential, &grad)) return false;
            const double norm = grad.norm();
            if (norm < 1e-12) break;
            ref = grad / norm;
            position -= ref * potential;
        } while (std::abs(potential) > epsilon_ && ++iters < prm_.max_projection_iters);
        *out = position;
        *normal = ref;
        return true;
    }

   private:
    // PCL leaves NaN where the fit failed (fewer than 3 neighbours, or all of them on a line).
    bool HasNormal(int i) const {
        const Eigen::Vector4f& n = normals_[i];
        return std::isfinite(n.x()) && std::isfinite(n.y()) && std::isfinite(n.z());
    }
    Eigen::Vector3d Normal(int i) const { return normals_[i].head<3>().cast<double>(); }

    // rimls.tpp computePotentialAndGradient: f(x) and its gradient; `ref` orients the neighbours' normals.
    bool PotentialAndGradient(const Eigen::Vector3d& x, double search_radius, const Eigen::Vector3d& ref,
                              double* potential, Eigen::Vector3d* grad) const {
        pcl::PointXYZ qp;
        qp.x = float(x.x());
        qp.y = float(x.y());
        qp.z = float(x.z());
        std::vector<int> ids;
        std::vector<float> d2s;
        if (tree_->radiusSearch(qp, search_radius, ids, d2s) <= 0) return false;

        // mlssurface.tpp computeNeighborhood: spatial weights and their gradients
        struct Sample {
            Eigen::Vector3d diff, normal, grad_w;
            double w;
        };
        std::vector<Sample> samples;
        samples.reserve(ids.size());
        for (size_t s = 0; s < ids.size(); ++s) {
            const int id = ids[s];
            const double scale = 1.0 / (radii_[id] * prm_.filter_scale);
            const double s2 = scale * scale;
            const auto& p = cloud_->points[id];
            const Eigen::Vector3d diff = x - Eigen::Vector3d(p.x, p.y, p.z);
            const double aux = 1.0 - diff.squaredNorm() * s2;
            if (aux <= 0 || !HasNormal(id)) continue;  // outside the support of point id, or no normal
            const Eigen::Vector3d normal = Normal(id);
            Sample smp;
            smp.diff = diff;
            smp.normal = normal.dot(ref) < 0 ? -normal : normal;
            smp.w = aux * aux * aux * aux;
            smp.grad_w = diff * (-2.0 * s2 * 4.0 * aux * aux * aux);
            samples.push_back(smp);
        }
        if (samples.empty()) return false;

        const double inv_sigma2 = 1.0 / (prm_.sigma_n * prm_.sigma_n);
        Eigen::Vector3d g = Eigen::Vector3d::Zero(), prev_g;
        double f = 0;
        int iters = 0;
        do {
            prev_g = g;
            Eigen::Vector3d sum_grad_w = Eigen::Vector3d::Zero(), sum_grad_w_f = Eigen::Vector3d::Zero(),
                            sum_n = Eigen::Vector3d::Zero();
            double sum_w = 0;
            f = 0;
            for (const Sample& smp : samples) {
                const double fi = smp.diff.dot(smp.normal);
                const double refit = iters > 0 ? std::exp(-(smp.normal - prev_g).squaredNorm() * inv_sigma2) : 1.0;
                const double w = smp.w * refit;
                const Eigen::Vector3d gw = smp.grad_w * refit;
                sum_grad_w += gw;
                sum_grad_w_f += gw * fi;
                sum_n += smp.normal * w;
                f += w * fi;
                sum_w += w;
            }
            if (sum_w == 0) return false;
            f /= sum_w;
            g = (-sum_grad_w * f + sum_grad_w_f + sum_n) / sum_w;
            ++iters;
        } while (iters < prm_.min_refitting_iters ||
                 ((g - prev_g).squaredNorm() > prm_.refitting_threshold && iters < prm_.max_refitting_iters));
        *potential = f;
        *grad = g;
        return true;
    }

    Cloud::ConstPtr cloud_;
    Search::ConstPtr tree_;
    const PcaNormals& normals_;
    const std::vector<double>& radii_;
    RimlsParam prm_;
    double epsilon_ = 0;
};

// Binary little-endian PLY with pcl::PointXYZINormal's field names (what pcl::io::savePLYFile writes, so
// pcl::io::loadPLYFile reads it back), written block by block as the points are projected: the whole output cloud
// is never held.  Each vertex is kPlyFloats floats: x y z intensity normal_x normal_y normal_z curvature.
constexpr int kPlyFloats = 8;
std::unique_ptr<std::ofstream> OpenPly(const std::string& fname, size_t num_points) {
    std::unique_ptr<std::ofstream> ofs(new std::ofstream(fname, std::ios::binary));
    if (!*ofs) return nullptr;
    *ofs << "ply\nformat binary_little_endian 1.0\nelement vertex " << num_points << "\n";
    for (const char* name : {"x", "y", "z", "intensity", "normal_x", "normal_y", "normal_z", "curvature"})
        *ofs << "property float " << name << "\n";
    *ofs << "end_header\n";
    return ofs;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_fname, output_fname;
    const RimlsParam prm;
    // statistical outlier removal first: a point goes when its mean distance to its kSorMeanK neighbours exceeds the
    // global mean + kSorStddev standard deviations.  1.0 would cut the top ~16 % of a normal distribution; a LiDAR
    // map's density falls with range, so its sparse far points fail a global mean + k sigma test
    constexpr bool kEnableSor = true;
    constexpr int kSorMeanK = 16;
    constexpr double kSorStddev = 3.0;

    po::options_description desc("run_mls: RIMLS point cloud smoothing");
    // clang-format off
    desc.add_options()
        ("help,h", "show this help")
        ("input,i", po::value<std::string>(&input_fname), "input PLY")
        ("output,o", po::value<std::string>(&output_fname), "output PLY");
// clang-format on
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n\n" << desc << std::endl;
        return 1;
    }
    if (vm.count("help") || input_fname.empty() || output_fname.empty()) {
        std::cerr << desc << std::endl;
        return vm.count("help") ? 0 : 1;
    }

    // 1) load, keep the finite points: x y z in `cloud`, the intensities apart
    Cloud::Ptr cloud(new Cloud);
    std::vector<float> intensity;
    {
        pcl::PointCloud<pcl::PointXYZI> raw;
        if (pcl::io::loadPLYFile(input_fname, raw) != 0) {
            std::cerr << "[ERROR] Failed to load " << input_fname << std::endl;
            return 1;
        }
        long finite = 0;
        for (const auto& p : raw.points)
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) ++finite;
        cloud->reserve(finite);
        intensity.reserve(finite);
        for (const auto& p : raw.points)
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
                intensity.push_back(p.intensity);
            }
    }
    std::cout << "[INFO] Loaded " << cloud->size() << " points from " << input_fname << std::endl;

    // 2) statistical outlier removal.  filter(indices) is what filter(cloud) does before copying the kept points
    // (StatisticalOutlierRemoval::applyFilter); the filter and its search tree go before the copy.
    if (kEnableSor) {
        std::vector<int> kept;
        {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud);
            sor.setMeanK(kSorMeanK);
            sor.setStddevMulThresh(kSorStddev);
            sor.filter(kept);
        }
        Cloud::Ptr filtered(new Cloud);
        pcl::copyPointCloud(*cloud, kept, *filtered);
        std::vector<float> filtered_intensity(kept.size());
        for (size_t j = 0; j < kept.size(); ++j) filtered_intensity[j] = intensity[kept[j]];
        std::cout << "[INFO] SOR kept " << filtered->size() << " of " << cloud->size() << " points ("
                  << 100.0 * (cloud->size() - filtered->size()) / std::max<size_t>(cloud->size(), 1) << " % removed)"
                  << std::endl;
        cloud = filtered;
        intensity.swap(filtered_intensity);
    }
    if (cloud->size() <= size_t(prm.k)) {
        std::cerr << "[ERROR] " << cloud->size() << " points, not more than k = " << prm.k << std::endl;
        return 1;
    }

    // 3) per-point radius (mlssurface.tpp computeVertexRadius) over the k nearest neighbours
    Search::Ptr knn(new Search);
    knn->setInputCloud(cloud);
    const long n = long(cloud->size());
    std::vector<double> radii(n);
    std::cout << "[INFO] point radii" << std::endl;
    boost::progress_display bar_radius(n);
#pragma omp parallel
    {
        std::vector<int> ids;
        std::vector<float> d2s;
#pragma omp for schedule(dynamic, kProgressBlock)
        for (long i = 0; i < n; ++i) {
            const int found = knn->nearestKSearch(cloud->points[i], prm.k, ids, d2s);
            const double d2_max = found > 0 ? *std::max_element(d2s.begin(), d2s.begin() + found) : 0.0;
            radii[i] = 2.0 * std::sqrt(d2_max / std::max(found, 1));
            Tick(bar_radius, i);
        }
    }
    bar_radius += n % kProgressBlock;  // the last, partial block

    // Steps 4 and 5 go block by block (a point's normal and projection do not depend on the block), so only one
    // block of the pcl normal cloud and of the output is ever held.
    constexpr long kBlock = 1L << 20;  // a multiple of kProgressBlock, so Tick still counts whole blocks
    static_assert(kBlock % kProgressBlock == 0, "kBlock");

    // 4) PCA normal over the same k neighbours; its sign is fixed per neighbourhood in Rimls
    std::cout << "[INFO] normals (pcl::NormalEstimationOMP)" << std::endl;
    PcaNormals pca(n);
    {
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setNumberOfThreads(omp_get_max_threads());
        ne.setInputCloud(cloud);
        ne.setSearchMethod(knn);  // the same tree: Feature::initCompute keeps it, its input is already `cloud`
        ne.setKSearch(prm.k);
        pcl::IndicesPtr block(new std::vector<int>);
        pcl::PointCloud<pcl::Normal> block_normals;
        for (long begin = 0; begin < n; begin += kBlock) {
            const long end = std::min(n, begin + kBlock);
            block->resize(end - begin);
            for (long i = begin; i < end; ++i) (*block)[i - begin] = int(i);
            ne.setIndices(block);
            ne.compute(block_normals);
            for (long i = begin; i < end; ++i) {
                const pcl::Normal& nm = block_normals.points[i - begin];
                pca[i] = Eigen::Vector4f(nm.normal_x, nm.normal_y, nm.normal_z, nm.curvature);
            }
        }
    }
    long no_normal = 0;
    for (const Eigen::Vector4f& nm : pca)
        if (!std::isfinite(nm.x()) || !std::isfinite(nm.y()) || !std::isfinite(nm.z())) ++no_normal;
    if (no_normal > 0) std::cout << "[INFO] " << no_normal << " points without a normal are left in place" << std::endl;

    // 5) RIMLS projection of every point, written out block by block.  RIMLS reads only the input points and the
    // PCA normals, never the projected points, so a block can go to the file as soon as it is done.
    std::unique_ptr<std::ofstream> ofs = OpenPly(output_fname, size_t(n));
    if (!ofs) {
        std::cerr << "[ERROR] Failed to save " << output_fname << std::endl;
        return 1;
    }
    long failed = 0;
    {
        const Rimls rimls(cloud, knn, pca, radii, prm);
        std::cout << "[INFO] RIMLS projection" << std::endl;
        boost::progress_display bar_project(n);
        std::vector<float> buf(size_t(kBlock) * kPlyFloats);
        for (long begin = 0; begin < n; begin += kBlock) {
            const long end = std::min(n, begin + kBlock);
#pragma omp parallel for schedule(dynamic, kProgressBlock) reduction(+ : failed)
            for (long i = begin; i < end; ++i) {
                Eigen::Vector3d p, nrm;
                if (!rimls.Project(int(i), &p, &nrm)) ++failed;
                float* o = &buf[size_t(i - begin) * kPlyFloats];
                o[0] = float(p.x());
                o[1] = float(p.y());
                o[2] = float(p.z());
                o[3] = intensity[i];
                o[4] = float(nrm.x());
                o[5] = float(nrm.y());
                o[6] = float(nrm.z());
                o[7] = pca[i].w();  // the PCA curvature
                Tick(bar_project, i);
            }
            ofs->write(reinterpret_cast<const char*>(buf.data()),
                       std::streamsize(size_t(end - begin) * kPlyFloats * sizeof(float)));
        }
        bar_project += n % kProgressBlock;
    }
    if (failed > 0)
        std::cout << "[INFO] " << failed << " points without a normal or a neighbour were left in place" << std::endl;

    ofs->close();
    if (!*ofs) {
        std::cerr << "[ERROR] Failed to save " << output_fname << std::endl;
        return 1;
    }
    std::cout << "[INFO] Smoothed cloud saved to " << output_fname << " (" << n << " points)." << std::endl;
    return 0;
}

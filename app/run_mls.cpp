// run_mls.cpp
//
// Point cloud smoothing by RIMLS projection: every point is moved onto the robust implicit MLS surface of the
// cloud (Oztireli, Guennebaud, Gross, "Feature Preserving Point Set Surfaces based on Non-Linear Kernel
// Regression", Eurographics 2009), after MeshLab's "MLS projection (RIMLS)" filter
// (src/meshlabplugins/filter_mls/rimls.tpp and mlssurface.tpp of https://github.com/cnr-isti-vclab/meshlab).
//
//   run_mls -i map.ply -o map_post.ply [--filter_scale 2] [--sigma_n 0.75] ...  (-h for help)
//
// What is the same as MeshLab: the per-point radius h = 2 sqrt(d2_max / k) over the k nearest neighbours, the
// spatial weight (1 - d^2 / (scale h_j)^2)^4 and its gradient, the refitting loop with the normal weight
// exp(-|n_j - grad|^2 / sigma_n^2), and the projection x <- x - f(x) grad / |grad|.
// What differs:
//  - MeshLab takes the normals of the input mesh; a LiDAR map has none, so they are fitted by PCA over the same
//    k neighbours (pcl::NormalEstimationOMP).  Their sign is arbitrary, and RIMLS only needs it consistent inside one
//    neighbourhood, so each neighbour's normal is flipped to agree with the current surface direction before it is
//    used;
//  - MeshLab's ball tree returns every point j with |x - p_j| < scale h_j.  Here a radius search of
//    2 scale h_i around the point i being projected is filtered by that test, so a neighbour whose radius is more
//    than twice h_i can be missed;
//  - the projection threshold is scaled by the mean point radius (MeshLab leaves mAveragePointSpacing unset);
//  - points are projected in parallel, and intensity is kept;
//  - the output is pcl::PointXYZINormal: the normal is the RIMLS gradient direction at the projected point (the
//    PCA normal where the projection failed, NaN where that failed too), curvature is PCA's surface variation.
//

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace po = boost::program_options;

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZI>;
using Normals = pcl::PointCloud<pcl::Normal>;

/// boost::progress_display is not thread safe: the loops below advance it once per kProgressBlock points, under a
/// critical section, from inside `omp for schedule(dynamic, kProgressBlock)` (whose chunks start at multiples of it).
constexpr long kProgressBlock = 1024;
inline void Tick(boost::progress_display& bar, long i) {
    if (i % kProgressBlock == kProgressBlock - 1) {
#pragma omp critical(progress)
        bar += kProgressBlock;
    }
}

struct RimlsParam {
    int k = 16;                 ///< neighbours of the per-point radius and of the PCA normal
    double filter_scale = 2.0;  ///< support radius of point j is filter_scale * h_j
    double sigma_n = 0.75;      ///< width of the normal refitting weight: 0.5 sharp ... 2 smooth
    int min_refitting_iters = 1;
    int max_refitting_iters = 3;  ///< 0 or 1 is plain IMLS
    double refitting_threshold = 1e-3;
    int max_projection_iters = 15;
    double projection_accuracy = 1e-4;  ///< stop when |f| < projection_accuracy * mean radius
};

/// The robust implicit MLS surface of a point set with normals (MeshLab's GaelMls::RIMLS).
class Rimls {
   public:
    Rimls(const Cloud::ConstPtr& cloud, const Normals::ConstPtr& normals, const std::vector<double>& radii,
          const RimlsParam& prm)
        : cloud_(cloud), normals_(normals), radii_(radii), prm_(prm) {
        tree_.setInputCloud(cloud_);
        double sum = 0;
        for (double h : radii_) sum += h;
        epsilon_ = prm_.projection_accuracy * sum / std::max<size_t>(radii_.size(), 1);
    }

    /// Point i moved onto the surface, with the surface normal there; false (point unchanged, normal the PCA one,
    /// NaN if none) if it has no normal or no neighbour.
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
    /// PCL leaves NaN where the fit failed (fewer than 3 neighbours, or all of them on a line).
    bool HasNormal(int i) const {
        const pcl::Normal& n = normals_->points[i];
        return std::isfinite(n.normal_x) && std::isfinite(n.normal_y) && std::isfinite(n.normal_z);
    }
    Eigen::Vector3d Normal(int i) const { return normals_->points[i].getNormalVector3fMap().cast<double>(); }

    /// rimls.tpp computePotentialAndGradient: f(x) and its gradient; `ref` orients the neighbours' normals.
    bool PotentialAndGradient(const Eigen::Vector3d& x, double search_radius, const Eigen::Vector3d& ref,
                              double* potential, Eigen::Vector3d* grad) const {
        pcl::PointXYZI qp;
        qp.x = float(x.x());
        qp.y = float(x.y());
        qp.z = float(x.z());
        std::vector<int> ids;
        std::vector<float> d2s;
        if (tree_.radiusSearch(qp, search_radius, ids, d2s) <= 0) return false;

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
    Normals::ConstPtr normals_;
    const std::vector<double>& radii_;
    RimlsParam prm_;
    pcl::KdTreeFLANN<pcl::PointXYZI> tree_;
    double epsilon_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::string input_fname, output_fname;
    RimlsParam prm;
    bool enable_sor = true;
    int sor_mean_k = 16;
    double sor_stddev = 3.0;  // was 1.0, which cuts the top ~16 % of a normal distribution; a LiDAR map's density
                              // falls with range, so its sparse far points fail a global mean + k sigma test

    po::options_description desc("run_mls: RIMLS point cloud smoothing");
    // clang-format off
    desc.add_options()
        ("help,h", "show this help")
        ("input,i", po::value<std::string>(&input_fname), "input PLY")
        ("output,o", po::value<std::string>(&output_fname), "output PLY")
        ("k,k", po::value<int>(&prm.k)->default_value(prm.k), "neighbours of the point radius and the PCA normal")
        ("filter_scale", po::value<double>(&prm.filter_scale)->default_value(prm.filter_scale),
         "spatial filter scale, relative to the point radius")
        ("sigma_n", po::value<double>(&prm.sigma_n)->default_value(prm.sigma_n),
         "sharpness: width of the normal refitting weight, 0.5 sharp ... 2 smooth")
        ("max_refitting_iters", po::value<int>(&prm.max_refitting_iters)->default_value(prm.max_refitting_iters),
         "max fitting iterations, 0 or 1 is plain IMLS")
        ("max_projection_iters", po::value<int>(&prm.max_projection_iters)->default_value(prm.max_projection_iters),
         "max projection iterations")
        ("projection_accuracy", po::value<double>(&prm.projection_accuracy)->default_value(prm.projection_accuracy),
         "projection stops at this fraction of the mean point radius")
        ("sor", po::value<bool>(&enable_sor)->default_value(enable_sor), "statistical outlier removal first")
        ("sor-mean-k", po::value<int>(&sor_mean_k)->default_value(sor_mean_k), "SOR mean K")
        ("sor-stddev", po::value<double>(&sor_stddev)->default_value(sor_stddev), "SOR: a point goes when its mean distance to its sor-mean-k neighbours exceeds the global mean + this many standard deviations; larger removes fewer");
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

    // 1) load, keep the finite points
    Cloud::Ptr raw(new Cloud);
    if (pcl::io::loadPLYFile(input_fname, *raw) != 0) {
        std::cerr << "[ERROR] Failed to load " << input_fname << std::endl;
        return 1;
    }
    Cloud::Ptr cloud(new Cloud);
    cloud->reserve(raw->size());
    for (const auto& p : raw->points)
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) cloud->push_back(p);
    raw.reset();
    std::cout << "[INFO] Loaded " << cloud->size() << " points from " << input_fname << std::endl;

    // 2) statistical outlier removal
    if (enable_sor) {
        Cloud::Ptr filtered(new Cloud);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(sor_mean_k);
        sor.setStddevMulThresh(sor_stddev);
        sor.filter(*filtered);
        std::cout << "[INFO] SOR kept " << filtered->size() << " of " << cloud->size() << " points ("
                  << 100.0 * (cloud->size() - filtered->size()) / std::max<size_t>(cloud->size(), 1) << " % removed)"
                  << std::endl;
        cloud = filtered;
    }
    if (cloud->size() <= size_t(prm.k)) {
        std::cerr << "[ERROR] " << cloud->size() << " points, not more than --k " << prm.k << std::endl;
        return 1;
    }

    // 3) per-point radius (mlssurface.tpp computeVertexRadius) over the k nearest neighbours
    pcl::search::KdTree<pcl::PointXYZI>::Ptr knn(new pcl::search::KdTree<pcl::PointXYZI>);
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

    // 4) PCA normal over the same k neighbours; its sign is fixed per neighbourhood in Rimls
    std::cout << "[INFO] normals (pcl::NormalEstimationOMP)" << std::endl;
    Normals::Ptr normals(new Normals);
    {
        pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
        ne.setNumberOfThreads(omp_get_max_threads());
        ne.setInputCloud(cloud);
        ne.setSearchMethod(knn);
        ne.setKSearch(prm.k);
        ne.compute(*normals);
    }
    long no_normal = 0;
    for (const pcl::Normal& nm : normals->points)
        if (!std::isfinite(nm.normal_x) || !std::isfinite(nm.normal_y) || !std::isfinite(nm.normal_z)) ++no_normal;
    if (no_normal > 0) std::cout << "[INFO] " << no_normal << " points without a normal are left in place" << std::endl;

    // 5) RIMLS projection of every point
    const Rimls rimls(cloud, normals, radii, prm);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr out(new pcl::PointCloud<pcl::PointXYZINormal>);
    out->resize(n);
    long failed = 0;
    std::cout << "[INFO] RIMLS projection" << std::endl;
    boost::progress_display bar_project(n);
#pragma omp parallel for schedule(dynamic, kProgressBlock) reduction(+ : failed)
    for (long i = 0; i < n; ++i) {
        Eigen::Vector3d p, nrm;
        if (!rimls.Project(int(i), &p, &nrm)) ++failed;
        pcl::PointXYZINormal& o = out->points[i];
        o.x = float(p.x());
        o.y = float(p.y());
        o.z = float(p.z());
        o.intensity = cloud->points[i].intensity;
        o.normal_x = float(nrm.x());
        o.normal_y = float(nrm.y());
        o.normal_z = float(nrm.z());
        o.curvature = normals->points[i].curvature;
        Tick(bar_project, i);
    }
    bar_project += n % kProgressBlock;
    out->width = std::uint32_t(out->size());
    out->height = 1;
    out->is_dense = no_normal == 0;  // a NaN normal makes the cloud not dense
    if (failed > 0)
        std::cout << "[INFO] " << failed << " points without a normal or a neighbour were left in place" << std::endl;

    if (pcl::io::savePLYFileBinary(output_fname, *out) != 0) {
        std::cerr << "[ERROR] Failed to save " << output_fname << std::endl;
        return 1;
    }
    std::cout << "[INFO] Smoothed cloud saved to " << output_fname << " (" << out->size() << " points)." << std::endl;
    return 0;
}

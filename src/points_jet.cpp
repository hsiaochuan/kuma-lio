// points_jet.cpp
//
// Point cloud smoothing by RIMLS projection: every point is moved onto the robust implicit MLS surface of the
// cloud (Oztireli, Guennebaud, Gross, "Feature Preserving Point Set Surfaces based on Non-Linear Kernel
// Regression", Eurographics 2009), after MeshLab's "MLS projection (RIMLS)" filter
// (src/meshlabplugins/filter_mls/rimls.tpp and mlssurface.tpp of https://github.com/cnr-isti-vclab/meshlab).
//
//   points_jet -i map.ply -o map_post.ply [--filter_scale 2] [--sigma_n 0.75] ...  (-h for help)
//
// What is the same as MeshLab: the per-point radius h = 2 sqrt(d2_max / k) over the k nearest neighbours, the
// spatial weight (1 - d^2 / (scale h_j)^2)^4 and its gradient, the refitting loop with the normal weight
// exp(-|n_j - grad|^2 / sigma_n^2), and the projection x <- x - f(x) grad / |grad|.
// What differs:
//  - MeshLab takes the normals of the input mesh; a LiDAR map has none, so they are fitted by PCA over the same
//    k neighbours.  Their sign is arbitrary, and RIMLS only needs it consistent inside one neighbourhood, so each
//    neighbour's normal is flipped to agree with the current surface direction before it is used;
//  - MeshLab's ball tree returns every point j with |x - p_j| < scale h_j.  Here a radius search of
//    2 scale h_i around the point i being projected is filtered by that test, so a neighbour whose radius is more
//    than twice h_i can be missed;
//  - the projection threshold is scaled by the mean point radius (MeshLab leaves mAveragePointSpacing unset);
//  - points are projected in parallel, and intensity is kept.
//

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/program_options.hpp>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace po = boost::program_options;

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZI>;

struct RimlsParam {
    int k = 16;                       ///< neighbours of the per-point radius and of the PCA normal
    double filter_scale = 2.0;        ///< support radius of point j is filter_scale * h_j
    double sigma_n = 0.75;            ///< width of the normal refitting weight: 0.5 sharp ... 2 smooth
    int min_refitting_iters = 1;
    int max_refitting_iters = 3;      ///< 0 or 1 is plain IMLS
    double refitting_threshold = 1e-3;
    int max_projection_iters = 15;
    double projection_accuracy = 1e-4;  ///< stop when |f| < projection_accuracy * mean radius
};

/// The robust implicit MLS surface of a point set with normals (MeshLab's GaelMls::RIMLS).
class Rimls {
   public:
    Rimls(const Cloud::ConstPtr& cloud, const std::vector<Eigen::Vector3d>& normals, const std::vector<double>& radii,
          const RimlsParam& prm)
        : cloud_(cloud), normals_(normals), radii_(radii), prm_(prm) {
        tree_.setInputCloud(cloud_);
        double sum = 0;
        for (double h : radii_) sum += h;
        epsilon_ = prm_.projection_accuracy * sum / std::max<size_t>(radii_.size(), 1);
    }

    /// Point i moved onto the surface; false (and the point unchanged) if it has no neighbour.
    bool Project(int i, Eigen::Vector3d* out) const {
        const auto& q = cloud_->points[i];
        const Eigen::Vector3d x(q.x, q.y, q.z);
        const double search_radius = 2.0 * prm_.filter_scale * radii_[i];
        Eigen::Vector3d position = x, ref = normals_[i], grad;
        double potential = 0;
        int iters = 0;
        do {
            if (!PotentialAndGradient(position, search_radius, ref, &potential, &grad)) {
                *out = x;
                return false;
            }
            const double norm = grad.norm();
            if (norm < 1e-12) break;
            ref = grad / norm;
            position -= ref * potential;
        } while (std::abs(potential) > epsilon_ && ++iters < prm_.max_projection_iters);
        *out = position;
        return true;
    }

   private:
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
            if (aux <= 0) continue;  // outside the support of point id
            Sample smp;
            smp.diff = diff;
            smp.normal = normals_[id].dot(ref) < 0 ? -normals_[id] : normals_[id];
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
    const std::vector<Eigen::Vector3d>& normals_;
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
    double sor_stddev = 1.0;

    po::options_description desc("points_jet: RIMLS point cloud smoothing");
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
        ("sor-stddev", po::value<double>(&sor_stddev)->default_value(sor_stddev), "SOR stddev multiplier");
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
        cloud = filtered;
        std::cout << "[INFO] SOR kept " << cloud->size() << " points" << std::endl;
    }
    if (cloud->size() <= size_t(prm.k)) {
        std::cerr << "[ERROR] " << cloud->size() << " points, not more than --k " << prm.k << std::endl;
        return 1;
    }

    // 3) per-point radius (mlssurface.tpp computeVertexRadius) and PCA normal over the k nearest neighbours
    pcl::KdTreeFLANN<pcl::PointXYZI> knn;
    knn.setInputCloud(cloud);
    const long n = long(cloud->size());
    std::vector<double> radii(n);
    std::vector<Eigen::Vector3d> normals(n);
#pragma omp parallel
    {
        std::vector<int> ids;
        std::vector<float> d2s;
#pragma omp for schedule(dynamic, 1024)
        for (long i = 0; i < n; ++i) {
            const int found = knn.nearestKSearch(cloud->points[i], prm.k, ids, d2s);
            const double d2_max = found > 0 ? *std::max_element(d2s.begin(), d2s.begin() + found) : 0.0;
            radii[i] = 2.0 * std::sqrt(d2_max / std::max(found, 1));
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            for (int s = 0; s < found; ++s) mean += cloud->points[ids[s]].getVector3fMap().cast<double>();
            mean /= std::max(found, 1);
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (int s = 0; s < found; ++s) {
                const Eigen::Vector3d d = cloud->points[ids[s]].getVector3fMap().cast<double>() - mean;
                cov += d * d.transpose();
            }
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
            normals[i] = es.eigenvectors().col(0);  // smallest eigenvalue; the sign is fixed per neighbourhood later
        }
    }

    // 4) RIMLS projection of every point
    const Rimls rimls(cloud, normals, radii, prm);
    Cloud::Ptr out(new Cloud(*cloud));
    long failed = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : failed)
    for (long i = 0; i < n; ++i) {
        Eigen::Vector3d p;
        if (!rimls.Project(int(i), &p)) ++failed;
        out->points[i].x = float(p.x());
        out->points[i].y = float(p.y());
        out->points[i].z = float(p.z());
    }
    out->width = std::uint32_t(out->size());
    out->height = 1;
    out->is_dense = true;
    if (failed > 0) std::cout << "[INFO] " << failed << " points without a neighbour were left in place" << std::endl;

    if (pcl::io::savePLYFileBinary(output_fname, *out) != 0) {
        std::cerr << "[ERROR] Failed to save " << output_fname << std::endl;
        return 1;
    }
    std::cout << "[INFO] Smoothed cloud saved to " << output_fname << " (" << out->size() << " points)." << std::endl;
    return 0;
}

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

#pragma once

#include <glog/logging.h>

#include "ba/plane_features.h"

namespace kuma_lio {
namespace ba {

using VecVec6 = Eigen::aligned_vector<Vec6>;
using VecMat6 = Eigen::aligned_vector<Mat6>;
// The poses seen by the planes, by dense index (the order of BundleAdjuster::ids)
using DensePoses = Eigen::aligned_vector<Eigen::Isometry3d>;

// T |+ dT, paper eq. (10).
inline Eigen::Isometry3d Boxplus(const Eigen::Isometry3d& x, const Vec6& d) {
    const Mat3 dR = ExpMat(Vec3(d.head<3>()));
    Eigen::Isometry3d y = Eigen::Isometry3d::Identity();
    y.linear() = dR * x.linear();
    y.translation() = dR * x.translation() + d.tail<3>();
    return y;
}

struct BAOptions {
    int max_outer = 300;  // k_max: surrogate rebuilds
    int max_inner = 5;    // j_max: LM steps per surrogate (the paper caps this)
    double eps = 1e-5;    // epsilon of Algorithm 1, on || dT || and || T^k - T^(k-1) ||
    bool verbose = true;
};

// A plane frozen at the current iterate: u = u_min^(k), q = u^T mean^(k), N = N_i.
struct PlaneAnchor {
    Vec3 u = Vec3::Zero();
    double q = 0.0;
    int N = 0;
};

// The block-diagonal normal equations of the surrogate and its cost; J[j], H[j] are the 6x1 / 6x6 blocks of
// pose j.  Summed across threads by an OpenMP reduction.
struct NormalEq {
    VecVec6 J;
    VecMat6 H;
    double cost = 0.0;

    explicit NormalEq(size_t nx) : J(nx, Vec6::Zero()), H(nx, Mat6::Zero()) {}
    void Add(const NormalEq& o) {
        for (size_t j = 0; j < J.size(); j++) {
            J[j] += o.J[j];
            H[j] += o.H[j];
        }
        cost += o.cost;
    }
};
#pragma omp declare reduction(neq_sum : NormalEq : omp_out.Add(omp_in)) initializer(omp_priv = NormalEq(omp_orig.J.size()))

class BundleAdjuster {
   public:
    std::vector<PlaneFeature> feats;
    int Nx = 0;                // number of poses seen by the planes
    std::vector<IndexT> ids;   // dense index (0..Nx-1) -> pose id
    BAOptions opt;

    // Take the plane features (whose clusters are indexed by pose id).  The poses seen by at least one
    // plane are numbered 0..Nx-1; a pose seen by no plane is not optimized and stays where it is.
    void Load(std::vector<PlaneFeature>&& planes) {
        feats.clear();
        feats.reserve(planes.size());
        for (PlaneFeature& pf : planes) {
            if (pf.clus.size() < 2) continue;  // one scan alone cannot move relative to anything
            pf.coeff = 0;
            for (const FrameCluster& o : pf.clus) pf.coeff += o.clu.N;
            feats.push_back(std::move(pf));
        }
        planes.clear();

        ids.clear();
        for (const PlaneFeature& f : feats)
            for (const FrameCluster& o : f.clus) ids.push_back(o.xid);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        Nx = (int)ids.size();
        for (PlaneFeature& f : feats)
            for (FrameCluster& o : f.clus) o.xid = (IndexT)(std::lower_bound(ids.begin(), ids.end(), o.xid) - ids.begin());
    }

    size_t NumFeatures() const { return feats.size(); }

    size_t NumObservations() const {
        size_t n = 0;
        for (const PlaneFeature& f : feats) n += f.clus.size();
        return n;
    }

    // How many plane features each pose (in the order of ids[]) participates in.
    std::vector<int> PlanesPerPose() const {
        std::vector<int> cnt(Nx, 0);
        for (const PlaneFeature& f : feats)
            for (const FrameCluster& o : f.clus) cnt[o.xid]++;
        return cnt;
    }

    // Optimize, in place, the poses seen by the planes (the others are untouched).
    void Optimize(Poses& poses) {
        Vec3 shift;
        DensePoses xs = Gather(poses, &shift);
        const DensePoses x_init = xs;
        OptimizeDense(xs);

        if (opt.verbose && Nx > 0) {
            double max_dt = 0, max_dr = 0, sum_dt = 0;
            for (int i = 0; i < Nx; i++) {
                const double dt = (xs[i].translation() - x_init[i].translation()).norm();
                const double dr = LogMat(Mat3(x_init[i].linear().transpose() * xs[i].linear())).norm() * 180.0 / M_PI;
                max_dt = std::max(max_dt, dt);
                max_dr = std::max(max_dr, dr);
                sum_dt += dt;
            }
            LOG(INFO) << "[ba] pose correction: mean |dt| " << sum_dt / Nx << " m, max |dt| " << max_dt << " m, max |dR| "
                      << max_dr << " deg";
        }
        for (int i = 0; i < Nx; i++) {
            xs[i].translation() += shift;
            poses.at(ids[i]) = xs[i];
        }
    }

   private:
    // The poses seen by the planes, densely (the order of ids[]), the translation centered on their mean
    // for numerical conditioning; `shift` is the vector to add back.
    DensePoses Gather(const Poses& poses, Vec3* shift) const {
        DensePoses xs(Nx);
        *shift = Vec3::Zero();
        for (int i = 0; i < Nx; i++) {
            xs[i] = poses.at(ids[i]);
            *shift += xs[i].translation();
        }
        if (Nx > 0) *shift /= Nx;
        for (Eigen::Isometry3d& x : xs) x.translation() -= *shift;
        return xs;
    }

    // The world-frame cluster of one feature at poses xs.
    static void Accumulate(const PlaneFeature& fea, const DensePoses& xs, PointCluster* total) {
        total->SetZero();
        PointCluster tr;
        for (const FrameCluster& o : fea.clus) {
            tr.Transform(o.clu, xs[o.xid].linear(), xs[o.xid].translation());
            *total += tr;
        }
    }

    // The plane of feature `a` at poses xs (u and q only if `axes`); returns lambda_min.
    double EvalPlane(long a, const DensePoses& xs, bool axes, PlaneAnchor* plane) const {
        PointCluster sig;
        Accumulate(feats[a], xs, &sig);
        *plane = PlaneAnchor();
        if (sig.N <= 0) return 0.0;
        const Eigen::SelfAdjointEigenSolver<Mat3> saes(sig.Cov(), axes ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
        plane->N = sig.N;
        if (axes) {
            plane->u = saes.eigenvectors().col(0);
            plane->q = plane->u.dot(sig.Mean());  // = 2 z / N_i
        }
        return saes.eigenvalues()[0];
    }

    // All the planes at poses xs; returns c(xs), paper eq. (1).
    double EvalPlanes(const DensePoses& xs, bool axes, Eigen::aligned_vector<PlaneAnchor>* planes) const {
        planes->resize(feats.size());
        double cost = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : cost)
        for (long a = 0; a < (long)feats.size(); a++) cost += feats[a].coeff * EvalPlane(a, xs, axes, &(*planes)[a]);
        return cost;
    }

    double OriginalCost(const DensePoses& xs) const {
        Eigen::aligned_vector<PlaneAnchor> planes;
        return EvalPlanes(xs, false, &planes);
    }

    // Freeze the planes at xs; returns c(xs) = c_M(xs | xs).
    double BuildSurrogate(const DensePoses& xs) { return EvalPlanes(xs, true, &anc_); }

    // Surrogate cost, paper eq. (8) in the point-to-plane form.
    double SurrogateCost(const DensePoses& xs) const {
        double cost = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : cost)
        for (long a = 0; a < (long)feats.size(); a++) {
            const double N = anc_[a].N;
            if (N <= 0) continue;
            const Vec3& u = anc_[a].u;
            const double q = anc_[a].q;
            PointCluster sig;
            Accumulate(feats[a], xs, &sig);
            cost += feats[a].coeff * (u.dot(sig.P * u) - 2.0 * q * u.dot(sig.v) + sig.N * q * q) / N;  // (1/N)[u'Pu - 2q u'v + Nq^2]
        }
        return cost;
    }

    // The normal equations of the surrogate and, in the same pass, its cost.
    void BuildNormalEquations(const DensePoses& xs, VecVec6* J, VecMat6* H, double* cost) const {
        NormalEq ne(Nx);
#pragma omp parallel for schedule(dynamic, 32) reduction(neq_sum : ne)
        for (long a = 0; a < (long)feats.size(); a++) {
            const PlaneFeature& f = feats[a];
            const double Ni = anc_[a].N;
            if (Ni <= 0) continue;
            const Vec3& u = anc_[a].u;
            const double q = anc_[a].q;
            const double w = f.coeff;
            const Mat3 uhat = Hat(u);
            const double k = 2.0 * w / Ni;

            PointCluster tr;
            for (const FrameCluster& o : f.clus) {
                tr.Transform(o.clu, xs[o.xid].linear(), xs[o.xid].translation());  // -> P_ij, v_ij, N_ij
                const Mat3& P = tr.P;
                const Vec3& v = tr.v;
                const double nij = tr.N;

                const double uTv = u.dot(v);
                ne.cost += w * (u.dot(P * u) - 2.0 * q * uTv + nij * q * q) / Ni;

                const Vec3 s = P * u - q * v;  // sum_k r_k x_k
                const double uTs = u.dot(s);   // sum_k r_k (u^T x_k)

                Vec6& Jj = ne.J[o.xid];
                Mat6& Hj = ne.H[o.xid];
                Jj.head<3>() += k * s.cross(u);
                Jj.tail<3>() += k * (uTv - nij * q) * u;
                Hj.block<3, 3>(0, 0) += k * (-uhat * P * uhat + 0.5 * (u * s.transpose() + s * u.transpose()) - uTs * Mat3::Identity());
                const Mat3 Hpt = -k * uhat * v * u.transpose();
                Hj.block<3, 3>(0, 3) += Hpt;
                Hj.block<3, 3>(3, 0) += Hpt.transpose();
                Hj.block<3, 3>(3, 3) += (k * nij) * u * u.transpose();
            }
        }
        *J = std::move(ne.J);
        *H = std::move(ne.H);
        *cost = ne.cost;
    }

    // Algorithm 1 of the paper: the outer loop rebuilds the surrogate, the inner loop runs LM on it with the
    // decoupled 6x6 solves of eq. (19).
    void OptimizeDense(DensePoses& xs) {
        VecVec6 J;
        VecMat6 H;
        VecVec6 dx(Nx);
        DensePoses x_new(Nx), x_prev;

        double c = OriginalCost(xs);
        const double c_init = c;
        if (opt.verbose)
            LOG(INFO) << "[ba] features " << feats.size() << ", observations " << NumObservations() << ", poses " << Nx
                      << ", initial cost " << c_init;

        for (int k = 0; k < opt.max_outer; k++) {
            c = BuildSurrogate(xs);  // majorize: the majorization is tight, so this IS c(T^k)
            const double c_k = c;
            x_prev = xs;

            double mu = 0.01, nu = 2.0;
            double cM = 0.0;
            int accepted = 0;
            bool rebuild = true;
            for (int j = 0; j < opt.max_inner; j++) {
                if (rebuild) BuildNormalEquations(xs, &J, &H, &cM);  // a rejected step reuses them

                double denom = 0.0, step_sq = 0.0;
                SolveBlocks(J, H, mu, &dx, &denom, &step_sq);
                if (std::sqrt(step_sq) < opt.eps) break;  // || dT || < eps

                for (int i = 0; i < Nx; i++) x_new[i] = Boxplus(xs[i], dx[i]);

                const double cM_new = SurrogateCost(x_new);
                const double c_new = OriginalCost(x_new);
                const double rho = (denom > 0.0) ? (cM - cM_new) / denom : -1.0;

                bool loose = false;
                if (rho > 0.0) {
                    loose = (cM - cM_new) > (c - c_new);  // the surrogate now promises more than the true cost delivers
                    xs.swap(x_new);
                    c = c_new;
                    cM = cM_new;
                    accepted++;
                    mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
                    nu = 2.0;
                    rebuild = true;
                } else {
                    mu *= nu;
                    nu *= 2.0;
                    rebuild = false;
                }
                if (loose) break;
            }

            double moved_sq = 0.0;  // || T^k [-] T^(k-1) ||
            for (int i = 0; i < Nx; i++)
                moved_sq += LogMat(Mat3(x_prev[i].linear().transpose() * xs[i].linear())).squaredNorm() +
                            (xs[i].translation() - x_prev[i].translation()).squaredNorm();

            VLOG(1) << "[ba] outer " << k << ": cost " << c_k << " -> " << c << " (accepted " << accepted << ", |dT| "
                    << std::sqrt(moved_sq) << ")";
            if (std::sqrt(moved_sq) < opt.eps) {
                if (opt.verbose) LOG(INFO) << "[ba] converged after " << k + 1 << " outer iterations";
                break;
            }
        }
        if (opt.verbose && c_init > 0) LOG(INFO) << "[ba] final cost " << c << " (" << 100.0 * c / c_init << "% of initial)";
    }

    // Central finite-difference check of J and H against the surrogate cost.  Second differences divide
    // roundoff by h^2, so the Hessian needs a much larger step than the gradient.
    double VerifyDerivativesDense(const DensePoses& xs, int n_check, double h = 1e-6, double h2 = 1e-4) {
        BuildSurrogate(xs);
        VecVec6 J;
        VecMat6 H;
        double c0 = 0;
        BuildNormalEquations(xs, &J, &H, &c0);

        const std::vector<int> cnt = PlanesPerPose();
        std::vector<int> probe;
        for (int i = 0; i < Nx && (int)probe.size() < n_check; i++)
            if (cnt[i] > 5) probe.push_back(i);

        auto CostAt = [&](int idx, const Vec6& d) {
            DensePoses x = xs;
            x[idx] = Boxplus(xs[idx], d);
            return SurrogateCost(x);
        };
        auto Unit = [](int a, double s) {
            Vec6 d = Vec6::Zero();
            d[a] = s;
            return d;
        };

        double worst = 0.0;
        for (int idx : probe) {
            Vec6 Jn;
            Mat6 Hn;
            for (int a = 0; a < 6; a++) {
                Jn[a] = (CostAt(idx, Unit(a, h)) - CostAt(idx, Unit(a, -h))) / (2 * h);
                Hn(a, a) = (CostAt(idx, Unit(a, h2)) - 2 * c0 + CostAt(idx, Unit(a, -h2))) / (h2 * h2);
            }
            for (int a = 0; a < 6; a++)
                for (int b = a + 1; b < 6; b++) {
                    const double cpp = CostAt(idx, Unit(a, h2) + Unit(b, h2));
                    const double cpm = CostAt(idx, Unit(a, h2) + Unit(b, -h2));
                    const double cmp = CostAt(idx, Unit(a, -h2) + Unit(b, h2));
                    const double cmm = CostAt(idx, Unit(a, -h2) + Unit(b, -h2));
                    Hn(a, b) = Hn(b, a) = (cpp - cpm - cmp + cmm) / (4 * h2 * h2);
                }
            const double ej = (J[idx] - Jn).cwiseAbs().maxCoeff() / std::max(1e-9, J[idx].cwiseAbs().maxCoeff());
            const double eh = (H[idx] - Hn).cwiseAbs().maxCoeff() / std::max(1e-9, H[idx].cwiseAbs().maxCoeff());
            worst = std::max(worst, std::max(ej, eh));
            LOG(INFO) << "[ba] pose " << idx << " (" << cnt[idx] << " planes): relative error J " << ej << ", H " << eh;
        }
        return worst;
    }

    void SolveBlocks(const VecVec6& J, const VecMat6& H, double mu, VecVec6* dx, double* denom, double* step_sq) const {
        double d = 0.0, s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : d, s)
        for (int j = 0; j < Nx; j++) {
            (*dx)[j].setZero();
            Mat6 A = H[j];
            if (!A.allFinite() || !J[j].allFinite()) continue;
            A.diagonal().array() += mu;
            Eigen::LDLT<Mat6> ldlt(A);
            const Vec6 step = ldlt.solve(-J[j]);
            if (ldlt.info() != Eigen::Success || !step.allFinite()) continue;
            (*dx)[j] = step;
            d += 0.5 * step.dot(mu * step - J[j]);  // Algorithm 1: rho denominator, 0.5 * dT . (mu dT - J^T)
            s += step.squaredNorm();
        }
        *denom = d;
        *step_sq = s;
    }

    Eigen::aligned_vector<PlaneAnchor> anc_;  // surrogate anchors, frozen for one outer iteration
};

}  // namespace ba
}  // namespace kuma_lio

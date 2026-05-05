#include "lidar_camera_calib.hpp"
#include <fstream>
#include <iostream>
#include "ceres/ceres.h"
#include "common.h"
// #include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>
#include <utility>
namespace po = boost::program_options;

// Data path
std::string image_file;
std::string pcd_file;
std::string output_dir;

// Calib config
bool use_rough_calib;
std::string calib_config_file;

template <typename T>
inline void ProjectPointWithDistortion(const T *_q, const T *_t, const T &x, const T &y, const T &z, T &ud, T &vd,
                                       const Eigen::Matrix3d &K, const Eigen::Vector4d &distort_param) {
    Eigen::Matrix<T, 3, 3> innerT = K.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distort_param.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(x, y, z);
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = T(1.0) + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) + distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo + distorT[2] * (r2 + yo * yo + yo * yo);
    ud = fx * xd + cx;
    vd = fy * yd + cy;
}

// pnp calib with direction vector
class VpnpCalib {
   public:
    explicit VpnpCalib(VPnPData p, Eigen::Matrix3d K, Eigen::Vector4d distort_param)
        : pd(std::move(p)), K(K), distort_param(distort_param) {}
    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const {
        T ud;
        T vd;
        ProjectPointWithDistortion(_q, _t, T(pd.lp.x()), T(pd.lp.y()), T(pd.lp.z()), ud, vd, K, distort_param);
        residuals[0] = ud - T(pd.u);
        residuals[1] = vd - T(pd.v);

        if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) {
            return true;
        } else {
            Eigen::Matrix<T, 2, 2> I = Eigen::Matrix<T, 2, 2>::Identity();
            Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
            Eigen::Matrix<T, 2, 2> V = n * n.transpose();
            V = I - V;
            Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<T, 2, 1>::Zero();
            R.coeffRef(0, 0) = residuals[0];
            R.coeffRef(1, 0) = residuals[1];
            R = V * R;
            // Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<float, 2,
            // 2>::Zero().cast<T>(); R.coeffRef(0, 0) = residuals[0];
            // R.coeffRef(1, 1) = residuals[1]; R = V * R * V.transpose();
            residuals[0] = R.coeffRef(0, 0);
            residuals[1] = R.coeffRef(1, 0);
        }
        return true;
    }
    static ceres::CostFunction *Create(VPnPData p, Eigen::Matrix3d K, Eigen::Vector4d distort_param) {
        return (new ceres::AutoDiffCostFunction<VpnpCalib, 2, 4, 3>(new VpnpCalib(std::move(p), K, distort_param)));
    }

   private:
    VPnPData pd;
    Eigen::Matrix3d K;
    Eigen::Vector4d distort_param;
};

void RoughCalib(Calibration &calibra, Eigen::Matrix3d &rot, const Eigen::Vector3d &tran, double search_resolution,
                int max_iter) {
    const int match_dis = 25;
    Eigen::Vector3d fixed_delta = Eigen::Vector3d::Zero();
    Eigen::Vector3d best_rot_vec = RotationVectorFromMatrix(rot);
    for (int n = 0; n < 2; n++) {
        for (int round = 0; round < 3; round++) {
            const Eigen::Vector3d base_rot_vec = best_rot_vec;
            double min_cost = 1000.0;
            for (int iter = 0; iter < max_iter; iter++) {
                Eigen::Vector3d delta = fixed_delta;
                delta[round] = fixed_delta[round] + pow(-1, iter) * int(iter / 2) * search_resolution;
                const Eigen::Vector3d test_rot_vec = base_rot_vec + delta;
                const Eigen::Matrix3d test_rot = RotationMatrixFromVector(test_rot_vec);
                std::vector<VPnPData> pnp_list;
                cv::Mat residual_img;
                calibra.BuildVPnp(test_rot, tran, match_dis, residual_img, calibra.rgb_egde_cloud_,
                                  calibra.lidar_processor_.plane_line_cloud_, pnp_list);
                const double cost =
                    static_cast<double>(calibra.lidar_processor_.plane_line_cloud_->size() - pnp_list.size()) /
                    static_cast<double>(calibra.lidar_processor_.plane_line_cloud_->size());
                if (cost < min_cost) {
                    std::cout << "Rough calibration min cost:" << cost << std::endl;
                    min_cost = cost;
                    best_rot_vec = test_rot_vec;
                    rot = test_rot;

                    cv::Mat residual_img;
                    calibra.BuildVPnp(rot, tran, match_dis, residual_img, calibra.rgb_egde_cloud_,
                                      calibra.lidar_processor_.plane_line_cloud_, pnp_list);
                }
            }
        }
    }
}

struct SingleTargetState {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

void WriteResultFile(const std::string &result_path, const Eigen::Matrix3d &rotation,
                     const Eigen::Vector3d &translation) {
    std::ofstream outfile(result_path);
    for (int i = 0; i < 3; i++) {
        outfile << rotation(i, 0) << "," << rotation(i, 1) << "," << rotation(i, 2) << "," << translation[i]
                << std::endl;
    }
    outfile << 0 << "," << 0 << "," << 0 << "," << 1 << std::endl;
}

void RunSingleTargetOptimization(Calibration &calibra, SingleTargetState &state, bool use_rough_calib) {
    // save init
    cv::Mat init_img = calibra.FusedProjectionImage(state.rotation, state.translation);
    cv::imwrite(output_dir + "/init.png", init_img);

    // rough calib
    if (use_rough_calib) {
        RoughCalib(calibra, state.rotation, state.translation, DEG2RAD(0.1), 50);
    }
    cv::Mat test_img = calibra.FusedProjectionImage(state.rotation, state.translation);
    cv::imwrite(output_dir + "/after_rough.png", test_img);

    // optimization
    int iter = 0;
    std::vector<VPnPData> vpnp_list;
    for (int dis_threshold = 30; dis_threshold > 10; dis_threshold -= 1) {
        for (int cnt = 0; cnt < 2; cnt++) {
            // build constrain
            cv::Mat residual_img;
            calibra.BuildVPnp(state.rotation, state.translation, dis_threshold, residual_img, calibra.rgb_egde_cloud_,
                              calibra.lidar_processor_.plane_line_cloud_, vpnp_list);

            cv::imwrite(output_dir + "/residual_" + std::to_string(iter) + ".png", residual_img);
            // to double vector
            Eigen::Quaterniond q(state.rotation);
            double ext[7];
            ext[0] = q.x();
            ext[1] = q.y();
            ext[2] = q.z();
            ext[3] = q.w();
            ext[4] = state.translation[0];
            ext[5] = state.translation[1];
            ext[6] = state.translation[2];
            Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
            Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

            // problem
            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem problem;
            problem.AddParameterBlock(ext, 4, q_parameterization);
            problem.AddParameterBlock(ext + 4, 3);

            // add constraint
            for (const auto &val : vpnp_list) {
                auto cam = std::dynamic_pointer_cast<faster_lio::PinholeRadialCamera>(calibra.camera_);
                Eigen::Vector4d distort_param;
                std::vector<double> param = cam->get_params();
                distort_param << param[4], param[5], param[7], param[8];
                Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
                K(0,0) = cam->fx_;
                K(1,1) = cam->fy_;
                K(0,2) = cam->cx_;
                K(1,2) = cam->cy_;
                ceres::CostFunction *cost_function = VpnpCalib::Create(val, K, distort_param);
                problem.AddResidualBlock(cost_function, nullptr, ext, ext + 4);
            }

            // solve
            ceres::Solver::Options options;
            options.preconditioner_type = ceres::JACOBI;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.minimizer_progress_to_stdout = false;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;

            // to state
            m_q.normalize();
            state.rotation = m_q.toRotationMatrix();
            state.translation = m_t;

            // print and save result
            printf("iter: %d, distance thr: %d, constraint count: %zu\n", iter, dis_threshold, vpnp_list.size());
            cv::Mat projection_img = calibra.FusedProjectionImage(state.rotation, state.translation);
            cv::imwrite(output_dir + "/opt_" + std::to_string(iter) + ".png", projection_img);
            iter++;
        }  // for iter
    }      // for dist thr

}

int main(int argc, char **argv) {
    po::options_description desc("Allowed options");
    // clang-format off
  desc.add_options()
    ("help,h", "Print help message")
    ("image_file", po::value<std::string>(&image_file)->required(), "Input file")
    ("pcd_file", po::value<std::string>(&pcd_file)->required(), "Input file")
    ("output_dir", po::value<std::string>(&output_dir)->required(), "Input file")
    ("calib_config_file", po::value<std::string>(&calib_config_file)->required(), "Input file")
    ("use_rough_calib", po::bool_switch(&use_rough_calib)->default_value(false), "Enable rough calibration stage")
  ;
    // clang-format on
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << "Argument error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    Calibration calibra(image_file, pcd_file, calib_config_file);
    SingleTargetState state;
    state.rotation = calibra.init_rotation_matrix_;
    state.translation = calibra.init_translation_vector_;

    std::cout << "Finish prepare!" << std::endl;
    std::cout << "Initial rotation matrix:" << std::endl << state.rotation << std::endl;
    std::cout << "Initial translation:" << state.translation.transpose() << std::endl;

    RunSingleTargetOptimization(calibra, state, use_rough_calib);
    WriteResultFile(output_dir + "/result.txt", state.rotation, state.translation);

    cv::Mat opt_img = calibra.FusedProjectionImage(state.rotation, state.translation);
    cv::imwrite(output_dir + "/result.png", opt_img);
    return 0;
}

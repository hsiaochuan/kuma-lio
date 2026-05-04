// points_jet.cpp
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <pcl/io/pcd_io.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/jet_smooth_point_set.h>
#include <pcl_conversions/pcl_conversions.h>

namespace po = boost::program_options;

// CGAL kernel
using Kernel = CGAL::Simple_cartesian<double>;
using Point = Kernel::Point_3;

int main(int argc, char** argv) {
    std::string input_fname;
    std::string output_fname;

    int jet_nb = 16;

    bool enable_sor = true;
    int sor_mean_k = 16;
    double sor_stddev = 1.0;
    po::options_description desc("Jet smoothing options");
    desc.add_options()
    // clang-format off
    ("help,h", "Show help message")
    ("input,i", po::value<std::string>(&input_fname)->required(),"Input PCD file")
    ("output,o", po::value<std::string>(&output_fname)->required(), "Output PCD file")

    ("k,k", po::value<int>(&jet_nb)->default_value(16), "Jet neighborhood size")

    ("sor", po::bool_switch(&enable_sor)->default_value(true), "Enable Statistical Outlier Removal")
    ("sor-mean-k", po::value<int>(&sor_mean_k)->default_value(16), "SOR mean K")
    ("sor-stddev", po::value<double>(&sor_stddev)->default_value(1.0), "SOR stddev multiplier");
    // clang-format on
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
        }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n\n" << desc << std::endl;
    }

    // 1) Load input PCD (XYZI as common lidar format)
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile(input_fname, *cloud_in) != 0) {
        std::cerr << "[ERROR] Failed to load input PCD: " << input_fname << std::endl;
        return 1;
    }
    if (cloud_in->empty()) {
        std::cerr << "[ERROR] Input cloud is empty." << std::endl;
        return 1;
    }
    std::cout << "[INFO] Loaded " << cloud_in->size() << " points from " << input_fname << std::endl;

    if (enable_sor) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
        sor.setInputCloud(cloud_in);
        sor.setMeanK(sor_mean_k);
        sor.setStddevMulThresh(sor_stddev);
        sor.filter(*cloud_filtered);
        if (cloud_filtered->empty()) {
            std::cerr << "[ERROR] SOR filtered cloud is empty." << std::endl;
            return 1;
        }
        cloud_in = cloud_filtered;
        std::cout << "[INFO] SOR filtered points: " << cloud_in->size() << std::endl;
    }

    // 2) Convert to CGAL point set
    std::vector<Point> points;
    points.reserve(cloud_in->size());
    for (const auto& p : cloud_in->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        points.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
    }

    if (points.size() < jet_nb) {
        std::cerr << "[ERROR] Valid point count (" << points.size() << ") is smaller than --k (" << jet_nb << ")."
                  << std::endl;
        return 1;
    }

    // 3) Jet smoothing (can iterate multiple times)
    for (unsigned int it = 0; it < 1; ++it) {
        CGAL::jet_smooth_point_set<CGAL::Sequential_tag>(points, jet_nb);
    }

    // 4) Convert back to PCL and save
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>);
    cloud_out->reserve(points.size());

    for (const auto& p : points) {
        pcl::PointXYZI q;
        q.x = static_cast<float>(p.x());
        q.y = static_cast<float>(p.y());
        q.z = static_cast<float>(p.z());
        q.intensity = 0.0f;  // intensity lost unless you use point+property map version
        cloud_out->push_back(q);
    }

    cloud_out->width = static_cast<std::uint32_t>(cloud_out->size());
    cloud_out->height = 1;
    cloud_out->is_dense = false;

    if (pcl::io::savePCDFileBinary(output_fname, *cloud_out) != 0) {
        std::cerr << "[ERROR] Failed to save output PCD: " << output_fname << std::endl;
        return 1;
    }

    std::cout << "[INFO] Smoothed cloud saved to " << output_fname << " (" << cloud_out->size() << " points)."
              << std::endl;

    return 0;
}
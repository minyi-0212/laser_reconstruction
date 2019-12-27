#pragma once
#include "coor_system.h"
#include <opencv2/core.hpp>
#include <vector>

// from origin pixel coordinate to undistort pixel coordinate
void my_undistort_points(const std::vector<cv::Point2d>& p_in, std::vector<cv::Point2d>& p_out,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

void my_print(std::vector<cv::Point2d>& t1);
void export_pointcloud_ply(const char *filename,
	const std::vector<cv::Point3f>& pos, const std::vector<cv::Point3f>& p_normal, const std::vector<cv::Point3f>& p_color);
void rename_file(const char path[], const char prifix[]);
void gaussian(const int dim, const int xigma, const std::vector<float>& value, std::vector<float>& result);
void image_rotate(const cv::Mat& src, cv::Mat& dst, float angle, cv::Mat& rot_mat);

void output_coor_system(const std::string& filename, const std::vector<coor_system>& coordinate);
void output_laser_plane(const std::string& filename, const std::vector<double>& laser_plane_in_camera);
void input_coor_system(const std::string& filename, std::vector<coor_system>& coordinate);
void input_laser_plane(const std::string& filename, std::vector<double>& laser_plane_in_camera);

void fitPlane_least_square(std::vector<cv::Point3d>& points, std::vector<double>& plane);
void fitPlane_svd(std::vector<cv::Point3d>& points, std::vector<double>& plane);

void gaussian_with_mask(const int dim, const int xigma, const cv::Mat& mask, cv::Mat& image);

void find_range(const cv::Mat& image, std::vector<float>& range, float percent = 0.9, int max_value = 150);
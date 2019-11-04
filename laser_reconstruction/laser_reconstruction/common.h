#pragma once
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
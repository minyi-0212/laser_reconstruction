#pragma once
#include "coor_system.h"
#include <opencv2/core.hpp>

void reconstruct_test(const char* filepath, const cv::Mat& camera_matrix, const cv::Mat& RT, const float rotate_angle);
void reconstruct_test2(const char* filepath, const cv::Mat& camera_matrix, const cv::Mat& RT,
	const std::vector<double>& laser_plane_in_camera, std::vector<coor_system>& coordinate);
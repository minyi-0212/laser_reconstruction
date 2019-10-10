#pragma once
#include <opencv2/core.hpp>

class coor_system
{
	cv::Mat cameraMatrix, RT, cam_pos;
	cv::Vec3d rvec, tvec;
	double k, b;
public:

	coor_system();
	coor_system(const cv::Mat& cameraM);
	~coor_system();
	void init(const cv::Mat& cameraM);
	void set_rt(const cv::Vec3d& r, const cv::Vec3d& t);
	void get_rt(cv::Vec3d& r, cv::Vec3d& t);
	void world_to_pixel(const std::vector<cv::Point3f>& p_world, std::vector<cv::Point2f>& p_pixel, 
		const cv::Mat& cm, const cv::Mat& distCoeffs);
	void pixel_to_world(const std::vector<cv::Point2f>& p_pixel, std::vector<cv::Point3f>& p_world, int z = 0);
	//void pixel_to_camera(const std::vector<cv::Point2f>& p_pixel, std::vector<cv::Point3f>& p_camera);
	void camera_to_pixel(const std::vector<cv::Point3f>& p_camera, std::vector<cv::Point2f>& p_pixel);
	void world_to_camera(const std::vector<cv::Point3f>& p_world, std::vector<cv::Point3f>& p_camera);
	void set_laser_line(double kk, double bb);
	cv::Point2f get_laser_line_point(double y);

	// 
	void set_RT_matrix(const cv::Mat& RT_Mat);
	void world_to_pixel(const std::vector<cv::Point3f>& p_world, std::vector<cv::Point2f>& p_pixel);
	void pixel_to_camera(const std::vector<cv::Point2f>& p_pixel, std::vector<cv::Point3f>& vec_camera_dir);
	void camera_to_world(const std::vector<cv::Point3f>& p_camera, std::vector<cv::Point3f>& p_world);
	// the origin point is campos, in camera system is (0,0,0)
};
#include "coor_system.h"
#include <iostream>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

coor_system::coor_system()
{
}

coor_system::coor_system(const Mat& cameraM)
{
	init(cameraM);
}

coor_system::~coor_system()
{
}

void coor_system::init(const Mat& cameraM)
{
	/*cameraMatrix << cameraM.at<double>(0, 0), cameraM.at<double>(0, 1), cameraM.at<double>(0, 2), 0,
		cameraM.at<double>(1, 0), cameraM.at<double>(1, 1), cameraM.at<double>(1, 2), 0,
		cameraM.at<double>(2, 0), cameraM.at<double>(2, 1), cameraM.at<double>(2, 2), 0,
		0, 0, 0, 1;*/
		//cout << cameraMatrix << endl;
	cameraMatrix = Mat::zeros(4, 4, CV_64FC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			cameraMatrix.at<double>(j, i) = cameraM.at<double>(j, i);
	cameraMatrix.at<double>(3, 3) = 1;
}

void coor_system::set_rt(const Vec3d& r, const Vec3d& t)
{
	rvec = r;
	tvec = t;

	// RT
	RT = Mat::zeros(4, 4, CV_64FC1);
	Mat R, rvecs = Mat(rvec);
	Rodrigues(rvecs, R);// R 3x3
	/*RT << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), tvec[0],
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), tvec[1],
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), tvec[2],
		0, 0, 0, 1;*/
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			RT.at<double>(j, i) = R.at<double>(j, i);
	RT.at<double>(0, 3) = tvec[0];
	RT.at<double>(1, 3) = tvec[1];
	RT.at<double>(2, 3) = tvec[2];
	RT.at<double>(3, 3) = 1;
	//cout << "the rt matrix: " << RT << endl;

	// cam_pos
	cam_pos = Mat::zeros(4, 1, CV_64FC1);
	cam_pos.at<double>(0) = 0;
	cam_pos.at<double>(1) = 0;
	cam_pos.at<double>(2) = 0;
	cam_pos.at<double>(3) = 1;
	cam_pos = RT.inv() * cam_pos;
	//cout << "cam_pos: " << cam_pos.t() << endl;
}

void coor_system::get_rt(Vec3d& r, Vec3d& t)
{
	r = rvec;
	t = tvec;
}

void coor_system::world_to_pixel(const vector<Point3f>& p_world, vector<Point2f>& p_pixel, 
	const Mat& cm, const Mat& distCoeffs)
{
	projectPoints(p_world, rvec, tvec, cm, distCoeffs, p_pixel);
}

void coor_system::pixel_to_world(const vector<Point2f>& p_pixel, vector<Point3f>& p_world, int z)
{
	p_world.resize(p_pixel.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1), direction(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_pixel.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_pixel[i].x;
		in_homogeneous.at<double>(1, 0) = p_pixel[i].y;
		in_homogeneous.at<double>(2, 0) = 1;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = RT.inv() * cameraMatrix.inv() * in_homogeneous;
		direction = out_homogeneous - cam_pos;
		//t = (z-cam_pos.at<double>(2, 0)) / direction.at<double>(2, 0);			//!!!!!!!!
		t = (z - cam_pos.at<double>(1, 0)) / direction.at<double>(1, 0);
		out_homogeneous = cam_pos + t * direction;
		p_world[i].x = out_homogeneous.at<double>(0, 0);
		p_world[i].y = out_homogeneous.at<double>(1, 0);
		p_world[i].z = out_homogeneous.at<double>(2, 0);

		//cout << p_world[0] << "	<-	" << p_pixel[0] << endl;
	}
}

/*void coor_system::pixel_to_camera(const std::vector<cv::Point2f>& p_pixel, std::vector<cv::Point3f>& p_camera)
{
	p_camera.resize(p_pixel.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_pixel.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_pixel[i].x;
		in_homogeneous.at<double>(1, 0) = p_pixel[i].y;
		in_homogeneous.at<double>(2, 0) = 1;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = cameraMatrix.inv() * in_homogeneous;
		p_camera[i].x = out_homogeneous.at<double>(0, 0);
		p_camera[i].y = out_homogeneous.at<double>(1, 0);
		p_camera[i].z = out_homogeneous.at<double>(2, 0);
		//cout << out_homogeneous.t() << endl;
		//cout << p_camera[0] << "	<-	" << p_pixel[0] << endl;
	}
}*/

void coor_system::camera_to_pixel(const std::vector<cv::Point3f>& p_camera, std::vector<cv::Point2f>& p_pixel)
{
	p_pixel.resize(p_camera.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_pixel.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_camera[i].x;
		in_homogeneous.at<double>(1, 0) = p_camera[i].y;
		in_homogeneous.at<double>(2, 0) = p_camera[i].z;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = cameraMatrix * in_homogeneous;
		p_pixel[i].x = out_homogeneous.at<double>(0, 0) / out_homogeneous.at<double>(2, 0);
		p_pixel[i].y = out_homogeneous.at<double>(1, 0) / out_homogeneous.at<double>(2, 0);
		//cout << out_homogeneous.t() << endl;
		//cout << p_camera[0] << "	->	" << p_pixel[0] << endl;
	}
}

void coor_system::world_to_camera(const std::vector<cv::Point3f>& p_world, std::vector<cv::Point3f>& p_camera)
{
	p_camera.resize(p_world.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_world.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_world[i].x;
		in_homogeneous.at<double>(1, 0) = p_world[i].y;
		in_homogeneous.at<double>(2, 0) = p_world[i].z;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = RT * in_homogeneous;
		p_camera[i].x = out_homogeneous.at<double>(0, 0);
		p_camera[i].y = out_homogeneous.at<double>(1, 0);
		p_camera[i].z = out_homogeneous.at<double>(2, 0);
		//cout << out_homogeneous.t() << endl << endl;
		//cout << p_camera[0] << "	->	" << p_pixel[0] << endl;
	}
}

void coor_system::set_laser_line(double kk, double bb)
{
	k = kk;
	b = bb;
}

Point2f coor_system::get_laser_line_point(double y)
{
	return Point2f((y - b) / k, y);
}

void coor_system::set_RT_matrix(const cv::Mat& RT_Mat)
{
	RT = RT_Mat;

	cam_pos = Mat::zeros(4, 1, CV_64FC1);
	cam_pos.at<double>(0) = 0;
	cam_pos.at<double>(1) = 0;
	cam_pos.at<double>(2) = 0;
	cam_pos.at<double>(3) = 1;
	cam_pos = RT.inv() * cam_pos;
	cout << "cam_pos: " << cam_pos.t() << endl;
}

void coor_system::world_to_pixel(const std::vector<cv::Point3f>& p_world, std::vector<cv::Point2f>& p_pixel)
{
	p_pixel.resize(p_world.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_pixel.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_world[i].x;
		in_homogeneous.at<double>(1, 0) = p_world[i].y;
		in_homogeneous.at<double>(2, 0) = p_world[i].z;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = cameraMatrix * RT * in_homogeneous;
		p_pixel[i].x = out_homogeneous.at<double>(0, 0) / out_homogeneous.at<double>(2, 0);
		p_pixel[i].y = out_homogeneous.at<double>(1, 0) / out_homogeneous.at<double>(2, 0);
		//cout << out_homogeneous.t() << endl;
		//cout << p_camera[0] << "	->	" << p_pixel[0] << endl;
	}
}

void coor_system::pixel_to_camera(const std::vector<cv::Point2f>& p_pixel, std::vector<cv::Point3f>& vec_camera_dir)
{
	vec_camera_dir.resize(p_pixel.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_pixel.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_pixel[i].x;
		in_homogeneous.at<double>(1, 0) = p_pixel[i].y;
		in_homogeneous.at<double>(2, 0) = 1;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = cameraMatrix.inv() * in_homogeneous;
		vec_camera_dir[i].x = out_homogeneous.at<double>(0, 0);
		vec_camera_dir[i].y = out_homogeneous.at<double>(1, 0);
		vec_camera_dir[i].z = out_homogeneous.at<double>(2, 0);
		//cout << out_homogeneous.t() << endl;
		//cout << p_camera[0] << "	<-	" << p_pixel[0] << endl;
	}
}

void coor_system::camera_to_world(const std::vector<cv::Point3f>& p_camera, std::vector<cv::Point3f>& p_world)
{
	p_world.resize(p_camera.size());
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1);
	double t;
	for (int i = 0; i < p_camera.size(); i++)
	{
		in_homogeneous.at<double>(0, 0) = p_camera[i].x;
		in_homogeneous.at<double>(1, 0) = p_camera[i].y;
		in_homogeneous.at<double>(2, 0) = p_camera[i].z;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = RT.inv() * in_homogeneous;
		p_world[i].x = out_homogeneous.at<double>(0, 0);
		p_world[i].y = out_homogeneous.at<double>(1, 0);
		p_world[i].z = out_homogeneous.at<double>(2, 0);
		//cout << out_homogeneous.t() << endl << endl;
		//cout << p_camera[0] << "	->	" << p_pixel[0] << endl;
	}
}
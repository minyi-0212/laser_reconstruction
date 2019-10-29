#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "3d_reconstruction.h"
#include "common.h"

using cv::Mat;
using std::cout;
using std::endl;

namespace {
	const char* about =
		"Calibration using a ChArUco board\n"
		"  To capture a frame for calibration, press 'c',\n"
		"  If input comes from video, press any key for next frame\n"
		"  To finish capturing, press 'ESC' key and calibration starts.\n";
	const char* keys =
		"{w        |       | Number of squares in X direction }"
		"{h        |       | Number of squares in Y direction }"
		"{sl       |       | Square side length (in meters) }"
		"{ml       |       | Marker side length (in meters) }"
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
		"{@infile  |<none> | input file with calibrated camera parameters }"
		"{v        |       | Input from video file, if ommited, input comes from camera }"
		"{folder   |       | Input from image file folder }"
		"{read     |       | Read the laser point file}"
		"{dp       |       | File of marker detector parameters }"
		"{rs       | false | Apply refind strategy }"
		"{zt       | false | Assume zero tangential distortion }"
		"{a        |       | Fix aspect ratio (fx/fy) to this value }"
		"{pc       | false | Fix the principal point at the center }"
		"{sc       | false | Show detected chessboard corners after calibration }";
}

extern void undistort_images(const std::string& pattern_jpg, const Mat& cameraMatrix, const Mat& distCoeffs);
extern void compute_laser_plane_test(const cv::CommandLineParser& parser, const char filepath[],
	const Mat& cameraMatrix, const Mat& distCoeffs,
	std::vector<double>& laser_plane_in_camera, std::vector<coor_system>& coordinate);
void check(const Mat& cameraMatrix, const Mat& RT);
extern int test_aruco(const cv::Mat& cameraMatrix);


//#define COMPUTE_LASER_PLANE
int main(int argc, char *argv[]) {
	/*rename_file("../virtual_checkboard2", "test_");
	system("pause");
	return 0;*/

	// parser the params of the exe
	cv::CommandLineParser parser(argc, argv, keys);
	{
		parser.about(about);

		if (argc < 10) {
			parser.printMessage();
			system("pause");
			return 0;
		}
		/*cout << "intrinsic file :	" << parser.get<string>(0) << endl
			<< "parameters :		" << parser.get<string>("dp") << endl
			<< "image folder :		" << parser.get<string>("folder") << endl
			<< "read :			" << parser.get<bool>("read") << endl << endl;*/
	}

#ifdef COMPUTE_LASER_PLANE
	//read camera intrinsic
	cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
	int width, height;
	{
		cv::FileStorage fs(parser.get<std::string>(0), cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cout << parser.get<std::string>(0) << "not exists." << std::endl;
			system("pause");
			return 0;
		}
		
		fs["camera_matrix"] >> intrinsic_matrix_loaded;
		fs["distortion_coefficients"] >> distortion_coeffs_loaded;
		fs["image_width"] >> width;
		fs["image_height"] >> height;
		fs.release();

		cout << "image width: " << width << endl
			<< "image height: " << height << endl
			<< "intrinsic matrix:" << endl << intrinsic_matrix_loaded << endl
			<< "distortion coefficients: " << endl << distortion_coeffs_loaded << endl << endl;
	}
	//test_aruco(intrinsic_matrix_loaded);

	// compute the laser plane
	/*std::string image_path = "../real/images/*.png";
	undistort_images(image_path, intrinsic_matrix_loaded, distortion_coeffs_loaded);*/
	std::vector<double> laser_plane_in_camera;
	std::vector<coor_system> coordinate;
	compute_laser_plane_test(parser, "./real/cube_checkboard/dist_pose_*.png", 
		intrinsic_matrix_loaded, distortion_coeffs_loaded,
		laser_plane_in_camera, coordinate);
#endif

#ifndef COMPUTE_LASER_PLANE
	cv::Mat camera_matrix, RT;
	{
		cv::FileStorage fs("../input/intrinsic.yml", cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cout << "../input/intrinsic.yml not exists." << std::endl;
			system("pause");
			return 0;
		}
		fs["camera_matrix"] >> camera_matrix;
		//fs["RT_matrix"] >> RT;
		fs.release();

		cout << "intrinsic matrix:" << endl << camera_matrix << endl;

		/*camera_matrix = (cv::Mat_<double>(3, 3) <<
			4.7044376140907189e3 / 2, 0, 1024 / 2,
			0, 4.7691463862436294e3 / 2, 768 / 2,
			0, 0, 1);
		cout << "intrinsic matrix:" << endl << camera_matrix << endl << endl;*/

		double unit=0.0206;
		cv::Vec3d cam_pos(0, 0.4 / unit, 0.5 / unit), lookat(0, 0, 0);
		cv::Vec3d z(lookat - cam_pos), y(0, -1, 0), x(y.cross(z));
		y = z.cross(x);
		cv::normalize(x, x);
		cv::normalize(y, y);
		cv::normalize(z, z);
		RT = (cv::Mat_<double>(4, 4) <<
				x[0], x[1], x[2], 0,
				y[0], y[1], y[2], 0,
				z[0], z[1], z[2], 0,
				0, 0, 0, 1) *
			(cv::Mat_<double>(4, 4) <<
				1, 0, 0, -cam_pos[0],
				0, 1, 0, -cam_pos[1],
				0, 0, 1, -cam_pos[2],
				0, 0, 0, 1);

		/*RT = (cv::Mat_<double>(4, 4) <<
			-4.37114e-08, 1.14662e-15, -1, 4.37114e-08,
			0.624695, -0.780869, -2.73063e-08, -4.76837e-08,
			-0.780869, -0.624695, 3.41329e-08, 1.28062,
			0, 0, 0, 1);*/
		cout << "RT matrix:" << endl << RT << endl << endl;
		/*Mat tmp = (cv::Mat_<double>(4, 1) << 1, 0.8, 0, 1);
		cout << tmp.t() << endl << endl;
		cout << (RT*tmp).t() << endl;*/

		/*cv::FileStorage fs("../input/intrinsic_virtual.yml", cv::FileStorage::WRITE);
		fs << "image_width" << 1024 << "image_height" << 768 << "camera_matrix" << camera_matrix << "RT_matrix" << RT;
		fs.release();*/
	}
	
	//check(camera_matrix, RT);
	////reconstruct_test("../virtual_ball", camera_matrix, RT, -0.2);
	//system("pause");
	//return 0;

	{
		std::vector<double> laser_plane_in_camera;
		std::vector<coor_system> coordinate;
		Mat distortion_coeffs = cv::Mat::zeros(cv::Size(1, 14), CV_64FC1);
		
		compute_laser_plane_test(parser, 
			//"../virtual_checkboard2/test_*.png",
			"./real/cube_checkboard/dist_pose_*.png",
			camera_matrix, distortion_coeffs,
			laser_plane_in_camera, coordinate);
		cout << "the plane: " << endl;
		for (auto a : laser_plane_in_camera)
		{
			cout << a << " ";
		}
		cout << endl << "-------------------------------------------" << endl;

		reconstruct_test2("../virtual_cube", camera_matrix, RT, laser_plane_in_camera, coordinate);
	}

#endif

	system("pause");
	return 0;
}
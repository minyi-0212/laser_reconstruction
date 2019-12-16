#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "3d_reconstruction.h"
#include "common.h"

using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
using std::string;

namespace {
	const char* about =
		"Calibration using a ChArUco board\n"
		"  To capture a frame for calibration, press 'c',\n"
		"  If input comes from video, press any key for next frame\n"
		"  To finish capturing, press 'ESC' key and calibration starts.\n";
	const char* keys =
		"{w        |		| Number of squares in X direction }"
		"{h        |		| Number of squares in Y direction }"
		"{sl       |		| Square side length (in meters) }"
		"{ml       |		| Marker side length (in meters) }"
		"{d        |		| dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
		"{input_file|		| eg: F:/800_2/checkboard2X800_dist/dist_pose_*.png}"
		"{output_file|		| eg: F:/800_2/checkboard2X800_dist}"
		"{laser_file|		| eg: F:/800_2/checkboard2X800_laser_green_1212/coor_laser.txt}"
		"{output_prefix|	| eg: dist_pose_}"
		"{@infile  |<none>  | input file with calibrated camera parameters }"
		"{v        |        | Input from video file, if ommited, input comes from camera }"
		"{folder   |	    | Input from image file folder }"
		"{read     |	    | Read the laser point file}"
		"{dp       |	    | File of marker detector parameters }"
		"{rs       | false  | Apply refind strategy }"
		"{zt       | false  | Assume zero tangential distortion }"
		"{a        |        | Fix aspect ratio (fx/fy) to this value }"
		"{pc       | false  | Fix the principal point at the center }"
		"{sc       | false  | Show detected chessboard corners after calibration }";
}

extern void undistort_images(const std::string& input_file, const std::string& output_path, const std::string& output_prefix,
	const Mat& cameraMatrix, const Mat& distCoeffs);
extern void compute_laser_plane_test(const cv::CommandLineParser& parser, const char filepath[], const std::string& output_path,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	std::vector<double>& laser_plane_in_camera, std::vector<coor_system>& coordinate);
extern void draw_laser_in_images(const char filepath[], const std::string& output_path,
	std::vector<double>& laser_plane_in_camera, std::vector<coor_system>& coordinate);
extern void check(const Mat& cameraMatrix, const Mat& RT);
extern int test_aruco(const cv::Mat& cameraMatrix);
extern void laser_points_find_analysis();


#define COMPUTE_LASER_PLANE
#define LASER_RED
int main(int argc, char *argv[]) {
	/*//rename_file("../images", "test_");
	//laser_points_find_analysis();
	Mat src, dst, rotate_mat;
	char file[_MAX_PATH];
	for (int i = 0; i < 40; i++)
	{
		sprintf_s(file, "./squirrel/dist_pose_%03d.png", i);
		//sprintf_s(file, "./cube_checkboard/coord_checkboard_%03d.png", i);
		src = cv::imread(file);
		image_rotate(src, dst, -59.45, rotate_mat);
	}
	system("pause");
	return 0;*/

	// parser the params of the exe
	cv::CommandLineParser parser(argc, argv, keys);
	cv::Mat camera_matrix, distortion_coeffs_loaded;
	int width, height;
	{
		parser.about(about);

		/*if (argc < 6) {
			parser.printMessage();
			system("pause");
			return 0;
		}*/
		cout << "intrinsic file :	" << parser.get<string>(0) << endl;
		/*cout << "intrinsic file :	" << parser.get<string>(0) << endl
			<< "parameters :		" << parser.get<string>("dp") << endl
			<< "image folder :		" << parser.get<string>("folder") << endl
			<< "read :			" << parser.get<bool>("read") << endl << endl;*/

		//read camera intrinsic
		{
			cv::FileStorage fs(parser.get<std::string>(0), cv::FileStorage::READ);
			if (!fs.isOpened())
			{
				std::cout << parser.get<std::string>(0) << "not exists." << std::endl;
				system("pause");
				return 0;
			}

			fs["camera_matrix"] >> camera_matrix;
			fs["distortion_coefficients"] >> distortion_coeffs_loaded;
			fs["image_width"] >> width;
			fs["image_height"] >> height;
			fs.release();

			cout << "image width: " << width << endl
				<< "image height: " << height << endl
				<< "intrinsic matrix:" << endl << camera_matrix << endl
				<< "distortion coefficients: " << endl << distortion_coeffs_loaded << endl << endl;
		}
	}
	std::string input_file = "F:/800_2/dist/laser_1214_select2/dist_pose_*.png",
#ifndef LASER_RED
		output_file_path = "F:/800_2/coordinate/laser_1214",
		output_coor_laser_path = "F:/800_2/checkboard_1212/green_draw",
		input_coor_laser_file = "F:/800_2/coordinate/checkboardX800_1214",	//"F:/800_2/coordinate/laser_1214",	
#else
		output_file_path = "F:/800_2/coordinate/laser_1214",
		output_coor_laser_path = "F:/800_2/checkboard_1212/red_draw",
		input_coor_laser_file  = "F:/800_2/coordinate/checkboardX800_1214",
#endif
		reconstruction_file_path = "F:/800_2/dist/cupX800_1214";			//"F:/800_2/dist/cupX40_1214";
#ifdef COMPUTE_LASER_PLANE
	//test_aruco(intrinsic_matrix_loaded);

	// compute the laser plane
	/*std::string image_path = parser.get<string>("input_file"),		// "F:/800_2/checkboard2X800/*.png",
		output_file = parser.get<string>("output_file"),			//"F:/800_2/checkboard2X800_dist",
		output_prefix = parser.get<string>("output_prefix");		//"dist_pose_";
	cout << "output_file_path: " << output_file << endl;
	undistort_images(image_path, output_file, output_prefix, camera_matrix, distortion_coeffs_loaded);*/

	//cout <<"output_file_path: "<< output_file_path << endl;
	std::vector<double> laser_plane_in_camera;
	std::vector<coor_system> coordinate;

#if 1
	/*input_file = parser.get<string>("input_file");
	output_file_path = parser.get<string>("output_file");*/
	cout << "input file : " << input_file << endl
		<< "output path : " << output_file_path << endl << endl;
	compute_laser_plane_test(parser, input_file.c_str(), output_file_path,
		camera_matrix, distortion_coeffs_loaded,
		laser_plane_in_camera, coordinate);
	//output_coor_system(output_file_path + "/coordinate.txt", coordinate);
	output_laser_plane(output_file_path + "/red_laser_plane2.txt", laser_plane_in_camera);
#else
	input_file = parser.get<string>("input_file");
	//input_coor_laser_file = parser.get<string>("laser_file");
	output_file_path = parser.get<string>("output_file");
	//input_file = "F:/800_2/checkboard2X800_laser_green_12102/result_*.png";
	cout << "input file : " << input_file << endl
		//<< "input_coor_laser_file : " << input_coor_laser_file << endl
		<< "output path : " << output_file_path << endl<< endl;

	input_coor_system(output_file_path + "/coordinate.txt", coordinate);
	input_laser_plane(output_file_path + "/laser_plane.txt", laser_plane_in_camera);
	cout << "laser: ";
	for(auto l: laser_plane_in_camera)
		cout << l << ", ";
	cout << endl;
	
	//for (int i = 0; i < 7; i++)
	//	cout << coordinate[i+1].RT.inv() * coordinate[i].RT - coordinate[i+2].RT.inv() * coordinate[i+1].RT << endl;
	//	//cout << coordinate[i].RT.inv() * coordinate[i + 1].RT - coordinate[i + 1].RT.inv()*coordinate[i + 2].RT << endl;
	//cout << coordinate[799].RT.inv() * coordinate[0].RT - coordinate[0].RT.inv()*coordinate[1].RT << endl;
	
	draw_laser_in_images(input_file.c_str(), output_file_path, laser_plane_in_camera, coordinate);
#endif
#endif
	
#ifndef COMPUTE_LASER_PLANE
	cv::Mat RT;
	{
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

		/*camera_matrix = (cv::Mat_<double>(3, 3) <<
			4.7044376140907189e3 / 2, 0, 1024 / 2,
			0, 4.7691463862436294e3 / 2, 768 / 2,
			0, 0, 1);
		cout << "intrinsic matrix:" << endl << camera_matrix << endl << endl;

		cv::FileStorage fs("../input/intrinsic_virtual.yml", cv::FileStorage::WRITE);
		fs << "image_width" << 1024 << "image_height" << 768 << "camera_matrix" << camera_matrix << "RT_matrix" << RT;
		fs.release();*/
	}
	
	//check(camera_matrix, RT);
	////reconstruct_test("../virtual_ball", camera_matrix, RT, -0.2);
	//system("pause");
	//return 0;

	{
		/*std::vector<double> laser_plane_in_camera;
		std::vector<coor_system> coordinate;
		Mat distortion_coeffs = cv::Mat::zeros(cv::Size(1, 14), CV_64FC1);
		
		std::string output_file = "./checkboard_800_draw";
		compute_laser_plane_test(parser,
			//"./real/cube_checkboard/dist_pose_*.png",
			"checkboard_800/dist_pose_*.png",
			output_file, camera_matrix, distortion_coeffs,
			laser_plane_in_camera, coordinate);
		cout << "the plane: " << endl;
		for (auto a : laser_plane_in_camera)
		{
			cout << a << " ";
		}
		cout << endl << "-------------------------------------------" << endl;
		// -184.04 311.931 108.288 - 5375.86
		// laser_plane_in_camera = std::vector<double>({ -184.04, 311.931, 108.288, -5375.86 });
		reconstruct_test2("./squirrel", camera_matrix, RT, laser_plane_in_camera, coordinate);*/

		/*input_coor_laser_file = parser.get<string>("laser_file");
		reconstruction_file_path = parser.get<string>("input_file");*/
		std::vector<double> laser_plane_in_camera;
		std::vector<coor_system> coordinate;
		cout << "input path : " << reconstruction_file_path << endl
			<< "input_coor_laser_file : " << input_coor_laser_file << endl << endl;
		input_coor_system(input_coor_laser_file + "/coordinate.txt", coordinate);
		input_laser_plane(input_coor_laser_file + "/red_laser_plane2.txt", laser_plane_in_camera);
		cout << "laser: ";
		for (auto l : laser_plane_in_camera)
			cout << l << ", ";
		cout << endl;
		//draw_laser_in_images(input_file.c_str(), output_file_path, laser_plane_in_camera, coordinate);
		reconstruct_test2(reconstruction_file_path.c_str(), camera_matrix, RT, laser_plane_in_camera, coordinate);
	}
#endif

	system("pause");
	return 0;
}
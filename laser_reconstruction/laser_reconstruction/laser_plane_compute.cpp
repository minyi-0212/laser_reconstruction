#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <direct.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include "Ransac.h"
#include "common.h"
#include "coor_system.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using namespace cv;

#define M_PI 3.14159265358979323846
#define output_drawAxis_pix
//#define MASK_SHOW
#define VIRTUAL

static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

// for overall new test
void undistort_images(const string& pattern_jpg, const Mat& cameraMatrix, const Mat& distCoeffs)
{
	//string pattern_jpg = "E:/mygu/laser/laser_plane/image/*.png";
	vector<String> image_files;
	cv::glob(pattern_jpg, image_files);

	_mkdir("./images");
	char path[_MAX_PATH];
	for (int i = 0; i < image_files.size(); i++)
	{
		Mat inputImage, imageCopy;
		inputImage = imread(image_files[i]);
		inputImage.copyTo(imageCopy);
		undistort(imageCopy, inputImage, cameraMatrix, distCoeffs);
		sprintf_s(path, "./images/dist_pose_%03d.png", atoi(
			(image_files[i].substr(image_files[i].find_last_of("_") + 1,
				image_files[i].find_last_of(".") - image_files[i].find_last_of("_") - 1).c_str())
		));
		cout << "undistort output: " << path << endl;
		imwrite(path, inputImage);
	}
}

int sovle_pnp(const Mat& inputImage, Mat& imageCopy, coor_system& coordinate_system,
	const cv::CommandLineParser& parser,
	int squaresX, int squaresY, float squareLength, float markerLength, float axisLength,
	const Mat& cameraMatrix, const Mat& distCoeffs)
{
	// aruco detectMarkers
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	cv::Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
	if (parser.has("dp"))
	{
		bool readOk = readDetectorParameters(parser.get<string>("dp"), parameters);
		if (!readOk) {
			std::cerr << "Invalid detector parameters file" << endl;
			return -1;
		}
	};
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

	// refind strategy to detect more markers
	// create charuco board object
	Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create(squaresX, squaresY,
		squareLength, markerLength, dictionary);

	/*Ptr<aruco::CharucoBoard> charucoboard_draw = aruco::CharucoBoard::create(12, 9,
		squareLength, markerLength, dictionary);
	cv::Mat boardImage;
	charucoboard_draw->draw(cv::Size(2224, 1668), boardImage, 10, 1);
	imwrite("test.png", boardImage);
	cv::imshow("boardImage", boardImage);
	cv::waitKey(0);*/

	if (parser.get<bool>("rs"))
	{
		cout << "refineDetectedMarkers..." << endl;
		Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();
		aruco::refineDetectedMarkers(inputImage, board, markerCorners, markerIds, rejectedCandidates,
			cameraMatrix, distCoeffs);
	}

	// interpolate charuco corners
	int interpolatedCorners = 0;
	vector<int> charucoIds;
	vector<Point2f> charucoCorners;
	if (markerIds.size() > 0)
		interpolatedCorners =
		aruco::interpolateCornersCharuco(markerCorners, markerIds, inputImage, charucoboard,
			charucoCorners, charucoIds, cameraMatrix, distCoeffs);
	if (charucoIds.size() < 4 || charucoCorners.size() < 4)
	{
		cout << " not enough charuco corners for pose detect" << endl;
		return -1;
	}

	// 3d-2d pose estimation
	bool validPose = false;
	Vec3d rvec, tvec;
	vector<Point2f> camera_coord;
	vector<Point3f> world_coord;
	for (int i = 0; i < charucoCorners.size(); i++)
	{
		//Point2f charucoCoord = charucoCorners[i];
		//Point2i charucoCoordi(int(charucoCoord.x + 0.5), int(charucoCoord.y + 0.5));
		//cout << "checkerFile: " << charucoCoordi << endl;
		//cout << "Point2f charucoCoord = " << charucoCoord << endl;
		//circle(imageCopy, charucoCoord, 3, Scalar(0, 0, 255), 3);
		//camera_coord.push_back(charucoCoord);
		camera_coord.push_back(charucoCorners[i]);
		int idx = charucoIds[i];
		int x = idx % 4;
		int y = idx / 4;
		Point3f world_sample(x, y, 0);
		//cout << "Point3f world_sample = " << world_sample << endl;
		world_coord.push_back(world_sample);
	}
	bool xequal = true, yequal = true;
	for (int i = 0; i < world_coord.size() - 1; i++)
	{
		if (world_coord[i].x != world_coord[i + 1].x)
		{
			xequal = false;
			break;
		}
	}
	for (int i = 0; i < world_coord.size() - 1; i++)
	{
		if (world_coord[i].y != world_coord[i + 1].y)
		{
			yequal = false;
			break;
		}
	}
	if (xequal || yequal)
	{
		cout << "detect world coord is a line, can't fit a plane" << endl;
		return -1;
	}
	validPose = solvePnP(world_coord, camera_coord, cameraMatrix, distCoeffs, rvec, tvec, false, 0);
	/*cout << "rvec = " << rvec << endl;
	cout << "tvec = " << tvec << endl << endl;*/
	coordinate_system.set_rt(rvec, tvec);

#ifdef output_drawAxis_pix
	if (validPose)
	{
		cout << " valid." << endl;
		cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, axisLength);
	}
	else
	{
		cout << " not valid." << endl;
		return -1;
	}
	if (markerIds.size() > 0)
		cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
	if (charucoIds.size() > 0)
		cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(0, 0, 255));
	//cv::resize(imageCopy, imageCopy, Size(), 0.5, 0.5, INTER_LINEAR);
	//cv::imshow("test", imageCopy);
	//cv::waitKey(0);
#endif
	return 0;
}

void compute_laser_line(const Mat& inputImage, const vector<Point2f>& mask_point_in_pixel,
	vector<Point2f>& laser, vector<Point3d>& laser_points_all_in_camera, coor_system& coordinate_system)
{
	Mat Mask;
	Mask.create(inputImage.size(), CV_8UC1);
	Mask.setTo(0);
	vector<Point> maskOuterRect;
	for (int i = 4; i < 8; i++)
		maskOuterRect.push_back(mask_point_in_pixel[i]);
	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < inputImage.cols; j++)
		{
			double inRect = pointPolygonTest(maskOuterRect, Point2f(j, i), false);
			if (inRect != -1)
			{
				Mask.at<uchar>(i, j) = 255;
			}
		}
	}

	vector<Point> maskInnerRect;
	for (int i = 0; i < 4; i++)
		maskInnerRect.push_back(mask_point_in_pixel[i]);
	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < inputImage.cols; j++)
		{
			double inWrap = pointPolygonTest(maskInnerRect, Point2f(j, i), false);
			if (inWrap != -1)
			{
				Mask.at<uchar>(i, j) = 0;
			}
		}
	}

	Mat maskImg;
	inputImage.copyTo(maskImg, Mask);
	//Mat undistImg;
	//undistort(maskImg, undistImg, intrinsic_matrix_loaded, distortion_coeffs_loaded);

	/*Mat maskShow;
	maskImg.copyTo(maskShow);
	for (int i = 0; i < twodvecMask.size(); i++)
		circle(maskShow, twodvecMask[i], 3, Scalar(0, 0, 255), 3);
	resize(Mask, Mask, Size(), 0.5, 0.5);
	resize(maskShow, maskShow, Size(), 0.5, 0.5);
	imshow("mask", Mask);
	imshow("mask_add", maskShow);*/

	vector<Mat> channels;
	split(maskImg, channels);
	Mat dst = channels[1];

	//vector<Point> laser;

	for (int i = 0; i < dst.rows; i++)
	{
		uchar val = 0;
		int maxj = 0;
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) > val)
			{
				val = dst.at<uchar>(i, j);
				maxj = j;
			}
		}
		if (val >= 120)
		{
			laser.push_back(Point2f(maxj, i));
		}
	}

	Vec4f line;
	if (laser.size() <= 2)
	{
		cout << "laser line cannot find." << endl;
		return;
	}
	fitLine(laser, line, CV_DIST_L2, 0, 0.01, 0.01);
	double k = line[1] / line[0],
		b = line[3] - k * line[2];
	coordinate_system.set_laser_line(k, b);
	vector<Point2f> tmp_points_in_pixel({ Point2d((100 - b) / k, 100), Point2d((2000 - b) / k, 2000) });
	vector<Point3f> tmp_points_in_camera, tmp_points_in_world;
	coordinate_system.pixel_to_world(tmp_points_in_pixel, tmp_points_in_world);
	coordinate_system.world_to_camera(tmp_points_in_world, tmp_points_in_camera);
	for (auto p : tmp_points_in_camera)
	{
		//cout << p << endl;
		laser_points_all_in_camera.push_back(p);
	}

	/*vector<Point2d> laser_points_in_pixel_coord;
	laser_points_in_pixel_coord.push_back(Point2d((100 - b) / k, 100));
	laser_points_in_pixel_coord.push_back(Point2d((2000 - b) / k, 2000));
	cout << laser_points_in_pixel_coord[0] << endl << laser_points_in_pixel_coord[1] << endl;
	cv::line(maskImg, laser_points_in_pixel_coord[0], laser_points_in_pixel_coord[1], Scalar(0, 255, 0), 2);
	cv::resize(maskImg, maskImg, Size(), 0.5, 0.5);
	cv::imshow("test", maskImg);
	cv::waitKey(0);*/
}

void compute_laser_line_virtual(const Mat& inputImage, const vector<Point2f>& mask_point_in_pixel,
	vector<Point2f>& laser, vector<Point3d>& laser_points_all_in_camera, coor_system& coordinate_system)
{
	Mat Mask;
	Mask.create(inputImage.size(), CV_8UC1);
	Mask.setTo(0);
	vector<Point> maskOuterRect;
	for (int i = 4; i < 8; i++)
		maskOuterRect.push_back(mask_point_in_pixel[i]);
	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < inputImage.cols; j++)
		{
			double inRect = pointPolygonTest(maskOuterRect, Point2f(j, i), false);
			if (inRect != -1)
			{
				Mask.at<uchar>(i, j) = 255;
			}
		}
	}

	/*vector<Point> maskInnerRect;
	for (int i = 0; i < 4; i++)
		maskInnerRect.push_back(mask_point_in_pixel[i]);
	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < inputImage.cols; j++)
		{
			double inWrap = pointPolygonTest(maskInnerRect, Point2f(j, i), false);
			if (inWrap != -1)
			{
				Mask.at<uchar>(i, j) = 0;
			}
		}
	}*/

	Mat maskImg;
	inputImage.copyTo(maskImg, Mask);
	//Mat undistImg;
	//undistort(maskImg, undistImg, intrinsic_matrix_loaded, distortion_coeffs_loaded);

#ifdef MASK_SHOW
	Mat maskShow;
	maskImg.copyTo(maskShow);
	for (int i = 0; i < mask_point_in_pixel.size(); i++)
		circle(maskShow, mask_point_in_pixel[i], 3, Scalar(0, 0, 255), 3);
	resize(Mask, Mask, Size(), 0.5, 0.5);
	resize(maskShow, maskShow, Size(), 0.5, 0.5);
	imshow("mask", Mask);
	imshow("mask_add", maskShow);
#endif

	vector<Mat> channels;
	split(maskImg, channels);
#ifndef VIRTUAL
	Mat dst = channels[1];
	//vector<Point> laser;

	for (int i = 0; i < dst.rows; i++)
	{
		uchar val = 0;
		int maxj = 0;
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) > val)
			{
				val = dst.at<uchar>(i, j);
				maxj = j;
			}
		}
		if (val >= 120)
		{
			laser.push_back(Point2f(maxj, i));
		}
	}
#endif

#ifdef VIRTUAL
	Mat dst = channels[2];
	//vector<Point> laser;

	for (int i = 0; i < dst.rows; i++)
	{
		uchar val = 0;
		int maxj = 0;
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) > val)
			{
				val = dst.at<uchar>(i, j);
				maxj = j;
			}
		}
		if (val >= 150)
		{
			laser.push_back(Point2f(maxj, i));
		}
	}
#endif

	Vec4f line;
	if (laser.size() <= 2)
	{
		cout << "laser line cannot find." << endl;
		return;
	}
	fitLine(laser, line, CV_DIST_L2, 0, 0.01, 0.01);
	double k = line[1] / line[0],
		b = line[3] - k * line[2];
	coordinate_system.set_laser_line(k, b);
	vector<Point2f> tmp_points_in_pixel/*({ Point2d((100 - b) / k, 100), Point2d((2000 - b) / k, 2000) })*/;
	for (int i = 0; i <= 2000; i+=300)
	{
		tmp_points_in_pixel.push_back(Point2d((i - b) * 1.0 / k, i));
	}
	vector<Point3f> tmp_points_in_camera, tmp_points_in_world;
	coordinate_system.pixel_to_world(tmp_points_in_pixel, tmp_points_in_world);
	coordinate_system.world_to_camera(tmp_points_in_world, tmp_points_in_camera);
	//cout << "before size:" << laser_points_all_in_camera.size() << endl;
	for (auto p : tmp_points_in_camera)
	{
		//cout << p << endl;
		laser_points_all_in_camera.push_back(p);
	}

#ifdef MASK_SHOW
	for (auto p : laser)
	{
		circle(maskImg, p, 2, Scalar(255, 255, 0), 2);
	}

	vector<Point2d> laser_points_in_pixel_coord;
	laser_points_in_pixel_coord.push_back(Point2d((100 - b) / k, 100));
	laser_points_in_pixel_coord.push_back(Point2d((2000 - b) / k, 2000));
	cout << laser_points_in_pixel_coord[0] << endl << laser_points_in_pixel_coord[1] << endl;
	cv::line(maskImg, laser_points_in_pixel_coord[0], laser_points_in_pixel_coord[1], Scalar(0, 255, 0), 2);
	cv::resize(maskImg, maskImg, Size(), 0.5, 0.5);
	cv::imshow("test", maskImg);
	cv::waitKey(0);
#endif
}

void plane_section(const vector<double>& a, const vector<double>& b, vector<Point3f>& p)
{
	double z1 = 30, z2 = 50,
		xd = a[0] * b[1] - a[1] * b[0],
		xz = a[1] * b[2] - a[2] * b[1],
		x0 = a[1] * b[3] - a[3] * b[1],
		yd = a[0] * b[1] - a[1] * b[0],
		yz = a[2] * b[0] - a[0] * b[2],
		y0 = a[3] * b[0] - a[0] * b[3];
	/*p.push_back( Point3f((xz*z1 + x0) / xd, (yz*z1 + y0) / yd, z1) );
	p.push_back( Point3f((xz*z2 + x0) / xd, (yz*z2 + y0) / yd, z2) );*/
	for (int z = 5; z < 100; z += 10)
		p.push_back(Point3f((xz*z + x0) / xd, (yz*z + y0) / yd, z));
}

void check_laser_plane(const vector<double>& laser_plane_in_camera, vector<coor_system>& coordinate)
{
	vector<Point3f> pos, normal, color;
	float axis_size = 1.0;
	double axis_value;
	// add coordinate
	{
		pos.push_back(Point3f(0, 0, 0));
		color.push_back(Point3f(0, 0, 0));
		for (int i = 1; i < 100; i++)
		{
			axis_value = i * axis_size;
			pos.push_back(Point3f(axis_value, 0, 0));
			color.push_back(Point3f(1, 0, 0));
			pos.push_back(Point3f(0, axis_value, 0));
			color.push_back(Point3f(0, 1, 0));
			pos.push_back(Point3f(0, 0, axis_value));
			color.push_back(Point3f(0, 0, 1));
		}
	}

	// add checkboard
	{
		vector<Point3f> tmp;
		for (int j = 0; j < 7; j++)
			for (int i = -50; i < -40; i++)
			{
				tmp.push_back(Point3f(i, j, 0));
			}
		vector<Point3f> pos_world;
		coordinate[0].world_to_camera(tmp, pos_world);
		for (int i = 0; i < pos_world.size(); i++)
		{
			//cout << pos[i] << "	->	" << pos_camera[i] << endl;
			pos.push_back(pos_world[i]);
			color.push_back(Point3f(1, 1, 0));
		}
	}

	// add laser plane
	{
		vector<Point3f> laser_point_in_camera;
		for (double i = -1; i <= 1; i += 0.1)
		{
			for (double j = -10; j <= 0; j += 0.1)
			{
				laser_point_in_camera.push_back(Point3f(i,
					-(laser_plane_in_camera[0] * i + laser_plane_in_camera[2] * j
						+ laser_plane_in_camera[3]) / laser_plane_in_camera[1], j));
			}
		}
		for (int j = 0; j < laser_point_in_camera.size(); j++)
		{
			pos.push_back(laser_point_in_camera[j]);
			color.push_back(Point3f(0, 1, 1));
		}
	}

	{
		for (int i = 0; i < 36; i+=40)
		{
				// add axis
				{
					vector<Point3f> tmp;
					tmp.push_back(Point3f(0, 0, 0));
					for (int i = 1; i < 100; i++)
					{
						axis_value = i * axis_size;
						tmp.push_back(Point3f(axis_value, 0, 0));
						tmp.push_back(Point3f(0, axis_value, 0));
						tmp.push_back(Point3f(0, 0, axis_value));
					}
					vector<Point3f> pos_world;
					coordinate[i].world_to_camera(tmp, pos_world);
					cout << "the pos of world origin (0,0,0) in camera = " << pos_world[0] << endl;
					pos.push_back(pos_world[0]);
					color.push_back(Point3f(0, 0, 0));
					//cout << pos[0] << "	->	" << pos_camera[0] << endl;
					for (int i = 0; i < 99; i++)
					{
						//cout << pos[i] << "	->	" << pos_camera[i] << endl;
						pos.push_back(pos_world[i * 3 + 1]);
						color.push_back(Point3f(0.5, 0, 0));
						pos.push_back(pos_world[i * 3 + 2]);
						color.push_back(Point3f(0, 0.5, 0));
						pos.push_back(pos_world[i * 3 + 3]);
						color.push_back(Point3f(0, 0, 0.5));
					}
				}
		}
	}
	export_pointcloud_ply("./output_virtual/camera_coord.ply", pos, normal, color);
}

#include "coor_system.h"
void compute_laser_plane_test(const cv::CommandLineParser& parser, const char filepath[],
	const Mat& cameraMatrix, const Mat& distCoeffs, 
	vector<double>& laser_plane_in_camera, vector<coor_system>& coordinate)
{
	int squaresX = parser.get<int>("w"),
		squaresY = parser.get<int>("h");
	float squareLength = parser.get<float>("sl"),
		markerLength = parser.get<float>("ml"),
		axisLength = 100 * ((float)max(squaresX, squaresY) * (squareLength));
	Mat one_distCoeffs = Mat::zeros(1, 5, CV_64FC1);
	vector<Point3f> mask_in_world, laser_points_in_world;
	{
#ifndef VIRTUAL
		mask_in_world.push_back(Point3f(-1.3, -1.25, 0));
		mask_in_world.push_back(Point3f(4.5, -1.25, 0));
		mask_in_world.push_back(Point3f(4.5, 6.35, 0));
		mask_in_world.push_back(Point3f(-1.3, 6.35, 0));
		mask_in_world.push_back(Point3f(-1.5, -3.85, 0));
		mask_in_world.push_back(Point3f(4.7, -3.85, 0));
		mask_in_world.push_back(Point3f(4.7, 8.95, 0));
		mask_in_world.push_back(Point3f(-1.5, 8.95, 0));
#endif

#ifdef VIRTUAL
		mask_in_world.push_back(Point3f(-1, -1, 0));
		mask_in_world.push_back(Point3f(4, -1, 0));
		mask_in_world.push_back(Point3f(4, 6, 0));
		mask_in_world.push_back(Point3f(-1, 6, 0));
		mask_in_world.push_back(Point3f(-1.5, -3.85, 0));
		mask_in_world.push_back(Point3f(4.7, -3.85, 0));
		mask_in_world.push_back(Point3f(4.7, 8.95, 0));
		mask_in_world.push_back(Point3f(-1.5, 8.95, 0));
#endif
	}

	char out_file_path[_MAX_PATH];

	_mkdir("./virtual");
	vector<String> image_files;
	cv::glob(filepath, image_files);
	//vector<coor_system> coordinate(image_files.size(), coor_system(cameraMatrix));
	coordinate.resize(image_files.size());
	for (int i = 0; i < coordinate.size(); i++)
		coordinate[i] = coor_system(cameraMatrix);

	vector<Point3d> laser_points_all_in_camera;
	int start_index = 0, end_index = image_files.size() - 1;
	//int start_index = 20, end_index = 30;
	for (int i = start_index; i <= end_index; i++)
	{
		//if (i % 20 == 0)
			cout << i << " : " << image_files[i] << endl;

		Mat inputImage, imageCopy;
		inputImage = imread(image_files[i]);
		//inputImage = imread("./images/dist_pose_120.png");
		//inputImage = imread("../image/pose_120.png");
		inputImage.copyTo(imageCopy);

		/*{
			vector<Point2d> pixels({
					Point2d(495, 1664),
					Point2d(0, 1000)
				}),
				pixels_out;
			my_undistort_points(pixels, pixels_out, cameraMatrix, distCoeffs);
			my_print(pixels_out);
			cv::line(imageCopy, pixels[0], pixels_out[0], Scalar(255, 0, 0), 3);
			cv::line(imageCopy, pixels[1], pixels_out[1], Scalar(0, 0, 255), 3);
		}*/

		if (sovle_pnp(inputImage, imageCopy, coordinate[i], parser,
			squaresX, squaresY, squareLength, markerLength, axisLength,
			cameraMatrix, one_distCoeffs))
			continue;

#ifdef output_drawAxis_pix
		sprintf_s(out_file_path, "./virtual/coord_checkboard_%03d.png", i);
		imwrite(out_file_path, imageCopy);
#endif
		// use to test the coordinate transfer
		/*{
			vector<Point3f> mask_in_world, mask_in_world_recovery, mask_in_camera, mask_in_camera_recovery;
			vector<Point2f> mask_in_pixel;
			mask_in_world.push_back(Point3f(-1.3, -1.25, 1));
			// world->pixel
			projectPoints(mask_in_world, coordinate[i].rvec, coordinate[i].tvec, cameraMatrix, one_distCoeffs, mask_in_pixel);
			cout << mask_in_world[0] << "	->	" << mask_in_pixel[0] << endl;
			// pixel->world
			coordinate[i].pixel_to_world(mask_in_pixel, mask_in_world_recovery);

			cout << endl;
			coordinate[i].pixel_to_camera(mask_in_pixel, mask_in_camera_recovery);
			coordinate[i].camera_to_pixel(mask_in_camera_recovery, mask_in_pixel);
		}*/
		vector<Point2f> mask_in_pixel;
		vector<Point2f> laser_points_in_pixel;
		coordinate[i].world_to_pixel(mask_in_world, mask_in_pixel, cameraMatrix, one_distCoeffs);
#ifndef VIRTUAL
		if(i>=20 && i<=30)
			compute_laser_line(inputImage, mask_in_pixel, laser_points_in_pixel, laser_points_all_in_camera, coordinate[i]);
#endif
#ifdef VIRTUAL
			compute_laser_line_virtual(inputImage, mask_in_pixel, laser_points_in_pixel, laser_points_all_in_camera, coordinate[i]); // for virtual
#endif
	}
	cout << "image rt compute ok. " << laser_points_all_in_camera.size() << endl;

	//vector<double> laser_plane_in_camera;
	vector<Point3f> pos, normal, color;
	{
		//float x, y, z;
		//for (int i = 1; i < laser_points_all_in_camera.size()-1; i++)
		//{
		//	cout << laser_points_all_in_camera[i] << endl;
		//	x = (laser_points_all_in_camera[i].x - laser_points_all_in_camera[i - 1].x) /
		//		(laser_points_all_in_camera[i + 1].x - laser_points_all_in_camera[i].x);
		//	y = (laser_points_all_in_camera[i].y - laser_points_all_in_camera[i - 1].y) /
		//		(laser_points_all_in_camera[i + 1].y - laser_points_all_in_camera[i].y);
		//	z = (laser_points_all_in_camera[i].z - laser_points_all_in_camera[i - 1].z) /
		//		(laser_points_all_in_camera[i + 1].z - laser_points_all_in_camera[i].z);
		//	cout << x / y << " " << y / z << endl;
		//	/*cout << laser_points_all_in_camera[i] <<" : "
		//		<< (laser_points_all_in_camera[i].x - laser_points_all_in_camera[i - 1].x)/
		//		(laser_points_all_in_camera[i+1].x - laser_points_all_in_camera[i].x) << ", "
		//		<< (laser_points_all_in_camera[i].y - laser_points_all_in_camera[i - 1].y)/
		//		(laser_points_all_in_camera[i+1].y - laser_points_all_in_camera[i].y) << ", "
		//		<< (laser_points_all_in_camera[i].z - laser_points_all_in_camera[i - 1].z)/
		//		(laser_points_all_in_camera[i+1].z - laser_points_all_in_camera[i].z) << endl;*/
		//}
		Ransac ransac_laser(laser_points_all_in_camera);
		laser_plane_in_camera/*({ 21.2851, 18.0615, -10.2061, 357.398 })*/ = ransac_laser.fitPlane();

		{
			pos.push_back(Point3f(0, 0, 0));
			color.push_back(Point3f(0, 0, 1));
			for (int i = 0; i < laser_points_all_in_camera.size(); i++)
			//for (int i = 60; i < 66; i++)
			{
				pos.push_back(laser_points_all_in_camera[i]);
				color.push_back(Point3f(1, 1, 0));
			}
		}

		check_laser_plane(laser_plane_in_camera, coordinate);
		return;
	}

	// compute the plane
	for (int image_index = start_index; image_index <= end_index; image_index++)
	{
		cout << image_index << endl;
		// draw the origin laser line of the image
		//int image_index = 20;
		Mat image = imread(image_files[image_index]);
		Point2f laser_point_pixel_1 = coordinate[image_index].get_laser_line_point(50),
			laser_point_pixel_2 = coordinate[image_index].get_laser_line_point(2000);
		line(image, laser_point_pixel_1, laser_point_pixel_2, Scalar(0, 0, 255), 5);

		// compute the image plane
		vector<Point3f> mask_in_camera;
		vector<Point3d> mask_in_camera_double;
		coordinate[image_index].world_to_camera(mask_in_world, mask_in_camera);
		for (auto p : mask_in_camera)
			mask_in_camera_double.push_back(p);
		/*cout << laser_points_all_in_camera.size() << endl;
		for (auto p : laser_points_all_in_camera)
		{
			cout << p << endl;
		}*/
		Ransac ransac_image(mask_in_camera_double);
		vector<double> image_plane_in_camera = ransac_image.fitPlane();
		/*{
			for (auto item : laser_plane_in_camera)
				cout << item << ", ";
			cout << endl;
			for (auto item : image_plane_in_camera)
				cout << item << ", ";
			cout << endl;
		}*/

		vector<Point3f> section_point_camera;
		vector<Point2f> section_point_pixel;
		plane_section(laser_plane_in_camera, image_plane_in_camera, section_point_camera);
		coordinate[image_index].camera_to_pixel(section_point_camera, section_point_pixel);
		/*for (int i = 0; i < section_point_camera.size(); i++)
			cout << section_point_camera[i] << endl
			<< section_point_pixel[i] << endl

			<< section_point_camera[i].x*laser_plane_in_camera[0]
			+ section_point_camera[i].y*laser_plane_in_camera[1] +
			section_point_camera[i].z*laser_plane_in_camera[2]
			+ laser_plane_in_camera[3] << endl

			<< section_point_camera[i].x*image_plane_in_camera[0]
			+ section_point_camera[i].y*image_plane_in_camera[1] +
			section_point_camera[i].z*image_plane_in_camera[2] +
			image_plane_in_camera[3] << endl << endl;*/

		if (image_index == 25)
		{
			for (auto p: section_point_camera)
			{
				pos.push_back(p);
				color.push_back(Point3f(1, 0, 0));
			}
		}

		line(image, section_point_pixel[0], section_point_pixel[section_point_pixel.size() - 1], Scalar(255, 0, 0), 3);
		//line(image, section_point_pixel[3], section_point_pixel[4], Scalar(255, 0, 0), 3);
		sprintf_s(out_file_path, "./virtual/dist_pose_%03d.png", atoi(
			(image_files[image_index].substr(image_files[image_index].find_last_of("_") + 1,
				image_files[image_index].find_last_of(".") - image_files[image_index].find_last_of("_") - 1).c_str())
		));
		cout <<"output: "<< out_file_path << endl;
		imwrite(out_file_path, image);
		/*resize(image, image, Size(), 0.5, 0.5);
		imshow("output", image);
		waitKey(0);*/
	}
	export_pointcloud_ply("./output_virtual/laser_plane_check.ply", pos, normal, color);
}

void check(const Mat& cameraMatrix, const Mat& RT)
{
	string pattern_jpg = "../virtual_checkboard/test_*.png";
	vector<String> image_files;
	cv::glob(pattern_jpg, image_files);

	_mkdir("./virtual");
	char path[_MAX_PATH];
	Mat in_homogeneous(4, 1, CV_64FC1), out_homogeneous(4, 1, CV_64FC1), inputImage,
		camera_matrix = (cv::Mat_<double>(4, 4) <<
			cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(0, 1), cameraMatrix.at<double>(0, 2), 0,
			cameraMatrix.at<double>(1, 0), cameraMatrix.at<double>(1, 1), cameraMatrix.at<double>(1, 2), 0,
			cameraMatrix.at<double>(2, 0), cameraMatrix.at<double>(2, 1), cameraMatrix.at<double>(2, 2), 0,
			0, 0, 0, 1);

	// rotate
	Point3f point(0, 1, -0.2);
	vector<Point3f> section;
	vector<double> a({ -0.707106829, -0.693375289, 0.138675064, 0.2*0.693375289 - 0.138675064 });
	{
		Point3f axis(0,0.2,1);
		double cosa = cos(-45 * M_PI / 180), sina = sin(-45 * M_PI / 180);
		Mat RT = (cv::Mat_<double>(3, 3) <<
			axis.x*axis.x*(1 - cosa) + cosa, axis.x*axis.y*(1 - cosa) + axis.z*sina, axis.x*axis.z*(1 - cosa) - axis.y*sina,
			axis.x*axis.y*(1 - cosa) - axis.z*sina, axis.y*axis.y*(1 - cosa) + cosa, axis.y*axis.z*(1 - cosa) + axis.x*sina,
			axis.x*axis.z*(1 - cosa) + axis.y*sina, axis.y*axis.z*(1 - cosa) - axis.x*sina, axis.z*axis.z*(1 - cosa) + cosa),
			tmp(3, 1, CV_64FC1);
		{
			cout << point << "->";
			tmp = RT * (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
			point.x = tmp.at<double>(0, 0);
			point.y = tmp.at<double>(1, 0);
			point.z = tmp.at<double>(2, 0);
			cout << point << endl;
		}
		Point3f nor = point.cross(axis);
		nor /= sqrt(nor.x*nor.x + nor.y*nor.y + nor.z*nor.z);
		cout << "laser plane:" << nor << " " << -nor.x*point.x - nor.y*point.y - nor.z*point.z << endl;
		cout << "laser plane2:" << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << endl;
		vector<double> laser({ nor.x , nor.y ,nor.z, -nor.x*point.x - nor.y*point.y - nor.z*point.z }),
			check_board({ 0.25,-1,0,0 });
		plane_section(laser, check_board, section);
		cout << "section[0] : " << section[0] << endl << section[0].x*0.25 - section[0].y << endl
			<< section[0].x*laser[0] + section[0].y*laser[1] + section[0].z*laser[2] + laser[3] << endl
			//<< -section[0].x*0.707 - (section[0].y-0.2)*0.703 + (section[0].z-1)*0.139 << endl
			<< -section[0].x*0.707 - section[0].y*0.703 + section[0].z*0.139 + 0.2*0.703 - 0.139
			<< endl;
	}

	Point2d pixel_coord;
	for (int i = 0; i < image_files.size(); i++)
	{
		inputImage = imread(image_files[i]);
		in_homogeneous.at<double>(0, 0) = 0;
		in_homogeneous.at<double>(1, 0) = 0;
		in_homogeneous.at<double>(2, 0) = 0;
		in_homogeneous.at<double>(3, 0) = 1;
		out_homogeneous = camera_matrix * RT * in_homogeneous;
		pixel_coord.x = out_homogeneous.at<double>(0, 0) / out_homogeneous.at<double>(2, 0);
		pixel_coord.y = out_homogeneous.at<double>(1, 0) / out_homogeneous.at<double>(2, 0);
		circle(inputImage, pixel_coord, 1, Scalar(0, 255, 0), 1);

		{
			section.clear();
			//vector<double> a({ nor.x , nor.y ,nor.z, -nor.x*point.x - nor.y*point.y - nor.z*point.z }),
			//vector<double> a({ -0.707106829, -0.693375289, 0.138675064, 0.2*0.693375289 - 0.138675064 }),
			vector<double> b({ 0.25*cos(M_PI * i / 18),-1,0.25*sin(M_PI * i / 18),0 });
			cout << "check: " << a[3] << endl;
			double z1 = 30, z2 = 50,
				xd = a[0] * b[1] - a[1] * b[0],
				xz = a[1] * b[2] - a[2] * b[1],
				x0 = a[1] * b[3] - a[3] * b[1],
				yd = a[0] * b[1] - a[1] * b[0],
				yz = a[2] * b[0] - a[0] * b[2],
				y0 = a[3] * b[0] - a[0] * b[3];
			for (double z = -0.1; z < 0.1; z += 0.05)
				section.push_back(Point3f((xz*z + x0) / xd, (yz*z + y0) / yd, z));
		}

		for (int i = 0; i < section.size(); i++)
		{
			in_homogeneous.at<double>(0, 0) = section[i].x;
			in_homogeneous.at<double>(1, 0) = section[i].y;
			in_homogeneous.at<double>(2, 0) = section[i].z;
			in_homogeneous.at<double>(3, 0) = 1;
			out_homogeneous = camera_matrix * RT * in_homogeneous;
			pixel_coord.x = out_homogeneous.at<double>(0, 0) / out_homogeneous.at<double>(2, 0);
			pixel_coord.y = out_homogeneous.at<double>(1, 0) / out_homogeneous.at<double>(2, 0);
			//cout << pixel_coord << endl;
			circle(inputImage, pixel_coord, 1, Scalar(0, 255, 0), 1);
		}

		sprintf_s(path, "./virtual/check_%03d.png", atoi(
			(image_files[i].substr(image_files[i].find_last_of("_") + 1,
				image_files[i].find_last_of(".") - image_files[i].find_last_of("_") - 1).c_str())
		));
		//cout << "output: " << path << endl;
		imwrite(path, inputImage);
	}
}
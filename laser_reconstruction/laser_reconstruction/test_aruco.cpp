#include "3d_reconstruction.h"
#include "Ransac.h"
#include "common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace std;
using namespace cv;
#define M_PI 3.14159265358979323846

int test_aruco(const Mat& cameraMatrix)
{
	//cv::VideoCapture inputVideo;
	//inputVideo.open(0);
	//cv::Mat cameraMatrix, distCoeffs;
	vector<String> image_files;
	cv::glob("./cube_checkboard/dist_pose_*.png", image_files);
	//cv::glob("./images2/dist_pose_*.png", image_files);
	Mat distCoeffs = Mat::zeros(1, 5, CV_64FC1);

	// camera parameters are read from somewhere
	//readCameraParameters(cameraMatrix, distCoeffs);
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(12, 9, 0.01, 0.006, dictionary);
	//cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.01, 0.006, dictionary);
	for(int i=0;i< image_files.size();i++)
	{
		cout << image_files[i] << endl;
		cv::Mat image, imageCopy;
		image = imread(image_files[i]);
		image.copyTo(imageCopy);
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners;
		cv::aruco::detectMarkers(image, dictionary, corners, ids);
		// if at least one marker detected
		if (ids.size() > 0) {
			std::vector<cv::Point2f> charucoCorners;
			std::vector<int> charucoIds;
			cv::aruco::interpolateCornersCharuco(corners, ids, image, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs);
			// if at least one charuco corner detected
			if (charucoIds.size() > 0) {
				cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
				cv::Vec3d rvec, tvec;
				bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);
				// if charuco pose is valid
				if (valid)
					cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
			}
		}
		cv::resize(imageCopy, imageCopy, Size(), 0.5, 0.5);
		cv::imshow("out", imageCopy);
		char key = (char)cv::waitKey(0);
		if (key == 27)
			break;
	}
}
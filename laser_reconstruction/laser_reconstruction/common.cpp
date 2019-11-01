#include "common.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

void my_undistort_points(const std::vector<cv::Point2d>& p_in, std::vector<cv::Point2d>& p_out,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
	undistortPoints(p_in, p_out, cameraMatrix, distCoeffs);
	for (int i = 0; i < p_out.size(); i++)
	{
		Mat tmp = Mat::zeros(3, 1, CV_64FC1);
		tmp.at<double>(0, 0) = p_out[i].x;
		tmp.at<double>(1, 0) = p_out[i].y;
		tmp.at<double>(2, 0) = 1;
		tmp = cameraMatrix * tmp;
		cout << tmp.at<double>(2, 0) << endl;
		p_out[i].x = tmp.at<double>(0, 0) / tmp.at<double>(2, 0);
		p_out[i].y = tmp.at<double>(0, 1) / tmp.at<double>(2, 0);
	}
}

void my_print(std::vector<cv::Point2d>& t)
{
	for (int i = 0; i < t.size(); i++)
	{
		cout << t[i] << " ";
	}
	cout << endl;
}

void export_pointcloud_ply(const char *filename,
	const std::vector<Point3f>& pos,
	const std::vector<Point3f>& p_normal,
	const std::vector<Point3f>& p_color)
{
	FILE *fp;

	fopen_s(&fp, filename, "wt");

	fprintf_s(fp, "ply\n");
	fprintf_s(fp, "format ascii 1.0\n");


	fprintf_s(fp, "element vertex %d\n", pos.size());
	fprintf_s(fp, "property float x\n");
	fprintf_s(fp, "property float y\n");
	fprintf_s(fp, "property float z\n");
	if (p_normal.size())
	{
		fprintf_s(fp, "property float nx\n");
		fprintf_s(fp, "property float ny\n");
		fprintf_s(fp, "property float nz\n");
	}
	if (p_color.size())
	{
		fprintf_s(fp, "property uchar red\n");
		fprintf_s(fp, "property uchar green\n");
		fprintf_s(fp, "property uchar blue\n");
		fprintf_s(fp, "property uchar alpha\n");
	}

	fprintf_s(fp, "end_header\n");

	for (int i = 0; i < pos.size(); i++)
	{
		fprintf_s(fp, "%g %g %g ", pos[i].x, pos[i].y, pos[i].z);

		if (p_normal.size())
		{
			fprintf_s(fp, "%g %g %g ", p_normal[i].x, p_normal[i].y, p_normal[i].z);
		}

		if (p_color.size())
		{
			int r = __max(__min(int(p_color[i].x * 255), 255), 0),
				g = __max(__min(int(p_color[i].y * 255), 255), 0),
				b = __max(__min(int(p_color[i].z * 255), 255), 0);
			fprintf_s(fp, "%d %d %d %d ", r, g, b, 255);
		}

		fprintf_s(fp, "\n");
	}

	fclose(fp);
}

void rename_file(const char path[], const char prifix[])
{
	char files[_MAX_PATH];
	sprintf_s(files, "%s/%s*.ppm", path, prifix);
	vector<String> image_files;
	cv::glob(files, image_files);
	for (int i = 0; i < image_files.size(); i++)
	{
		Mat inputImage;
		inputImage = imread(image_files[i]);
		sprintf_s(files, "%s/%s%03d.png",
			path, prifix,
			atoi((image_files[i].substr(image_files[i].find(prifix) + strlen(prifix),
				image_files[i].find_last_of(".") - image_files[i].find(prifix) - strlen(prifix)).c_str())));
		cout << "output: " << image_files[i] << endl << " => " << files << endl;
			
		/*cout << image_files[i] <<endl << image_files[i].find(prifix) + strlen(prifix) << " "
			<< image_files[i].find_last_of(".") - image_files[i].find(prifix) - strlen(prifix) << endl
			<< image_files[i].substr(image_files[i].find(prifix) + strlen(prifix),
			image_files[i].find_last_of(".") - image_files[i].find(prifix) - strlen(prifix)).c_str() << endl;*/
		imwrite(files, inputImage);
	}
}

void laser_points_find_analysis()
{
	Mat image = imread("./rabbit/dist_pose_009.png");
	ofstream outFile;
	outFile.open("./rabbit/rabbit_009_data.csv");
	int begin_line = 1738, end_line = begin_line + 10;
	cout << image.type() <<" " << CV_8UC3 << endl;
	double tmp;
	for (int j = begin_line; j < end_line; j++)
	{
		//for (int i = 0; i < image.cols; i++)
		for (int i = 1722; i < 2121; i++)
		{
			tmp = image.at<Vec3b>(j, i)[1];
			outFile << tmp << ",";
		}
		outFile << endl;
	}
	outFile.close();
}
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
	Mat image = imread("./squirrel/dist_pose_009.png");
	ofstream outFile;
	outFile.open("./squirrel/rabbit_009_data.csv");
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

// gaussian与求导相关
float f_x(float x, float miu, float xigma) {
	return 1 / sqrt(2 * 3.1415 * xigma * xigma)
		* exp(-(x - miu)*(x - miu) / (2 * xigma * xigma));
}

void gaussian(const int dim, const int xigma, const vector<float>& value, vector<float>& result) {
	vector<float> g(dim * 2 + 1);
	float sum_g = 0;
	for (int i = 0; i < g.size(); i++) {
		g[i] = f_x(i - dim, 0, xigma);
		sum_g += g[i];
	}
	for (int i = 0; i < g.size(); i++) {
		g[i] /= sum_g;
	}
	result.resize(value.size(), 0);

	if (g.size() > value.size()) {
		for (int i = 0; i < value.size(); i++) {
			float sum_gg = 0;
			vector<float> gg(value.size());
			for (int j = 0; j < gg.size(); j++) {
				if (dim + j - i < 0 || dim + j - i>2 * dim)
					gg[j] = 0;
				else
					gg[j] = g[dim + j - i];
				sum_gg += gg[j];
			}
			for (int j = 0; j < gg.size(); j++) {
				gg[j] /= sum_gg;
			}

			float re = 0;
			for (int j = 0; j < gg.size(); j++) {
				if (gg[j] != 0)
					re += value[j] * gg[j];
			}
			result[i] = re;
		}
		return;
	}

	for (int i = dim; i < value.size() - dim; i++) {
		float re = 0;
		for (int j = 0; j < g.size(); j++) {
			re += value[i + j - dim] * g[j];
		}
		result[i] = re;
	}

	// 重新计算边缘高斯滤波系数
	for (int i = 0; i < dim; i++) {
		float sum_gg = 0;
		vector<float> gg(g.size());
		for (int j = 0; j < g.size(); j++) {
			if (i + j < dim) {
				gg[j] = 0;
			}
			else {
				gg[j] = g[j];
			}
			sum_gg += gg[j];
		}
		for (int j = 0; j < gg.size(); j++) {
			gg[j] /= sum_gg;
		}

		float re = 0;
		for (int j = 0; j < gg.size(); j++) {
			if (gg[j] != 0)
				re += value[i + j - dim] * gg[j];
		}
		result[i] = re;
	}

	// 重新计算边缘高斯滤波系数
	for (int i = value.size() - dim; i < value.size(); i++) {
		float sum_gg = 0;
		vector<float> gg(g.size());
		for (int j = 0; j < g.size(); j++) {
			if (j - ((int)value.size() - 1 - i) > dim) {
				gg[j] = 0;
			}
			else {
				gg[j] = g[j];
			}
			sum_gg += gg[j];
		}
		for (int j = 0; j < gg.size(); j++) {
			gg[j] /= sum_gg;
		}

		float re = 0;
		for (int j = 0; j < gg.size(); j++) {
			if (gg[j] != 0)
				re += value[i + j - dim] * gg[j];
		}
		result[i] = re;
	}
}

// image rotate
void image_rotate(const Mat& src, Mat& dst, float angle, cv::Mat& rot_mat)
{
	// [x', y'] = rot * [x, y, 1].trans()
	cv::Point2f center(src.cols / 2, src.rows / 2);
	//cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1); //CV_64F, 2*3
	rot_mat = cv::getRotationMatrix2D(center, angle, 1); //CV_64F, 2*3
	cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();
	// 旋转中心为变换后的图片大小中心
	rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;
	cv::warpAffine(src, dst, rot_mat, bbox.size());

	// test re-transform	
	/*// (0,0)-> 2230.83775639315,0.8434878840091642
	// (100,100)-> 2223.436334481296, 142.0710310676344
	cout << rot_mat << endl << endl;
	cout <<"origin(100,100)-> "<< rot_mat * (Mat_<double>(3, 1) << 100,100, 1) << endl;
	Mat tmp = Mat::zeros(3, 3, CV_64F);
	tmp.at<double>(0, 0) = rot_mat.at<double>(0, 0);
	tmp.at<double>(0, 1) = rot_mat.at<double>(0, 1);
	tmp.at<double>(0, 2) = rot_mat.at<double>(0, 2);
	tmp.at<double>(1, 0) = rot_mat.at<double>(1, 0);
	tmp.at<double>(1, 1) = rot_mat.at<double>(1, 1);
	tmp.at<double>(1, 2) = rot_mat.at<double>(1, 2);
	tmp.at<double>(2, 2) = 1;
	cout <<"matrix: "<< tmp << endl << endl
		<< "matrix inv: " << tmp.inv() << endl << endl
		<< (tmp.inv() * (Mat_<double>(3, 1) << 2223.436334481296, 142.0710310676344, 1)) << endl;*/

	
	/*imwrite("./squirrel/tmp.png", dst);
	cv::imshow("src", src);
	resize(dst, dst, Size(), 0.25, 0.25);
	cv::imshow("dst", dst);
	cv::waitKey(0);*/
}
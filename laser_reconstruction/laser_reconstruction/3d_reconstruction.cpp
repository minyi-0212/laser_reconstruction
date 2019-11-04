#include "3d_reconstruction.h"
#include "Ransac.h"
#include "common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
#define M_PI 3.14159265358979323846

extern void check_laser_plane(const vector<double>& laser_plane_in_camera, vector<coor_system>& coordinate);

// compute the section of [laser plane] and [the camera_pos->pixel line]
// so the origin of line is (0,0,0), the distance frome the origin to the plane is d(plane is ax+by+cz+d=0).
void compute_plane_line_section(const vector<double>& plane, const vector <Point3f>& line_dir, vector<Point3f>& section_point)
{
	/*Vec3d lined(line_dir.x, line_dir.y, line_dir.z),
		n(plane[0], plane[1], plane[2]);
	lined = normalize(lined);
	double distance = plane[3] / norm(n);
	n = -normalize(n);
	double l = distance / n.dot(lined);
	section_point.x = l * lined[0];
	section_point.y = l * lined[1];
	section_point.z = l * lined[2];*/

	section_point.resize(line_dir.size());
	vector<Vec3d> lined(line_dir.size());
	Vec3d n(plane[0], plane[1], plane[2]);
	double distance = plane[3] / norm(n), l;
	n = -normalize(n);
	for (int i = 0; i < line_dir.size(); i++)
	{
		lined[i] = Vec3d(line_dir[i].x, line_dir[i].y, line_dir[i].z);
		lined[i] = normalize(lined[i]);
		l = distance / n.dot(lined[i]);
		section_point[i].x = l * lined[i][0];
		section_point[i].y = l * lined[i][1];
		section_point[i].z = l * lined[i][2];
	}
}

void find_the_max_point_of_each_line(const Mat& image, vector<Point2f>& laser_line)
{
	laser_line.clear();
	//cout << image.size() <<", c=" << image.channels() << end	aAq`1w23	`4l << image.type() << endl;
	double max_val;
	Point2f max_point;
	for (int j = 0; j < image.rows; j++)
	{
		max_val = 0;
		for (int i = 0; i < image.cols; i++)
		{
			if (image.at<Vec3b>(j, i)[1] > max_val)
			{
				max_val = image.at<Vec3b>(j, i)[1];
				max_point.x = i;
				max_point.y = j;
			}
		}
		if (max_val >= 120)
			laser_line.push_back(max_point);

		/*for (int i = 0; i < image.cols; i++)
		{
			if (image.at<Vec3b>(j, i)[1] > 0)
			{
				max_point.x = i;
				max_point.y = j;
				laser_line.push_back(max_point);
			}
		}*/
	}
}

void find_the_max_point_of_each_column(const Mat& image, vector<Point2f>& laser_line, const cv::Rect& region)
{
	/*laser_line.clear();
	double max_val;
	Point2f max_point;
	for (int i = region.x; i < region.x+region.width; i++)
	{
		max_val = 0;
		for (int j = region.y; j < region.y+region.height; j++)
		{
			if (image.at<Vec3b>(j, i)[1] > max_val)
			{
				max_val = image.at<Vec3b>(j, i)[1];
				max_point.x = i;
				max_point.y = j;
			}
		}
		if (max_val >= 120)
			laser_line.push_back(max_point);
	}*/

	laser_line.clear();
	double max_val;
	Point2f max_point;
	vector<float> image_vec(region.height, 0),
		image_result(region.height, 0);
	for (int i = 0; i < region.width; i++)
	{
		for (int k = 0; k < region.height; k++)
		{
			image_vec[k] = (uchar)image.at<Vec3b>(k + region.y, i + region.x)[1];
			//cout << image_vec[k] << ",";
		}
		//cout << endl;
		gaussian(3, 2, image_vec, image_result);
		max_val = 0;
		for (int j = 0; j < region.height; j++)
		{
			if (image_result[j] > max_val)
			{
				max_val = image_result[j];
				max_point.x = i + region.x;
				max_point.y = j + region.y;
			}
		}
		//cout <<"max value: "<< max_val<<endl;
		if (max_val >= 120)
			laser_line.push_back(max_point);
	}
}

// ÈÆaxisÖáÐý×ª
void rotate(const Point3f& axis, double angle, vector<Point3f>& points)
{
	/*cout << angle << endl;
	cv::Vec3d z(sin(angle * M_PI / 180), 0, cos(angle * M_PI / 180)),
		x(cos(angle * M_PI /180 ), 0, -sin(angle * M_PI / 180)),
		y(0, 1, 0);
	Mat RT_tmp = (cv::Mat_<double>(3, 3) <<
		x[0], x[1], x[2],
		y[0], y[1], y[2],
		z[0], z[1], z[2]);*/
	double cosa = cos(angle * M_PI / 180), sina = sin(angle * M_PI / 180);
	Mat RT = (cv::Mat_<double>(3, 3) <<
		axis.x*axis.x*(1 - cosa) + cosa, axis.x*axis.y*(1 - cosa) + axis.z*sina, axis.x*axis.z*(1 - cosa) - axis.y*sina,
		axis.x*axis.y*(1 - cosa) - axis.z*sina, axis.y*axis.y*(1 - cosa) + cosa, axis.y*axis.z*(1 - cosa) + axis.x*sina,
		axis.x*axis.z*(1 - cosa) + axis.y*sina, axis.y*axis.z*(1 - cosa) - axis.x*sina, axis.z*axis.z*(1 - cosa) + cosa),
		tmp(3, 1, CV_64FC1);
	//cout <<"rt: "<< RT << endl << "rt_tmp: " << RT_tmp << endl;
	for (auto& p : points)
	{
		//cout << p << "->";
		tmp = RT * (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
		p.x = tmp.at<double>(0, 0);
		p.y = tmp.at<double>(1, 0);
		p.z = tmp.at<double>(2, 0);
		//cout << p << endl;
	}
}

//#define OUTPUT_PLY
//#define cam_coord_ply
//#define real_world_coordinate
//#define laser_plane
//#define laser_plane_compute
//#define checkboard
//#define ball

void reconstruct_test(const char* filepath, const Mat& camera_matrix, const Mat& RT, const float rotate_angle)
{
	coor_system coordinate(camera_matrix);
	coordinate.set_RT_matrix(RT);
	// get_laser_plane
	vector<double> laser_plane_in_camera;
	{
		vector<Point3f> laser_point_in_world({
			/*Point3f(0, 0, 0),
			Point3f(-0.5, cos(rotate_angle), sin(rotate_angle)),
			Point3f(-0.2, cos(rotate_angle), sin(rotate_angle)),
			Point3f(-2, cos(rotate_angle), sin(rotate_angle)),
			Point3f(4, 2 * cos(rotate_angle), 2 * sin(rotate_angle))*/

			Point3f(0, 0, 0),
			Point3f(sin(rotate_angle),cos(rotate_angle), -0.5),
			Point3f(sin(rotate_angle), cos(rotate_angle), -0.2),
			Point3f(sin(rotate_angle), cos(rotate_angle), -2),
			Point3f(2 * sin(rotate_angle),2 * cos(rotate_angle), 4),
			Point3f(2 * sin(rotate_angle),2 * cos(rotate_angle), 10)

			/*Point3f(0, 0, 0),
			Point3f(0, 1, 1),
			Point3f(0, 1, -1),
			Point3f(0, -1, 1),
			Point3f(-0, 1, -1)*/
			}), laser_point_in_camera;
		coordinate.world_to_camera(laser_point_in_world, laser_point_in_camera);
		vector<Point3d> laser_point_in_camera_double;
		for (auto p : laser_point_in_camera)
		{
			laser_point_in_camera_double.push_back(p);
		}
		Ransac ransac_laser(laser_point_in_camera_double);
		laser_plane_in_camera = ransac_laser.fitPlane();
		cout << "laser plane of camera system:" << endl;
		for (int i = 0; i < laser_plane_in_camera.size(); i++)
			cout << laser_plane_in_camera[i] << " ";
		cout << endl;
		cout << "---------------------------------------------------" << endl;
	}

	// output ply
#ifdef OUTPUT_PLY
	{
		vector<Point3f> pos, normal, color;
		// add coordinate
		{
			pos.push_back(Point3f(0, 0, 0));
			color.push_back(Point3f(0, 0, 0));
			for (int i = 1; i < 100; i++)
			{
				pos.push_back(Point3f(i*0.01, 0, 0));
				color.push_back(Point3f(1, 0, 0));
				pos.push_back(Point3f(0, i*0.01, 0));
				color.push_back(Point3f(0, 1, 0));
				pos.push_back(Point3f(0, 0, i*0.01));
				color.push_back(Point3f(0, 0, 1));
			}
			vector<Point3f> pos_camera;
			coordinate.world_to_camera(pos, pos_camera);
			pos.push_back(pos_camera[0]);
			color.push_back(Point3f(0, 0, 0));
			//cout << pos[0] << "	->	" << pos_camera[0] << endl;
			for (int i = 0; i < 99; i++)
			{
				//cout << pos[i] << "	->	" << pos_camera[i] << endl;
				pos.push_back(pos_camera[i * 3 + 1]);
				color.push_back(Point3f(0.5, 0, 0));
				pos.push_back(pos_camera[i * 3 + 2]);
				color.push_back(Point3f(0, 0.5, 0));
				pos.push_back(pos_camera[i * 3 + 3]);
				color.push_back(Point3f(0, 0, 0.5));
			}
		}
		/*export_pointcloud_ply("./output_virtual/test_coor1.ply", pos, normal, color);
		return;*/

		// pixel -> camera
		{
			vector<Point3f> point_in_camera;
			vector<Point2f> point_in_pixel({ Point2f(512, 384) });
			point_in_pixel.push_back(Point2f(0, 0));
			point_in_pixel.push_back(Point2f(1024, 0));
			point_in_pixel.push_back(Point2f(0, 768));
			point_in_pixel.push_back(Point2f(1024, 768));
			for (int i = 10; i < 200; i += 10)
			{
				point_in_pixel.push_back(Point2f(512 + i, 384 + i));
				point_in_pixel.push_back(Point2f(512 + i, 384 - i));
				point_in_pixel.push_back(Point2f(512 - i, 384 + i));
				point_in_pixel.push_back(Point2f(512 - i, 384 - i));
			}
			coordinate.pixel_to_camera(point_in_pixel, point_in_camera);
			for (int i = 0; i < point_in_camera.size(); i++)
			{
				pos.push_back(point_in_camera[i]);
				color.push_back(Point3f(1, 1, 0));
			}

			// compute the section
			vector<Point3f> section_point;
			compute_plane_line_section(laser_plane_in_camera, point_in_camera, section_point);
			for (int i = 0; i < section_point.size(); i++)
			{
				pos.push_back(section_point[i]);
				color.push_back(Point3f(1, 0, 1));
			}
		}
		// add laser plane
		{
			vector<Point3f> laser_point_in_world({
					Point3f(0, 0, 1),
					Point3f(sin(rotate_angle),cos(rotate_angle),  1),
					Point3f(sin(rotate_angle),cos(rotate_angle),  1),
					Point3f(sin(rotate_angle),cos(rotate_angle),  1),
					Point3f(2 * sin(rotate_angle),2 * cos(rotate_angle),  1)
				}), laser_point_in_camera;
			for (double i = 0.005; i < 2; i+=0.01)
			{
				laser_point_in_world.push_back(Point3f(0, 0, -i));
				for (double j = -0.25; j < 2; j += 0.01)
				{
					/*laser_point_in_world.push_back(Point3f(-j, i * cos(rotate_angle), i * sin(rotate_angle)));
					laser_point_in_world.push_back(Point3f(-j, -i * cos(rotate_angle), -i * sin(rotate_angle)));*/
					laser_point_in_world.push_back(Point3f(i * sin(rotate_angle), i * cos(rotate_angle), -j));
					laser_point_in_world.push_back(Point3f(-i * sin(rotate_angle), -i * cos(rotate_angle), -j));
				}
			}
			/*vector<Point3f> laser_point_in_world({ Point3f(0, 0, 0) }), laser_point_in_camera;
			for (double x = 0.5; x > -2; x -= 0.1)
			{
				for (double z = -5; z < 5; z += 0.1)
				{
					laser_point_in_world.push_back(Point3f(x, 0, z));
				}
			}*/
			coordinate.world_to_camera(laser_point_in_world, laser_point_in_camera);
			cout << laser_point_in_camera.size() << endl;
			for (int i = 0; i < laser_point_in_camera.size(); i++)
			{
				pos.push_back(laser_point_in_camera[i]);
				color.push_back(Point3f(0, 1, 1));
			}
		}
		export_pointcloud_ply("./output_virtual/test_coor2.ply", pos, normal, color);
		return;
	}
#endif

	// 3d-reconstruction
	{
		char file[_MAX_PATH];
		Mat image, image_show;
		vector<Point2f> laser_line_point;
		vector<Point3f> laser_line_point_in_camera, section_point, laser_line_point_in_world,
			pos, normal, color;

		for (int i = 0; i < 1; i++)
		{
			//sprintf_s(file, "%s/test%d.ppm", filepath, 0);
			sprintf_s(file, "%s/ball_000.png", filepath);
			image = imread(file);
			//image_show = image.clone();

			// find the max value of each line
			find_the_max_point_of_each_line(image, laser_line_point);					// need to fix to the real!!!
			//cout << laser_line_point.size() << endl;
			/*for(auto p: laser_line_point)
				circle(image_show, p, 2, Scalar(0,0,255));
			imshow("tmp", image_show);
			waitKey(0);*/

			coordinate.pixel_to_camera(laser_line_point, laser_line_point_in_camera);
			compute_plane_line_section(laser_plane_in_camera, laser_line_point_in_camera, section_point);
			coordinate.camera_to_world(section_point, laser_line_point_in_world);
			//rotate(Point3f(0,cos(1),sin(1)), i * 10, laser_line_point_in_world);		// need to fix to the real!!!
			rotate(Point3f(0, 0, 1), i * 10, laser_line_point_in_world);				// need to fix to the real!!!
			coordinate.world_to_camera(laser_line_point_in_world, laser_line_point_in_camera);
			// output
			for (int j = 0; j < laser_line_point_in_camera.size(); j++)
			{
				pos.push_back(laser_line_point_in_camera[j]);
				color.push_back(Point3f(i / 36.0, (i % 24) / 24.0, (i % 12) / 12.0));
				//color.push_back(Point3f(1, 1, 1));
			}
		}
		export_pointcloud_ply("./output_virtual/reconstruction_virtual.ply", pos, normal, color);
	}
}

//#define FIND_LASER
void reconstruct_test2(const char* filepath, const Mat& camera_matrix, const Mat& RT,
	const vector<double>& laser_plane_in_camera, vector<coor_system>& coordinate)
{
	char file[_MAX_PATH];
	Mat image, image_show;
	vector<Point2f> laser_line_point;
	vector<Point3f> laser_line_point_in_camera, section_point, laser_line_point_in_world,
		pos, normal, color;

#ifdef OUTPUT_PLY
	//check_laser_plane(laser_plane_in_camera, coordinate);

	// add coordinate
	vector<Point3f> tmp_axis;
	{
		float axis_size = 1;
		double axis_value;
		tmp_axis.push_back(Point3f(0, 0, 0));
		for (int i = 1; i < 100; i++)
		{
			axis_value = i * axis_size;
			tmp_axis.push_back(Point3f(axis_value, 0, 0));
			tmp_axis.push_back(Point3f(0, axis_value, 0));
			tmp_axis.push_back(Point3f(0, 0, axis_value));
		}

#ifdef cam_coord_ply
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
		export_pointcloud_ply("./output_virtual/camera_coord_big.ply", pos, normal, color);
		return;
#endif
#ifdef checkboard
		// add checkboard
		axis_size = 1;
		{
			vector<Point3f> tmp;
			for (int j = 0; j < 7; j++)
				for (int i = 0; i < 5; i++)
				{
					tmp.push_back(Point3f(i, j, 0));
				}
			vector<Point3f> pos_world;
			coordinate[0].world_to_camera(tmp, pos_world);
			for (int i = 0; i < pos_world.size(); i++)
			{
				pos.push_back(pos_world[i]);
				color.push_back(Point3f(1, 1, 0));
			}

			coordinate[0].world_to_camera(tmp_axis, pos_world);
			pos.push_back(pos_world[0]);
			color.push_back(Point3f(0, 0, 0));
			for (int i = 0; i < 99; i++)
			{
				pos.push_back(pos_world[i * 3 + 1]);
				color.push_back(Point3f(0.5, 0, 0));
				pos.push_back(pos_world[i * 3 + 2]);
				color.push_back(Point3f(0, 0.5, 0));
				pos.push_back(pos_world[i * 3 + 3]);
				color.push_back(Point3f(0, 0, 0.5));
			}
		}

		export_pointcloud_ply("./output_virtual/checkboard-pic1.ply", pos, normal, color);
		return;
#endif

#ifdef real_world_coordinate
		// add real_world_coordinate
		{
			vector<Point3f> pos_world;
			coor_system real_coordinate(camera_matrix);
			real_coordinate.set_RT_matrix(RT);
			real_coordinate.world_to_camera(tmp_axis, pos_world);
			for (int i = 0; i < 99; i++)
			{
				//cout << pos[i] << "	->	" << pos_camera[i] << endl;
				pos.push_back(pos_world[i * 3 + 1]);
				color.push_back(Point3f(1, 0, 0));
				pos.push_back(pos_world[i * 3 + 2]);
				color.push_back(Point3f(0, 1, 0));
				pos.push_back(pos_world[i * 3 + 3]);
				color.push_back(Point3f(0, 0, 1));
			}
		}
		export_pointcloud_ply("./output_virtual/real_world_coord.ply", pos, normal, color);
		return;
#endif

#ifdef laser_plane
		// add laser plane
		{
			vector<Point3f> laser_point_in_camera;
			for (double i = -10; i <= 10; i += 1)
			{
				for (double j = -100; j <= 100; j += 1)
				{
					laser_point_in_camera.push_back(Point3f(i,
						-(laser_plane_in_camera[0] * i + laser_plane_in_camera[2] * j
							+ laser_plane_in_camera[3]) / laser_plane_in_camera[1], j));
				}
			}
			/*for (int j = 0; j < laser_point_in_camera.size(); j++)
			{
				pos.push_back(laser_point_in_camera[j]);
				color.push_back(Point3f(0, 1, 1));
			}*/
			vector<Point3f> pos_world;
			coor_system real_coordinate(camera_matrix);
			real_coordinate.set_RT_matrix(RT);
			real_coordinate.world_to_camera(laser_point_in_camera, pos_world);
			for (int j = 0; j < pos_world.size(); j++)
			{
				pos.push_back(pos_world[j]);
				color.push_back(Point3f(0, 1, 1));
			}

			cout << laser_point_in_camera.size() << endl;
			vector<Point3d> pos_world_double;
			for (auto p : pos_world)
				pos_world_double.push_back(p);
			Ransac ransac_laser(pos_world_double);
			vector<double> plane;
			plane = ransac_laser.fitPlane();
			cout << "plane in cam: " << endl;
			for (auto val : plane)
				cout << val << ",";
			cout << endl;
		}
		export_pointcloud_ply("./output_virtual/laser_plane.ply", pos, normal, color);
		return;
#endif

#ifdef laser_plane_compute
		// add laser plane
		{
			vector<Point3f> laser_point_in_camera;
			for (double i = -10; i <= 10; i += 1)
			{
				for (double j = -100; j <= 100; j += 1)
				{
					laser_point_in_camera.push_back(Point3f(i,
						-(laser_plane_in_camera[0] * i + laser_plane_in_camera[2] * j
							+ laser_plane_in_camera[3]) / laser_plane_in_camera[1], j));
				}
			}
			for (int j = 0; j < laser_point_in_camera.size(); j++)
			{
				pos.push_back(laser_point_in_camera[j]);
				color.push_back(Point3f(1, 0, 1));
			}
		}
		export_pointcloud_ply("./output_virtual/laser_plane_compute.ply", pos, normal, color);
		return;
#endif

#ifdef ball
		// add laser plane
		{
			vector<Point3f> ball_in_world, ball_in_cam;
			double radius = 0.04 / 0.0206, limit, z;
			for (double x = -radius; x <= radius; x += radius/10)
			{
				limit = sqrt(radius * radius - x * x);
				for (double y = -limit; y <= limit; y += radius / 10)
				{
					z = sqrt(radius * radius - x * x - y * y);
					ball_in_world.push_back(Point3f(x, y, z));
					ball_in_world.push_back(Point3f(x, y, -z));
				}
			}
			for (int j = 0; j < ball_in_world.size(); j++)
			{
				pos.push_back(ball_in_world[j]);
				color.push_back(Point3f(0, 1, 1));
			}
			coor_system real_coordinate(camera_matrix);
			real_coordinate.set_RT_matrix(RT);
			real_coordinate.world_to_camera(ball_in_world, ball_in_cam);
			for (int j = 0; j < ball_in_cam.size(); j++)
			{
				pos.push_back(ball_in_cam[j]);
				color.push_back(Point3f(1, 0, 1));
			}
		}
		export_pointcloud_ply("./output_virtual/ball_virtual.ply", pos, normal, color);
		return ;
#endif
	}
#endif

	sprintf_s(file, "%s/dist_pose_*.png", filepath);
	vector<String> image_files;
	cv::glob(file, image_files);
	for (int i = 0; i < image_files.size(); i++)
	{
#ifdef OUTPUT_PLY
		{
			// add axis
			{
				vector<Point3f> pos_world;
				coordinate[i].world_to_camera(tmp_axis, pos_world);
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
#endif

		//sprintf_s(file, "%s/test_%03d.png", filepath, 0);
		//sprintf_s(file, "%s/dist_pose_%03d.png", filepath, i);
		cout << image_files[i] << endl;
		image = imread(image_files[i]);
#ifdef FIND_LASER
		image_show = image.clone();
#endif
		// find the max value of each line
		//find_the_max_point_of_each_line(image, laser_line_point);					// need to fix to the real!!!
		Rect region(1511, 1335, 720, 547);
		find_the_max_point_of_each_column(image, laser_line_point, region);
		//cout << laser_line_point.size() << endl;
#ifdef FIND_LASER
		//cout << laser_line_point.size() << endl;
		for (auto p : laser_line_point)
			circle(image_show, p, 4, Scalar(0, 0, 255), 2);
		resize(image_show, image_show, Size(image_show.cols / 2, image_show.rows / 2));
		imshow("tmp", image_show);
		waitKey(0);
		continue;
#endif

		coordinate[i].pixel_to_camera(laser_line_point, laser_line_point_in_camera);
		compute_plane_line_section(laser_plane_in_camera, laser_line_point_in_camera, section_point);
		//coordinate[i].camera_to_world(section_point, laser_line_point_in_world);
		////rotate(Point3f(0,cos(1),sin(1)), i * 10, laser_line_point_in_world);		// need to fix to the real!!!
		//rotate(Point3f(0, 0, 1), i * 10, laser_line_point_in_world);				// need to fix to the real!!!
		//coordinate[i].world_to_camera(laser_line_point_in_world, laser_line_point_in_camera);
		coordinate[i].camera_to_world(section_point, laser_line_point_in_world);
		coordinate[0].world_to_camera(laser_line_point_in_world, laser_line_point_in_camera);

		// output
		cout << section_point[0] << "->" << laser_line_point_in_world[0] << endl;
		for (int j = 0; j < laser_line_point_in_camera.size(); j++)
		{
			pos.push_back(laser_line_point_in_camera[j]);
			color.push_back(Point3f(i / 40.0, (i % 24) / 24.0, (i % 12) / 12.0));
			//color.push_back(Point3f(1, 1, 1));
		}
	}
	sprintf_s(file, "%s/reconstruction.ply", filepath);
	export_pointcloud_ply(file, pos, normal, color);
}
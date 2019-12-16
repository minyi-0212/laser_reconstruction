#include "Ransac.h"

//vector<double> Ransac::fitCircle()
//{
//	double mindist = 0.01;
//	vector<int> counts;
//	vector<vector<double>> params;
//	for (int i = 0; i < 1000000; i++)
//	{
//		random_shuffle(Coords.begin(), Coords.end());
//		assert(Coords.size() >= 3);
//
//		Point3d p1 = Coords[0];
//		Point3d p2 = Coords[1];
//		Point3d p3 = Coords[2];
//		Vec3d p12 = p2 - p1;
//		Vec3d p13 = p3 - p1;
//	
//		Vec3d n = p12.cross(p13);
//		Vec3d n12 = n.cross(p12);
//		Vec3d n13 = p13.cross(n);
//
//		Point3d m12((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);
//		Point3d m13((p1.x + p3.x) / 2, (p1.y + p3.y) / 2, (p1.z + p3.z) / 2);
//
//		Mat left(2, 2, CV_64FC1);
//		Mat right(2, 1, CV_64FC1);
//		Mat lineCross(2, 1, CV_64FC1);
//		left.at<double>(0, 0) = n12[0];
//		left.at<double>(0, 1) = -n13[0];
//		left.at<double>(1, 0) = n12[1];
//		left.at<double>(1, 1) = -n13[1];
//		Mat invLeft;
//		invert(left, invLeft);
//		right.at<double>(0, 0) = m13.x - m12.x;
//		right.at<double>(1, 0) = m13.y - m12.y;
//		lineCross = invLeft * right;
//		double t = lineCross.at<double>(0, 0);
//		double s = lineCross.at<double>(1, 0);
//		double x = m12.x + t * n12[0];
//		double y = m12.y + t * n12[1];
//		double z = m12.z + t * n12[2];
//		Point3d center(x, y, z);
//		Vec3d r = p1 - center;
//		double rd = sqrt(r.dot(r));
//		
//		vector<double> param;
//		param.push_back(x);
//		param.push_back(y);
//		param.push_back(z);
//		param.push_back(rd);
//		params.push_back(param);
//		int count = 0;
//		for (int j = 0; j < Coords.size(); j++)
//		{
//			Point3d coord = Coords[j];
//			Vec3d rt = coord - center;
//			double dist = sqrt(rt.dot(rt));
//			if (fabs(dist - rd) <= mindist)
//			{
//				count++;
//			}
//		}
//		counts.push_back(count);
//	}
//	int maxcount = 0;
//	int idx = 0;
//	for (int i = 0; i < counts.size(); i++)
//	{
//		if (counts[i] > maxcount)
//		{
//			maxcount = counts[i];
//			idx = i;
//		}
//	}
//	cout << "maxcount = " << maxcount << endl;
//	return params[idx];
//}

vector<double> Ransac::fitPlane()
{
	vector<vector<double>> planes;
	vector<int> counts;
	double mindist = 0.001, maxdist=-1;
	int cnt = 0;
	for (int i = 0; i < 1000000; i++)
	{
		random_shuffle(Coords.begin(), Coords.end());
		assert(Coords.size() >= 3);
		
		Point3d p1 = Coords[0];
		Point3d p2 = Coords[1];
		Point3d p3 = Coords[2];
		Vec3d p12 = p2 - p1;
		Vec3d p13 = p3 - p1;
		p12 = normalize(p12);
		p13 = normalize(p13);
		
		if (abs(p12.dot(p13)) > 0.8) // deg < 36.86бу
		{
			//cout << abs(p12.dot(p13)) << endl;
			cnt++;
			continue;
		}

		Vec3d n = p12.cross(p13);
		double a = n[0];
		double b = n[1];
		double c = n[2];
		double d = -(a * p1.x + b * p1.y + c * p1.z);

		vector<double> plane;
		plane.push_back(a);
		plane.push_back(b);
		plane.push_back(c);
		plane.push_back(d);
		planes.push_back(plane);
		int count = 0;
		for (int j = 0; j < Coords.size(); j++)
		{
			Point3d coord = Coords[j];
			double dist = fabs(a * coord.x + b * coord.y + c * coord.z + d) / sqrt(a*a+b*b+c*c);
			if (dist <= mindist)
			{
				count++;
			}
		}
		counts.push_back(count);
	}

	int maxcount = 0;
	int idx = 0;
	for (int i = 0; i < counts.size(); i++)
	{
		if (counts[i] > maxcount)
		{
			maxcount = counts[i];
			idx = i;
		}
	}
	double a = planes[idx][0];
	double b = planes[idx][1];
	double c = planes[idx][2];
	double d = planes[idx][3];
	for (int j = 0; j < Coords.size(); j++)
	{
		Point3d coord = Coords[j];
		double dist = fabs(a * coord.x + b * coord.y + c * coord.z + d) / sqrt(a*a + b * b + c * c);
		if (dist > maxdist)
		{
			maxdist = dist;
		}
	}
	cout <<"miss "<<cnt<<","<< "maxcount = " << maxcount << ", maxdist = " << maxdist << endl;
	return planes[idx];
}

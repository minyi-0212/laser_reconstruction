#pragma once
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Ransac {
private:
	vector<Point3d> Coords;

public:
	Ransac(vector<Point3d> _Coords) :Coords(_Coords) {};
	//vector<double> fitCircle();
	vector<double> fitPlane();
};
#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#define PI acos(-1)
using namespace Eigen;
using namespace std;
using namespace cv;

int main()
{
    Mat a;
    Mat srcImage = imread("C:/Users/13992/Desktop/12.png"); 
    imshow("Test OpenCV", srcImage);
    waitKey(0);
    MatrixXf point(3, 1);
    MatrixXf transformation(3, 3);
    point << 2,
             1,
             1;
    transformation << cos(45.0 / 180.0 * PI), -sin(45.0 / 180.0 * PI), 1,
                      sin(45.0 / 180 * PI),   cos(45.0 / 180 * PI),    2,
                      0,                      0,                       1;
    cout << transformation * point << endl;
    system("pause");
    return 0;
}
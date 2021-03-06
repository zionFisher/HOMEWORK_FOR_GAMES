#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#define PI acos(-1)
using namespace Eigen;
using namespace std;

int main()
{
    MatrixXf point(3, 1);
    MatrixXf transformation(3, 3);
    point << 2,
             1,
             1;
    transformation << cos(45.0 / 180.0 * PI), -sin(45.0 / 180.0 * PI), 1,
                      sin(45.0 / 180 * PI),   cos(45.0 / 180 * PI),    2,
                      0,                      0,                       1;
    cout << transformation * point;
    system("pause");
    return 0;
}                  
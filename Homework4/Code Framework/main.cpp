#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    if (control_points.size() == 2)
        return control_points[0] + t * (control_points[1] - control_points[0]);

    std::vector<cv::Point2f> new_control_points;
    for (int count = 0; count < control_points.size() - 1; count++)
    {
        float x = (control_points[count] + t * (control_points[count + 1] - control_points[count])).x;
        float y = (control_points[count] + t * (control_points[count + 1] - control_points[count])).y;
        new_control_points.emplace_back(x, y);
    }

    return recursive_bezier(new_control_points, t);
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    float frame_buf[700][700] = {0};

    // Antialiasing
    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        cv::Point2f point = recursive_bezier(control_points, t);

        float x = point.x;
        float y = point.y;
        int xi = static_cast<int>(x);
        int yi = static_cast<int>(y);

        if (x - xi < 0.5 && y - yi < 0.5)
        {

        }
        else if (x - xi < 0.5 && y - yi > 0.5)
        {
            yi++;
        }
        else if (x - xi > 0.5 && y - yi < 0.5)
        {
            xi++;
        }
        else if (x - xi > 0.5 && y - yi > 0.5)
        {
            xi++; yi++;
        }

        cv::Point2i RTpoint(xi,     yi);
        cv::Point2i RBpoint(xi,     yi - 1);
        cv::Point2i LTpoint(xi - 1, yi);
        cv::Point2i LBpoint(xi - 1, yi - 1);

        float RTdistToPoint = sqrt((x - RTpoint.x) * (x - RTpoint.x) + (y - RTpoint.y) * (y - RTpoint.y));
        float RBdistToPoint = sqrt((x - RBpoint.x) * (x - RBpoint.x) + (y - RBpoint.y) * (y - RBpoint.y));
        float LTdistToPoint = sqrt((x - LTpoint.x) * (x - LTpoint.x) + (y - LTpoint.y) * (y - LTpoint.y));
        float LBdistToPoint = sqrt((x - LBpoint.x) * (x - LBpoint.x) + (y - LBpoint.y) * (y - LBpoint.y));

        if ((1 - RTdistToPoint / sqrt(2)) * 255 > frame_buf[RTpoint.x][RTpoint.y])
            frame_buf[RTpoint.x][RTpoint.y] = (1 - RTdistToPoint / sqrt(2)) * 255;
        if ((1 - RBdistToPoint / sqrt(2)) * 255 > frame_buf[RBpoint.x][RBpoint.y])
            frame_buf[RBpoint.x][RBpoint.y] = (1 - RBdistToPoint / sqrt(2)) * 255;
        if ((1 - LTdistToPoint / sqrt(2)) * 255 > frame_buf[LTpoint.x][LTpoint.y])
            frame_buf[LTpoint.x][LTpoint.y] = (1 - LTdistToPoint / sqrt(2)) * 255;
        if ((1 - LBdistToPoint / sqrt(2)) * 255 > frame_buf[LBpoint.x][LBpoint.y])
            frame_buf[LBpoint.x][LBpoint.y] = (1 - LBdistToPoint / sqrt(2)) * 255;

        float RTpercentage = 255;
        float RBpercentage = frame_buf[RBpoint.x][RBpoint.y];
        float LTpercentage = frame_buf[LTpoint.x][LTpoint.y];
        float LBpercentage = frame_buf[LBpoint.x][LBpoint.y];

        cv::Vec3b RTcolor(RTpercentage, RTpercentage, RTpercentage);
        cv::Vec3b RBcolor(RBpercentage, RBpercentage, RBpercentage);
        cv::Vec3b LTcolor(LTpercentage, LTpercentage, LTpercentage);
        cv::Vec3b LBcolor(LBpercentage, LBpercentage, LBpercentage);

        window.at<cv::Vec3b>(RTpoint) = RTcolor;
        window.at<cv::Vec3b>(RBpoint) = RBcolor;
        window.at<cv::Vec3b>(LTpoint) = LTcolor;
        window.at<cv::Vec3b>(LBpoint) = LBcolor;

        // not antialiasing
        // window.at<cv::Vec3b>(y, x)[0] = 255;
        // window.at<cv::Vec3b>(y, x)[1] = 255;
        // window.at<cv::Vec3b>(y, x)[2] = 255;
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            //naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}

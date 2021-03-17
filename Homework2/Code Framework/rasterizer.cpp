// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v) // 做叉积即可
{
    Vector3f AB = {_v[1].x() - _v[0].x(), _v[1].y() - _v[0].y(), 0};
    Vector3f BC = {_v[2].x() - _v[1].x(), _v[2].y() - _v[1].y(), 0};
    Vector3f CA = {_v[0].x() - _v[2].x(), _v[0].y() - _v[2].y(), 0};
    Vector3f AP = {x - _v[0].x(), y - _v[0].y(), 0};
    Vector3f BP = {x - _v[1].x(), y - _v[1].y(), 0};
    Vector3f CP = {x - _v[2].x(), y - _v[2].y(), 0};
    if (AP.cross(AB).z() < 0 && BP.cross(BC).z() < 0 && CP.cross(CA).z() < 0) // 由于是 AP 叉乘 AB，因此要得到均得到负值才代表在三角形内
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)  
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

float max(float a, float b)
{
    return (a > b ? a : b);
}

float min(float a, float b)
{
    return (a > b ? b : a);
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) 
{
    auto v = t.toVector4();
    int xmax = max(t.v[0].x(), max(t.v[1].x(), t.v[2].x())),
        xmin = min(t.v[0].x(), min(t.v[1].x(), t.v[2].x())), 
        ymax = max(t.v[0].y(), max(t.v[1].y(), t.v[2].y())),
        ymin = min(t.v[0].y(), min(t.v[1].y(), t.v[2].y()));

    // using 4xMSAA 
    for (int x = xmin; x <= xmax; x++)
    {
        for (int y = ymin; y <= ymax; y++)
        {
            insideFlag = false; // insideFlag 代表像素任意1/4是否位于三角形内
            depthFlag = false; // depthFlag 代表像素任意1/4深度是否小于原深度
            float X[4] = {x + 0.0f, x + 0.5f, x + 0.0f, x + 0.5f}; // 像素四等分，加四个0.0f是因为不想看到warning
            float Y[4] = {y + 0.0f, y + 0.0f, y + 0.5f, y + 0.5f};
            int INDEX[4] = {get_MSAA_index(X[0], Y[0]), // 获取 MSAA_depth_buf 和 MSAA_frame_buf 的 index
                            get_MSAA_index(X[1], Y[1]),
                            get_MSAA_index(X[2], Y[2]),
                            get_MSAA_index(X[3], Y[3])};
            
            for (int count = 0; count < 4; count++) // 遍历单像素的每个1/4
            {
                if (insideTriangle(X[count], Y[count], t.v))
                {
                    insideFlag = true;
                    set_depth_and_frame(X[count], Y[count], v, t, INDEX[count]); // 如果在三角形内就进行插值运算和深度判读
                }
            }
            if (insideFlag  == true && depthFlag == true) // 如果在三角形内且深度更小
            {
                auto color = (MSAA_frame_buf[INDEX[0]] + MSAA_frame_buf[INDEX[1]] + MSAA_frame_buf[INDEX[2]] + MSAA_frame_buf[INDEX[3]]) / 4;
                set_pixel(Vector3f(x, y, 1.0f), color); // color 是每个1/4取平均值后的颜色，这样才不会出现黑边
            }
        }
    }
}

void rst::rasterizer::set_depth_and_frame(float x, float y, std::array<Eigen::Vector4f, 3> v, const Triangle &t, int index)
{
    auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v); // 插值运算，从auto[alpha, beta, gamma]到*= w_reciprocal;的代码由代码框架给出
    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    z_interpolated *= w_reciprocal;
    if (z_interpolated < MSAA_depth_buf[index]) // 如果当前深度小于原深度
    {
        depthFlag = true;
        MSAA_depth_buf[index] = z_interpolated; // 更新深度
        MSAA_frame_buf[index] = t.getColor(); // 更新屏幕像素信息
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(MSAA_frame_buf.begin(), MSAA_frame_buf.end(), Eigen::Vector3f{0, 0, 0}); // 填充颜色值
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(MSAA_depth_buf.begin(), MSAA_depth_buf.end(), std::numeric_limits<float>::infinity()); // 填充深度值
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    MSAA_depth_buf.resize(4 * w * h); // 定义大小
    MSAA_frame_buf.resize(4 * w * h); 
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - 1 - y) * width + x;
}

int rst::rasterizer::get_MSAA_index(float x, float y)
{
    return (2 * height - 1 - 2 * y) * (2 * width) + 2 * x; // MSAA 的 index 推算结果是将原数据都乘 2
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;

}

// clang-format on
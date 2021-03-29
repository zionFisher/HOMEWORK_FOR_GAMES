#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection;
    Eigen::Matrix4f pToO, ortho, scale, translate;
    float t = tan((eye_fov/ 180.0 * MY_PI) / 2) * fabs(zNear);
    float r = aspect_ratio * t;
    float b = -t, l = -r;

    // 使用原矩阵会使三角形倒过来，这是由闫老师的采用的坐标系不同导致的:
    // pToO << zNear, 0,     0,            0,
    //         0,     zNear, 0,            0,
    //         0,     0,     zNear + zFar, -zNear * zFar,
    //         0,     0,     1,            0;
    pToO << -zNear, 0,      0,            0,
            0,      -zNear, 0,            0,
            0,      0,      zNear + zFar, -zNear * zFar,
            0,      0,      1,            0;

    scale << 2 / (r - l), 0,           0,                  0,
             0,           2 / (t - b), 0,                  0,
             0,           0,           2 / (zNear - zFar), 0,
             0,           0,           0,                  1;

    translate << 1, 0, 0, -(r + l) / 2,
                 0, 1, 0, -(t + b) / 2,
                 0, 0, 1, -(zNear + zFar) / 2,
                 0, 0, 0, 1;

    ortho = scale * translate;
    projection = ortho * pToO;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload) // 默认 shader
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()); // 用 texture coordinate （纹理坐标）找寻纹理颜色
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        Eigen::Vector3f l = (light.position - point).normalized();
        Eigen::Vector3f v = (eye_pos - point).normalized();
        Eigen::Vector3f I = light.intensity;
        float nl = normal.dot(l);
        float nh = normal.dot((v + l).normalized());
        float r2 = (light.position - point).squaredNorm();

        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        Eigen::Vector3f diffuse = kd.cwiseProduct(I / r2 * (nl < 0 ? 0.0 : nl));
        Eigen::Vector3f specular = ks.cwiseProduct(I / r2 * (std::pow(nh, p) < 0 ? 0.0 : std::pow(nh, p)));

        result_color += (ambient + diffuse + specular);
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; // 使用原生颜色
    Eigen::Vector3f point = payload.view_pos; // 接收点的坐标
    Eigen::Vector3f normal = payload.normal; // 接收点的法向量 normal

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights) // 对于每一束光
    {
        Eigen::Vector3f l = (light.position - point).normalized(); // 光源向量light
        Eigen::Vector3f v = (eye_pos - point).normalized(); // 视角向量view
        Eigen::Vector3f I = light.intensity; // 光强
        float nl = normal.dot(l); // normal dot product l
        float nh = normal.dot((v + l).normalized()); // normal dot product h, h = (v + l).normalized()
        float r2 = (light.position - point).squaredNorm(); // light.position - point 是由接收点指向光源的向量，squareNorm函数将这个向量的每一个分量的平方求和得到浮点数r2

        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity); // cwiseProduct函数将两向量每一个分量对应相乘 e.g: (1, 2, 3) cwiseProduct (3, 1, 2) = (3, 2, 6)
        Eigen::Vector3f diffuse = kd.cwiseProduct(I / r2 * (nl < 0 ? 0.0 : nl));
        Eigen::Vector3f specular = ks.cwiseProduct(I / r2 * (std::pow(nh, p) < 0 ? 0.0 : std::pow(nh, p)));

        result_color += (ambient + diffuse + specular);
    }

    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    float x = normal.x(),
          y = normal.y(),
          z = normal.z();

    Eigen::Vector3f t = {x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z)};
    Eigen::Vector3f b = normal.cross(t);
    Eigen::Matrix<Vector3f, 1, 3> TBN = {t, b, normal};
    auto dU = kh * kn * (payload.texture->getColor(payload.tex_coords.x() + 1.0 / payload.texture->width, payload.tex_coords.y()).norm() - payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()).norm());
    auto dV = kh * kn * (payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y() + 1.0 / payload.texture->height).norm() - payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()).norm());
    Vector3f ln = {-dU, -dV, 1};
    point = point + kn * normal * payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()).norm();
    normal = (TBN[0] * ln[0] + TBN[1] * ln[1] + TBN[2] * ln[2]).normalized();

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        Eigen::Vector3f l = (light.position - point).normalized();
        Eigen::Vector3f v = (eye_pos - point).normalized();
        Eigen::Vector3f I = light.intensity;
        float nl = normal.dot(l);
        float nh = normal.dot((v + l).normalized());
        float r2 = (light.position - point).squaredNorm();

        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        Eigen::Vector3f diffuse = kd.cwiseProduct(I / r2 * (nl < 0 ? 0.0 : nl));
        Eigen::Vector3f specular = ks.cwiseProduct(I / r2 * (std::pow(nh, p) < 0 ? 0.0 : std::pow(nh, p)));

        result_color += (ambient + diffuse + specular);
    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    float x = normal.x(),
          y = normal.y(),
          z = normal.z();

    Eigen::Vector3f t = {x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z)};
    Eigen::Vector3f b = normal.cross(t);
    Eigen::Matrix<Vector3f, 1, 3> TBN = {t, b, normal};
    auto dU = kh * kn * (payload.texture->getColor(payload.tex_coords.x() + 1.0 / payload.texture->width, payload.tex_coords.y()).norm() - payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()).norm());
    auto dV = kh * kn * (payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y() + 1.0 / payload.texture->height).norm() - payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()).norm());
    Vector3f ln = {-dU, -dV, 1};
    normal = (TBN[0] * ln[0] + TBN[1] * ln[1] + TBN[2] * ln[2]).normalized();

    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg"; // using "spot_texture.png" for texture_fragment_shader
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = normal_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}

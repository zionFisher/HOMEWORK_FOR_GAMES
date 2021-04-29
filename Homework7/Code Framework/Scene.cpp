//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"
#include <cmath>
#include <cstdlib>

void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    Intersection intersection = Scene::intersect(ray);

    if(!intersection.happened)
    {
        return Vector3f(0.0, 0.0, 0.0);
    }
    
    if(depth == 0 && intersection.m->hasEmission())
    {
        return intersection.m->getEmission();
    }

    Vector3f p = intersection.coords;
    Vector3f N = intersection.normal.normalized();
    Vector3f wo = (-ray.direction).normalized();
    Material* m = intersection.m;

    Intersection intercetionLight;
    float pdf_light;
    sampleLight(intercetionLight, pdf_light);

    Vector3f x = intercetionLight.coords;
    Vector3f ws = (x - p).normalized();
    Vector3f NN = intercetionLight.normal.normalized();
    Vector3f emit = intercetionLight.emit;

    bool block = (intersect(Ray(p, ws)).coords - x).norm() > EPSILON;

    Vector3f L_dir(0.0);

    if (!block)
    {
        L_dir = emit * m->eval(wo, ws, N) * dotProduct(ws, N) * dotProduct(-ws, NN)
                / (((x - p).norm() * (x - p).norm()) * pdf_light);
    }

    Vector3f L_indir(0.0);

    float seed = (double)rand() / RAND_MAX;
    if (seed < RussianRoulette)
    {
        Vector3f wi = m->sample(wo, N).normalized();
        Ray reflectRay(p, wi);
        Intersection reflectInter = intersect(reflectRay);
        if (reflectInter.happened && !reflectInter.m->hasEmission())
        {
            L_indir = castRay(reflectRay, depth + 1) * m->eval(wo, wi, N) * dotProduct(wi, N)
                      / (m->pdf(wo, wi, N) * RussianRoulette);
        }
    }

    return L_dir + L_indir;
}
#ifndef QUAD_HPP
#define QUAD_HPP

#include <tuple> 
#include <vector>
#include <opencv2/opencv.hpp>

class Model;  

class Quad {
public:
    Quad(Model& model, std::tuple<int, int, int, int> box, int depth);

    bool is_leaf();
    int compute_area();
    std::vector<Quad> split();
    std::vector<Quad*> get_leaf_nodes(int max_depth);

    // Custom assignment operator declaration
    Quad& operator=(const Quad& other);

    // Getter methods for private members
    double getError() const { return error; }
    int getArea() const { return area; }

private:
    Model& model;
    std::tuple<int, int, int, int> box;
    int depth;
    std::tuple<int, int, int> color;
    double error;
    bool leaf;
    int area;
    std::vector<Quad> children;

    std::tuple<int, int, int> color_from_histogram(const std::vector<int>& hist);
};

class Model {
public:
    Model(const std::string& path);

    std::vector<Quad*> get_quads() const;
    double get_average_error() const;
    cv::Mat getIm() const { return im; }
    Quad* pop();
    void split();

private:
    cv::Mat im;
    int width;
    int height;
    std::vector<std::tuple<bool, double, Quad*>> heap;
    Quad root;
    double error_sum;

    void push(Quad* quad);
};

#endif // QUAD_HPP

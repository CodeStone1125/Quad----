#ifndef QUAD_HPP
#define QUAD_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <queue>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

const double AREA_POWER = 2.0;
const int LEAF_SIZE = 10;  // You can adjust the leaf size accordingly

class Quad;

class Model {
public:
    Model(const std::string& path);
    const std::vector<Quad>& getQuads() const;
    double averageError() const;
    void push(Quad quad);
    Quad pop();
    void split();

// private... maybe... later...
    cv::Mat im;
    std::vector<Quad> heap;
    Quad root;
    double error_sum;
    int width;
    int height;
};

class Quad {
public:
    Quad(Model& model, std::tuple<int, int, int, int> box, int depth);
    bool is_leaf();
    double compute_area();
    std::vector<Quad> split();
    std::vector<Quad> get_leaf_nodes(int max_depth);

    Model& m_model;
    std::tuple<int, int, int, int> m_box;
    int m_depth;
    std::vector<int> hist;
    std::tuple<int, int, int> m_color;
    double m_error;
    bool m_leaf;
    double m_area;
    std::vector<Quad> children;
};

class Quad {
public:
    // Constructor
    Quad(cv::Mat model, std::tuple<int, int, int, int> box, int depth);

    // Member functions
    bool is_leaf();
    double compute_area();
    std::vector<Quad> split();
    std::vector<Quad> get_leaf_nodes(int max_depth);

// private... maybe... later...
    // Helper functions
    std::vector<int> calculate_histogram_cv(const cv::Mat& image);
    std::tuple<int, int, int> color_from_histogram(const std::vector<int>& hist);
    
    // Member variables
    cv::Mat m_model;
    std::tuple<int, int, int, int> m_box;
    int m_depth;
    std::vector<int> hist;
    std::tuple<int, int, int> m_color;
    double m_error;
    bool m_leaf;
    double m_area;
    std::vector<Quad> children;
};

#endif // QUAD_HPP

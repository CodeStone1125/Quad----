#ifndef MODEL_HPP
#define MODEL_HPP

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include <queue>

class Quad;

class Model {
public:
    cv::Mat im;
    int width;
    int height;
    std::vector<std::tuple<int, double, Quad*>> heap;
    Quad* root;
    double error_sum;

    Model(const std::string& path);
    std::vector<Quad*> quads();
    double average_error();
    void push(Quad* quad);
    Quad* pop();
    void split();
};


class Quad {
public:
    Quad(Model& model, const cv::Rect& box, int depth);

    bool is_leaf();
    double compute_area();
    std::vector<Quad> split();
    std::vector<Quad> get_leaf_nodes(int max_depth);
    Model& m_model;
    std::tuple<int, int, int, int> m_box;
    int m_depth;
    std::vector<Quad> children;
    bool leaf;
    cv::Mat hist;
    // Helper functions
    bool within_leaf_size();
};

#endif // MODEL_HPP

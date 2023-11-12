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
    // Constructor
    Quad(cv::Mat model, std::tuple<int, int, int, int> box, int depth);

    // Destructor (if needed)

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

#endif // MODEL_HPP

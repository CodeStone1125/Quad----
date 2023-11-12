#include "quad.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

const int MODE_RECTANGLE = 1;
const int MODE_ELLIPSE = 2;
const int MODE_ROUNDED_RECTANGLE = 3;

const int MODE = MODE_ELLIPSE;
const int ITERATIONS = 128;
const int LEAF_SIZE = 4;
const int PADDING = 1;
const cv::Scalar FILL_COLOR = cv::Scalar(0, 0, 0);
const bool SAVE_FRAMES = false;
const double ERROR_RATE = 0.5;
const double AREA_POWER = 0.25;
const double OUTPUT_SCALE = 1.0;

 /* 
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
*/
//Implementation of Model

// Constructor of Model
Model(const std::string& path) {
    // Load image from path
    im = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Error: Unable to read the image from " + imagePath);
    }
    // Covert from image to RGB
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);

    // Get the image size
    int width = im.cols;
    int height = im.rows;

    // Create empty heap
    std::vector<int> heap;

    // Not covert to cpp part
    // Self.root = Quad(self, (0, 0, self.width, self.height), 0)
    // root = Quad(0, 0, width, height);
    // self.error_sum = self.root.error * self.root.area
    // self.push(self.root)
}

/* 
class Quad {
public:
    Quad(Model& model, const cv::Rect& box, int depth);
    cv::Mat hist;
    bool is_leaf();
    double compute_area();
    std::vector<Quad> split();
    std::vector<Quad> get_leaf_nodes(int max_depth);
    Model& model;
    std::tuple<int, int, int, int> box;
    int depth;
    std::vector<Quad> children;
    bool leaf;

    // Helper functions
    bool within_leaf_size();
};
*/
// Set up parameters for the histogram
int histSize = 256;  // Number of bins
float range[] = {0, 256};  // Range of pixel values
const float* histRange = {range};
bool uniform = true;
bool accumulate = false;

quad(Model model, tuple<int, int, int, int> box, int depth) {
    m_model = model
    m_box = box
    m_depth = depth
    // Compute the histogram

    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    // hist = self.model.im.crop(self.box).histogram()
    // self.color, self.error = color_from_histogram(hist)
    // self.leaf = self.is_leaf()
    // self.area = self.compute_area()
    // self.children = []
}
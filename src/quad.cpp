#include "quad.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

// Function to calculate weighted average
extern "C" std::tuple<double, double> weighted_average(const std::vector<double>& hist) {
    double total = 0.0;
    double weighted_sum = 0.0;
    double squared_error_sum = 0.0;

    for (size_t i = 0; i < hist.size(); ++i) {
        total += hist[i];
        weighted_sum += i * hist[i];
    }

    double value = (total != 0.0) ? (weighted_sum / total) : 0.0;

    for (size_t i = 0; i < hist.size(); ++i) {
        squared_error_sum += hist[i] * std::pow(value - i, 2);
    }

    double error = (total != 0.0) ? std::sqrt(squared_error_sum / total) : 0.0;

    return std::make_tuple(value, error);
}

// CPP version histogram()
extern "C" std::vector<int> calculate_histogram_cv(const cv::Mat& rgb_image) {
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);

    // Initialize a vector to store histograms for each channel
    std::vector<double> hist_cv_channels;

    // Calculate histogram for each channel
    for (int channel = 0; channel < rgb_image.channels(); ++channel) {
        cv::Mat hist_channel;
        int histSize[] = {256};
        float range[] = {0, 256};
        const float* histRange[] = {range};
        cv::calcHist(&channels[channel], 1, nullptr, cv::Mat(), hist_channel, 1, histSize, histRange, true, false);

        // Extend the vector with the histogram values
        hist_cv_channels.insert(hist_cv_channels.end(), hist_channel.begin<float>(), hist_channel.end<float>());
    }

    // Convert the values to integers
    std::vector<int> hist_cv_channels_int;
    for (float value : hist_cv_channels) {
        hist_cv_channels_int.push_back(static_cast<int>(value));
    }
    return hist_cv_channels_int;
}

// Function to calculate color and luminance from histogram
extern "C" std::tuple<std::tuple<int, int, int>, double> color_from_histogram(const std::vector<int>& hist) {
    // Calculate weighted averages for each channel
    auto red_average = weighted_average(std::vector<double>(hist.begin(), hist.begin() + 256));
    auto green_average = weighted_average(std::vector<double>(hist.begin() + 256, hist.begin() + 512));
    auto blue_average = weighted_average(std::vector<double>(hist.begin() + 512, hist.end()));

    // Convert float values to integers by rounding
    int r = static_cast<int>(std::round(std::get<0>(red_average)));
    int g = static_cast<int>(std::round(std::get<0>(green_average)));
    int b = static_cast<int>(std::round(std::get<0>(blue_average)));

    // Calculate luminance
    double luminance = std::get<1>(red_average) * 0.2989 +
                       std::get<1>(green_average) * 0.5870 +
                       std::get<1>(blue_average) * 0.1140;
    return std::make_tuple(std::make_tuple(r, g, b), luminance);
}

//Cpp version crop(image, box)
cv::Mat cropImage(const cv::Mat& originalImage, const std::tuple<int, int, int, int> box) {
    // Extract values from the tuple
    int x, y, width, height;
    std::tie(x, y, width, height) = box;

    // Create a rectangle to define the region of interest (ROI)
    cv::Rect roi(x, y, width, height);

    // Crop the image using the defined ROI
    return originalImage(roi).clone();
}



/* Implementation of Model */ 
Model::Model(const std::string& path)
    : im(cv::imread(path)),
      width(im.cols),
      height(im.rows),
      heap(),
      root(new Quad(*this, std::make_tuple(0, 0, width, height), 0)),
      error_sum(root->m_error * root->m_area) {
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    push(*root);
}



const std::vector<Quad>& Model::getQuads() const {
        // Convert the priority_queue to a vector
        quads_vector = convertPriorityQueueToVector(heap);
        return quads_vector;
}

double Model::averageError() const {
    return error_sum / (width * height);
}

void Model::push(Quad& quad) {
    double score = -quad.m_error * std::pow(quad.m_area, AREA_POWER);
    heap.push(std::make_tuple(quad.m_leaf, score, quad));

    // Reconstruct the priority_queue to maintain the min heap property
    auto tempContainer = heap;
    std::vector<std::tuple<int, double, Quad>> heapCopy;
    while (!tempContainer.empty()) {
        heapCopy.push_back(tempContainer.top());
        tempContainer.pop();
    }

    std::make_heap(heapCopy.begin(), heapCopy.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // Compare elements based on score
    });

    // Reconstruct the priority_queue from the updated heapCopy
    heap = std::priority_queue<std::tuple<int, double, Quad>, std::vector<std::tuple<int, double, Quad>>, CompareQuad>(
        heapCopy.begin(), heapCopy.end()
    );
}

Quad Model::pop() {
    // Reconstruct the priority_queue to maintain the min heap property
    auto tempContainer = heap;
    std::vector<std::tuple<int, double, Quad>> heapCopy;
    while (!tempContainer.empty()) {
        heapCopy.push_back(tempContainer.top());
        tempContainer.pop();
    }

    std::pop_heap(heapCopy.begin(), heapCopy.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // Compare elements based on score
    });

    Quad quad = std::get<2>(heapCopy.front());

    // Remove the top element from the heapCopy
    heapCopy.pop_back();

    std::sort_heap(heapCopy.begin(), heapCopy.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // Compare elements based on score
    });

    // Reconstruct the priority_queue from the updated heapCopy
    heap = std::priority_queue<std::tuple<int, double, Quad>, std::vector<std::tuple<int, double, Quad>>, CompareQuad>(
        heapCopy.begin(), heapCopy.end()
    );

    return quad;
}

void Model::split() {
    Quad quad = pop();
    error_sum -= quad.m_error * quad.m_area;
    
    std::vector<Quad> children = quad.split();
    
    for (Quad& child : children) {
        push(child);
        error_sum += child.m_error * child.m_area;
    }
}/* End of Model implementation*/ 

// Implement of Quad
Quad::Quad(Model& model, std::tuple<int, int, int, int> box, int depth)
    : m_model(&model), m_box(box), m_leaf(is_leaf()),  hist(calculate_histogram_cv(cropImage(m_model->im, m_box))), m_area(compute_area()){
    // In the Quad constructor
    auto result = color_from_histogram(hist);
    // Unpack the tuple into m_color and m_error
    std::tie(m_color, m_error) = result;
    children = {};  // Initialize children directly
}


bool Quad::is_leaf() const{
    int l, t, r, b;
    std::tie(l, t, r, b) = m_box;
    return (r - l <= LEAF_SIZE || b - t <= LEAF_SIZE);
}

double Quad::compute_area() {
    int l, t, r, b;
    std::tie(l, t, r, b) = m_box;
    return static_cast<double>((r - l) * (b - t));
}

std::vector<Quad> Quad::split() {
    float l, t, r, b;
    std::tie(l, t, r, b) = m_box;  // 使用 m_box

    float lr = l + (r - l) / 2;
    float tb = t + (b - t) / 2;
    int depth = m_depth + 1;  // 使用 m_depth

    Quad tl(*m_model, std::make_tuple(l, t, lr, tb), depth);  // 使用 m_model
    Quad tr(*m_model, std::make_tuple(lr, t, r, tb), depth);  // 使用 m_model
    Quad bl(*m_model, std::make_tuple(l, tb, lr, b), depth);  // 使用 m_model
    Quad br(*m_model, std::make_tuple(lr, tb, r, b), depth);  // 使用 m_model

    // 使用 std::vector 返回新創建的 Quad 對象
    return {tl, tr, bl, br};
}

std::vector<Quad> Quad::get_leaf_nodes(int max_depth) const {
    std::vector<Quad> leaves;

    // 使用 is_leaf_node() 和 m_depth 來確定是否是葉子節點
    if (is_leaf() || m_depth >= max_depth) {
        leaves.push_back(*this);  // 如果是葉子節點，將當前節點加入結果中
    } else {
        // 遞迴調用 get_leaf_nodes
        for (const auto& child : children) {
            auto child_leaves = child.get_leaf_nodes(max_depth);
            leaves.insert(leaves.end(), child_leaves.begin(), child_leaves.end());
        }
    }

    return leaves;
}

void cpp_callback1(bool i, std::string id, py::array_t<uint8_t>& img)
{ 
    //auto im = img.unchecked<3>();
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;
    cv::Mat img2(rows, cols, type, (unsigned char*)img.data());
    cv::imshow(id, img2);
    cv::waitKey(0); // 等待用戶按下任意按鍵，以便保持顯示視窗
}




PYBIND11_MODULE(quad, m) {
    m.doc() = "Quad image compressor";

    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&>(), "Constructor for the Model class.")
        .def("getQuads", &Model::getQuads, "Get the quads from the model.")
        .def("averageError", &Model::averageError, "Calculate the average error of the model.")
        .def("push", &Model::push, "Push a quad into the model.")
        .def("pop", &Model::pop, "Pop a quad from the model.")
        .def("split", &Model::split, "Split the model into quads.");

    py::class_<Quad>(m, "Quad")
        .def(py::init<Model&, std::tuple<int, int, int, int>, int>(), "Constructor for the Quad class.")
        .def("is_leaf", &Quad::is_leaf, "Check if the quad is a leaf.")
        .def("compute_area", &Quad::compute_area, "Compute the area of the quad.")
        .def("split", &Quad::split, "Split the quad into child quads.")
        .def("get_leaf_nodes", &Quad::get_leaf_nodes, "Get the leaf nodes of the quad.");

    m.def("color_from_histogram", &color_from_histogram, "Calculate color and luminance from histogram.");
    m.def("weighted_average", &weighted_average, "Calculate the weighted average.");
    m.def("calculate_histogram_cv", &calculate_histogram_cv, "Calculate the histogram of an image.");
    m.def("cropImage", &cropImage, "Crop an image based on a given box.");
    m.def("cpp_callback1", &cpp_callback1, "A callback function");
    
    
}
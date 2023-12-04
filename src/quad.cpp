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
#include <vector>
#include <cstdint>


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
// Function to calculate color and luminance from histogram
extern "C" std::tuple<std::tuple<uint8_t, uint8_t, uint8_t>, double> color_from_histogram(const std::vector<int>& hist) {
    // Calculate weighted averages for each channel
    std::tuple<uint8_t, double> red_average = weighted_average(std::vector<double>(hist.begin(), hist.begin() + 256));
    std::tuple<uint8_t, double> green_average = weighted_average(std::vector<double>(hist.begin() + 256, hist.begin() + 512));
    std::tuple<uint8_t, double> blue_average = weighted_average(std::vector<double>(hist.begin() + 512, hist.end()));

    // Convert float values to integers by rounding
    int r = static_cast<uint8_t>(std::round(std::get<0>(red_average)));
    int g = static_cast<uint8_t>(std::round(std::get<0>(green_average)));
    int b = static_cast<uint8_t>(std::round(std::get<0>(blue_average)));

    // Calculate luminance
    double luminance = std::get<1>(red_average) * 0.2989 +
                       std::get<1>(green_average) * 0.5870 +
                       std::get<1>(blue_average) * 0.1140;
    return std::make_tuple(std::make_tuple(r, g, b), luminance);
}

extern "C" std::vector<int> calculate_histogram_cv(const cv::Mat& bgr_image) {
    std::vector<int> histogram(3 * 256, 0);

    if (bgr_image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(bgr_image, channels);

        for (int y = 0; y < bgr_image.rows; ++y) {
            for (int x = 0; x < bgr_image.cols; ++x) {
                for (int i = 0; i < 3; ++i) {
                    uchar pixel_value = channels[i].at<uchar>(y, x);
                    if (i == 0) {
                        // red
                        histogram[2 * 256 + pixel_value]++;
                    } else if (i == 2) {
                        // blue
                        histogram[0 * 256 + pixel_value]++;
                    } else {
                        // green
                        histogram[1 * 256 + pixel_value]++;
                    }
                }
            }
        }
    }

    return histogram;
}




//Cpp version crop(image, box)
cv::Mat cropImage(const cv::Mat& originalImage, const std::tuple<double, double, double, double>  box) {
    // Extract values from the tuple
    double x, y, width, height;
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
                            error_sum(root->m_error * root->m_area) 
{ 
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
}



std::vector<Quad> Model::getQuads() const {
    // 將優先級隊列轉換為向量
    std::vector<Quad> quads_vector = convertPriorityQueueToVector(heap);
    return quads_vector;
}

double Model::averageError() const {  return error_sum / (width * height);}

void Model::push(Quad& quad) {
    double score = -quad.m_error * std::pow(quad.m_area, AREA_POWER);
    heap.push(std::make_tuple(quad.m_leaf, score, quad));

    // Reconstruct the heap to maintain the min heap property
    auto heapCopy = heap;  // Copy the priority_queue
    std::vector<std::tuple<int, double, Quad>> heapVec;
    
    while (!heapCopy.empty()) {
        heapVec.push_back(heapCopy.top());
        heapCopy.pop();
    }

    std::make_heap(heapVec.begin(), heapVec.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // Compare elements based on score
    });

    // Reconstruct the heap from the updated container
    heap = decltype(heap)(heapVec.begin(), heapVec.end());
}

Quad Model::pop() {
    // Reconstruct the heap to maintain the min heap property
    auto heapCopy = heap;  // Copy the priority_queue
    std::vector<std::tuple<int, double, Quad>> heapVec;

    while (!heapCopy.empty()) {
        heapVec.push_back(heapCopy.top());
        heapCopy.pop();
    }

    std::pop_heap(heapVec.begin(), heapVec.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // Compare elements based on score
    });

    Quad quad = std::get<2>(heapVec.back());

    // Remove the top element from the heapCopy
    heapVec.pop_back();

    std::sort_heap(heapVec.begin(), heapVec.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // Compare elements based on score
    });

    // Reconstruct the heap from the updated container
    heap = decltype(heap)(heapVec.begin(), heapVec.end());

    return quad;
}

/* End of Model implementation*/ 

// Implement of Quad
Quad::Quad(Model& model, std::tuple<double, double, double, double> box, int depth)
    : m_model(&model), m_box(box), m_leaf(is_leaf()), hist(calculate_histogram_cv(cropImage(m_model->im, m_box))), m_depth(depth), m_area(compute_area()), children() {

    auto result = color_from_histogram(hist);
    
    // Unpack the tuple into m_color and m_error
    std::tie(m_color, m_error) = result;

}

bool Quad::is_leaf() const{
    double x, y, width, height;
    std::tie(x, y, width, height) = m_box;
    return (width  <= LEAF_SIZE || height <= LEAF_SIZE); //width and height
}

double Quad::compute_area() {
    int x, y, width, height;    //(x, y) is left-up
    std::tie(x, y, width, height) = m_box;
    return static_cast<double>(width * height);
}

// Quad::split() implementation
std::vector<Quad*> Quad::split() {
    double x, y, width, height;    //(x, y) is left-down
    std::tie(x, y, width, height) = m_box;  // 使用 m_box
    double newWidth = width / 2;
    double newHeight = height / 2;
    double x_mid = x + newWidth;
    double y_mid = y + newHeight;
    int depth = m_depth + 1;  // 使用 m_depth

    Quad* l_down = new Quad(*m_model, std::make_tuple(x, y, newWidth, newHeight), depth);  // 使用 m_model
    Quad* l_up = new Quad(*m_model, std::make_tuple(x, y_mid, newWidth, newHeight), depth);  // 使用 m_model
    Quad* r_down = new Quad(*m_model, std::make_tuple(x_mid, y, newWidth, newHeight), depth);  // 使用 m_model
    Quad* r_up = new Quad(*m_model, std::make_tuple(x_mid, y_mid, newWidth, newHeight), depth);  // 使用 m_model

    children = {l_down, l_up, r_down, r_up};  // Use children instead of childern
    return children;
}


PYBIND11_MODULE(quad, m) {
    m.doc() = "Quad image compressor";

    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&>(), "Constructor for the Model class.")
        .def("getQuads", &Model::getQuads, "Get the quads from the model.")
        .def("averageError", &Model::averageError, "Calculate the average error of the model.")
        .def("push", &Model::push, "Push a quad into the model.")
        .def("pop", &Model::pop, "Pop a quad from the model.")
        .def_property("error_sum", &Model::getErrorsum, &Model::setErrorSum)
        .def_property_readonly("width", &Model::getWidth)
        .def_property_readonly("height", &Model::getHeight)
        .def_property_readonly("root", &Model::getRoot);
        

    py::class_<Quad>(m, "Quad")
        .def(py::init<Model&, std::tuple<double, double, double, double>, int>(), "Constructor for the Quad class.")
        .def("is_leaf", &Quad::is_leaf, "Check if the quad is a leaf.")
        .def("compute_area", &Quad::compute_area, "Compute the area of the quad.")
        .def("split", &Quad::split, "Split the quad into child quads.")
        .def_property("children", &Quad::getChildren, &Quad::setChildren)
        .def_property_readonly("error", &Quad::getError)
        .def_property_readonly("area", &Quad::getArea)
        .def_property_readonly("depth", &Quad::getDepth)
        .def_property_readonly("color", &Quad::getColor)
        .def_property_readonly("box", &Quad::getBox);

    m.def("color_from_histogram", &color_from_histogram, "Calculate color and luminance from histogram.");
    m.def("weighted_average", &weighted_average, "Calculate the weighted average.");
    m.def("calculate_histogram_cv", &calculate_histogram_cv, "Calculate the histogram of an image.");
    m.def("cropImage", &cropImage, "Crop an image based on a given box.");

}

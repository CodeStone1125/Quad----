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


// Function to calculate weighted average
std::tuple<double, double> weighted_average(const std::vector<double>& hist) {
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
std::tuple<std::tuple<int, int, int>, double> color_from_histogram(const std::vector<double>& hist) {
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


// CPP version histogram()
std::vector<int> calculate_histogram_cv(const cv::Mat& rgb_image) {
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);

    // Initialize a vector to store histograms for each channel
    std::vector<int> hist_cv_channels;

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

    // Cast the values to integers
    std::vector<int> hist_cv_channels_int;
    for (float value : hist_cv_channels) {
        hist_cv_channels_int.push_back(static_cast<int>(value));
    }

    return hist_cv_channels_int;
}


//Cpp version crop(image, box)
cv::Mat cropImage(const cv::Mat& originalImage, const std::tuple<int, int, int, int>& box) {
    // Extract values from the tuple
    int x, y, width, height;
    std::tie(x, y, width, height) = box;

    // Create a rectangle to define the region of interest (ROI)
    cv::Rect roi(x, y, width, height);

    // Crop the image using the defined ROI
    return originalImage(roi).clone();
}

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

//Implement of Quad
Quad::Quad(Model model, std::tuple<int, int, int, int> box, int depth) {
    m_model = model;
    m_box = box;
    m_depth = depth;
    hist = calculate_histogram_cv(cropImage(m_model.im, m_box));
    std::tie(m_color, m_error) = color_from_histogram(hist);
    m_leaf = is_leaf();
    m_area = compute_area();
    children = {};
}

bool Quad::is_leaf() {
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
    int l, t, r, b;
    std::tie(l, t, r, b) = m_box;

    int lr = l + (r - l) / 2;
    int tb = t + (b - t) / 2;
    int depth = m_depth + 1;

    Quad tl(m_model, std::make_tuple(l, t, lr, tb), depth);
    Quad tr(m_model, std::make_tuple(lr, t, r, tb), depth);
    Quad bl(m_model, std::make_tuple(l, tb, lr, b), depth);
    Quad br(m_model, std::make_tuple(lr, tb, r, b), depth);

    children = { tl, tr, bl, br };
    return children;
}

std::vector<Quad> Quad::get_leaf_nodes(int max_depth) {
    if (children.empty() || (max_depth != -1 && m_depth >= max_depth)) {
        return { *this };
    }

    std::vector<Quad> result;
    for (const auto& child : children) {
        auto child_leaves = child.get_leaf_nodes(max_depth);
        result.insert(result.end(), child_leaves.begin(), child_leaves.end());
    }

    return result;
}


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace py = pybind11;

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

PYBIND11_MODULE(example, m) {
    m.def("calculate_histogram_cv", &calculate_histogram_cv, "Calculate histogram for an RGB image");
}

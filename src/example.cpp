#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// Function to compute histogram using cv::calcHist
cv::Mat calculateHistogram(const cv::Mat& image) {
    // Split channels for multi-channel images
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // Number of bins in the histogram
    int histSize = 256;

    // Range of pixel values
    float range[] = {0, 256};
    const float* histRange = {range};

    // Compute histogram for each channel
    std::vector<cv::Mat> histograms;
    for (const auto& channel : channels) {
        cv::Mat hist;
        cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
        histograms.push_back(hist);
    }

    // Concatenate histograms for all channels
    cv::Mat resultHist;
    cv::vconcat(histograms, resultHist);

    return resultHist;
}


void process_image(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        throw std::runtime_error("Error: Unable to read the image from " + imagePath);
    }

    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    // Get the image size
    int width = image.cols;
    int height = image.rows;
    printf("Width: %d\n", width);
    printf("Height: %d\n", height);

    // Create empty heap
    std::vector<int> heap;

    // Self.root = Quad(self, (0, 0, self.width, self.height), 0)
    // root = Quad(0, 0, width, height);
    //
    // self.error_sum = self.root.error * self.root.area
    // self.push(self.root)
    // You can return or use 'rgbImage' as needed
    cv::imshow("Original Image", image);
    cv::imshow("RGB Image", rgbImage);
    
    cv::waitKey(0);
}


namespace py = pybind11;

cv::Mat calculateHistogramFromNumpy(py::array_t<uint8_t> image) {
    py::buffer_info buf_info = image.request();
    size_t rows = buf_info.shape[0];
    size_t cols = buf_info.shape[1];
    size_t channels = buf_info.shape[2];

    cv::Mat cv_image(rows, cols, CV_8UC3, buf_info.ptr);

    return calculateHistogram(cv_image);
}


PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("process_image", &process_image, "Load image, convert to RGB, and display");
     m.def("calculate_histogram", &calculateHistogramFromNumpy, "Calculate histogram of an image");
}
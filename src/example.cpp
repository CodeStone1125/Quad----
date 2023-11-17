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

py::array_t<uint8_t> flipcvMat(py::array_t<uint8_t>& img)
{
    auto rows = static_cast<size_t>(img.shape(0));
    auto cols = static_cast<size_t>(img.shape(1));
    auto channels = static_cast<size_t>(img.shape(2));
    std::cout << "rows: " << rows << " cols: " << cols << " channels: " << channels << std::endl;

    auto type = CV_8UC3;

    cv::Mat cvimg2(rows, cols, type, (unsigned char*)img.data());

    // Use the full path or a path relative to the current working directory
    cv::imwrite("assets/test.jpg", cvimg2);

    cv::Mat cvimg3(rows, cols, type);
    cv::flip(cvimg2, cvimg3, 0);

    // Use the full path or a path relative to the current working directory
    cv::imwrite("assets/testout.jpg", cvimg3);

    py::array_t<uint8_t> output(
        py::buffer_info(
            cvimg3.data,
            sizeof(uint8_t), // itemsize
            py::format_descriptor<uint8_t>::format(),
            3, // ndim
            std::vector<size_t>{rows, cols, 3}, // shape
            std::vector<size_t>{sizeof(uint8_t) * cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)} // strides
        )
    );
    return output;
}


PYBIND11_MODULE(example, m) {
    m.def("flipcvMat", &flipcvMat, "Calculate histogram for an RGB image");
}

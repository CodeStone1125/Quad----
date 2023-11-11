#include <pybind11/pybind11.h>

int add() {
    return 2;
}

PYBIND11_MODULE(quad, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}